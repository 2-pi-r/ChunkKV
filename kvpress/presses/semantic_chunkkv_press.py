# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoTokenizer
from transformers.pipelines.base import GenericTensor

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.snapkv_press import SnapKVPress

@dataclass
class SemanticChunkKVPress(BasePress):
    """
    ChunkKV 개선 연구
    (설명 나중에 추가)

    Parameters
    ----------
    press : ScorerPress
        The underlying scoring method used to compute global importance scores.
    (설명 나중에 추가)
    """

    press: ScorerPress
    delimiter_set: list = field(default_factory=lambda: [".", ",", "?", "!", ";", ":", " ", "\t", "\n"])
    current_context_ids: Optional[GenericTensor] = None

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "SemanticChunkKVPress requires a ScorerPress as input"

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)
        # Tokenizer를 사용하여 delimiter characters를 token IDs로 변환
        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
        ids = []
        for d in self.delimiter_set:
            token_id = tokenizer.encode(d, add_special_tokens=False)
            if token_id:
                ids.append(token_id[-1])
        self.delimiter_ids = torch.tensor(list(set(ids)), device=model.device)

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value
    

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.press.compression_ratio == 0:
            return keys, values

        # assert attentions is not None, "VariableChunkKV needs attentions."

        assert self.current_context_ids is not None, "SemanticChunkKVPress requires context_ids."
        context_ids = self.current_context_ids.to(keys.device)

        kv_len = keys.shape[2]

        # 1. Calculate global scores first
        global_scores = self.press.score( # (batch_size, num_kv_heads, seq_len)
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs,
        )

        # 2. Divide sequence into chunks
        is_delimiter = torch.isin(context_ids[0], self.delimiter_ids.to(context_ids.device)) # (seq_len)
        chunk_ids = is_delimiter.cumsum(dim=-1) # (seq_len)
        chunk_ids = F.pad(chunk_ids, (1, -1), value=0) # Delimiter 자체를 이전 chunk에 포함시키기 위해 shift

        # 3. Calculate chunk_scores from scores
        num_chunks = chunk_ids[-1].item() + 1
        chunk_scores = torch.zeros(num_chunks, device=keys.device, dtype=global_scores.dtype) # (num_chunks,)
        chunk_scores = chunk_scores.scatter_reduce(
            dim=0, index=chunk_ids, src=global_scores[0].mean(dim=0), reduce="mean", include_self=False
        ) # src: batch 차원으로 [0]이고, head 차원으로 평균 낸 global_scores

        # 4. Select chunks to preserve
        budget = int(kv_len * (1 - self.compression_ratio)) # 남길 토큰 개수
        chunk_scores, chunk_indices = torch.sort(chunk_scores, descending=True) # (num_chunks,)
    
        chunk_lengths = torch.bincount(chunk_ids)[chunk_indices] # (num_chunks,) # chunk_indices 순서대로 bincount 결과를 재배치
        cumulative_lengths = torch.cumsum(chunk_lengths, dim=0)  # (num_chunks,)

        num_complete_chunks = torch.searchsorted(cumulative_lengths, budget).item() # int
        remaining_budget = budget if num_complete_chunks == 0 else budget - cumulative_lengths[num_complete_chunks - 1].item() # int

        # indices에 반영하기
        indices = []
        if num_complete_chunks > 0: # complete_chunks 처리
            complete_chunk_ids = chunk_indices[:num_complete_chunks] # (num_complete_chunks,)
            indices.append(torch.where(torch.isin(chunk_ids, complete_chunk_ids))[0]) # torch.isin: boolean mask 반환. chunk_ids (seq_len)의 각 요소가 complete_chunk_ids 리스트에 포함되어 있나

        if remaining_budget > 0: # remaining_budget 처리: 다음 순위 청크에서 global_scores 기반 topk 토큰 선택
            next_chunk_id = chunk_indices[num_complete_chunks]
            token_indices = torch.where(chunk_ids == next_chunk_id)[0] # next_chunk에 속하는 토큰들 인덱스
            token_scores = global_scores[0, :, token_indices].mean(dim=0)
            _, relative_idx = token_scores.topk(remaining_budget) # 상대 인덱스. 즉 next_chunk 안에서 몇 번째 토큰인지
            indices.append(token_indices[relative_idx])

        # 이후 처리
        indices = torch.cat(indices).sort()[0]
        indices = indices.view(1, 1, -1, 1).expand(keys.shape[0], keys.shape[1], -1, module.head_dim)

        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values