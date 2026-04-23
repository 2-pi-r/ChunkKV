# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math

from transformers.models.llama.modeling_llama import repeat_kv, rotate_half
from kvpress.utils import get_prerope_query_states

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.snapkv_press import SnapKVPress

# import matplotlib.pyplot as plt
# import os

@dataclass
class VariableChunkKVPress3(BasePress):
    """
    ChunkKV 개선 연구 (v3)
    """

    press: ScorerPress
    chunking_window: int = 3 # 고정 경계를 기준으로 ±3 토큰을 봄.
    chunk_length: int = 10 # 고정 경계 

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "VariableChunkKVPress requires a ScorerPress as input"

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value
    
    @staticmethod
    def get_postrope_queries(module, hidden_states, position_embeddings):
        query_states = get_prerope_query_states(module, hidden_states)
        cos, sin = position_embeddings
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1)) # Apply RoPE
        return query_states


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

        bsz, num_kv_heads, kv_len, head_dim = keys.shape

        # 1. Calculate global scores first
        global_scores = self.press.score( # (batch_size, num_kv_heads, seq_len)
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs,
        )
        global_scores = global_scores.sum(dim=1) # head 차원으로 sum  # (batch_size, seq_len)


        # 2. Divide sequence into chunks
        queries = self.get_postrope_queries(module, hidden_states, kwargs["position_embeddings"])[0] # batch_idx=0
        num_heads = queries.shape[0]

        fixed_boundaries = torch.arange(self.chunk_length, kv_len, self.chunk_length, device=keys.device) # (num_boundaries) # 예: kv_len=100, chunk_len=10일 때 [10, 20, 30, 40, 50, 60, 70, 80, 90] --> 0, 100 제외한 나머지 경계
        num_boundaries = fixed_boundaries.shape[0]

        offsets = torch.arange(-self.chunking_window, self.chunking_window + 1, device=keys.device) # (window_width)
        window_width = offsets.shape[0] # 예: [-3, -2, -1, 0, ..., 3] --> (7)
        candidate_idx = (fixed_boundaries.view(-1, 1) + offsets.view(1, -1)).clamp(1, kv_len - 1) # (num_boundaries, window_width) # 각 고정 경계 주변의 후보 인덱스 # .clamp()로 idx가 시퀀스 범위 벗어나지 않게 예외처리

        # 2-1. 후보 토큰의 Query (q_candidate) 구하기
        q_idx = candidate_idx.view(1, num_boundaries, window_width, 1).expand(num_heads, -1, -1, head_dim) # candidate_idx의 shape을 확장
        q_candidate = queries.unsqueeze(1).expand(-1, num_boundaries, -1, -1).gather(2, q_idx) # (num_heads, num_boundaries, window_width, head_dim)

        # 2-2. 앞 토큰들의 Key 평균 (k_interval_mean) 구하기 --> [chunk_starts : candidate_idx] 범위
        chunk_starts = (fixed_boundaries - self.chunk_length).view(-1, 1).expand(-1, window_width) # (num_boundaries, window_width)

        key_cumsum = torch.cumsum(keys[0].float(), dim=1) # 누적합 (청크 내 평균 Key를 빠르게 구하기 위함) # (num_kv_heads, kv_len, head_dim)
        def get_k_cumsum_at_idx(cumsum, idx):
            expanded_idx = idx.view(1, num_boundaries, window_width, 1).expand(num_kv_heads, -1, -1, head_dim) # (num_kv_heads, num_boundaries, window_width, head_dim)로 확장
            return cumsum.unsqueeze(1).expand(-1, num_boundaries, -1, -1).gather(2, expanded_idx)
        
        k_candidates = get_k_cumsum_at_idx(key_cumsum, (candidate_idx - 1).clamp(min=0)) # 후보 토큰 직전까지
        k_starts = get_k_cumsum_at_idx(key_cumsum, (chunk_starts - 1).clamp(min=0)) # 청크 시작부터
        k_starts = torch.where(chunk_starts.unsqueeze(0).unsqueeze(-1).expand_as(k_starts) > 0, k_starts, 0.0) # chunk_starts가 0인 지점의 k_starts를 0으로 세팅 

        interval_len = (candidate_idx - chunk_starts).clamp(min=1).unsqueeze(0).unsqueeze(-1)
        k_interval_mean = ((k_candidates - k_starts) / interval_len.float()).to(keys.dtype) # (num_heads, num_boundaries, window_width, head_dim)

        # 2-3. QK^T로 Similarity 계산
        if num_heads != num_kv_heads: # custom repeat_kv
            n_rep = num_heads // num_kv_heads
            k_interval_mean = ( 
                k_interval_mean[:, None, :, :, :]
                .expand(num_kv_heads, n_rep, num_boundaries, window_width, head_dim)
                .reshape(num_kv_heads * n_rep, num_boundaries, window_width, head_dim)
            )
        
        similarity = (q_candidate * k_interval_mean).sum(dim=-1) / math.sqrt(head_dim) # (num_heads, num_boundaries, window_width)
        similarity = similarity.sum(dim=0) # head 차원으로 합산                         -> (           num_boundaries, window_width)
        
        best_offsets = similarity.argmin(dim=-1) # (num_boundaries)
        final_boundaries = candidate_idx.gather(1, best_offsets.view(-1, 1)).squeeze(-1) # (num_boundaries)


        # 3. Calculate chunk_scores from scores
        chunk_ids = torch.zeros_like(global_scores[0]).scatter_(0, final_boundaries, 1).cumsum(dim=0).long() # (kv_len)
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