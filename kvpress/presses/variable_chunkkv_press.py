# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.snapkv_press import SnapKVPress


@dataclass
class VariableChunkKVPress(BasePress):
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
    threshold: float # 0.0-1.0
    max_chunk_size: int = 20
    seed_ratio: float = 0.05 # num_seeds = kv_len * ratio

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "ChunkKVPress requires a ScorerPress as input"

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

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

        assert attentions is not None, "VariableChunkKV needs attentions."

        kv_len = keys.shape[2]
        bsz = keys.shape[0]

        num_seeds = int(kv_len * self.seed_ratio)

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
        top_tokens = global_scores.topk(num_seeds, dim=-1) # seeds for clustering    # top_tokens.indices: (batch_size, num_seeds)
        top_tokens_indices = top_tokens.indices[0].sort()[0]

        # attentions : torch.Tensor: Attention weights from the layer with shape (batch_size, num_heads, seq_len, seq_len) # weight니까 정규화된 거겠지?
        # print(kv_len)
        chunk_boundary = [] # 2차원 list of [start_idx, end_idx]

        # if isinstance(self.press, SnapKVPress):
        #     chunk_boundary.append([kv_len - self.press.window_size, kv_len])
    
        for seed_idx in top_tokens_indices:
            # print(f"[{chunk_id}, {seed_idx}]")
            start_idx = max(0, seed_idx - self.max_chunk_size // 2)
            end_idx = min(kv_len, seed_idx + 1 + self.max_chunk_size // 2)

            # Get and Update start_idx
            for j in range(seed_idx - 1, start_idx - 1, -1): # 앞 토큰들 (seed_idx 미만, start_idx 이상)
                # print(f"{j} ", end="")
                if attentions[0, :, seed_idx, j].mean() < self.threshold:
                    start_idx = j + 1 # threshold 못 넘기 직전까지 청킹
                    # print(f" << start!", end="")
                    break
            # print("")
            
            # Get and Update end_idx
            for j in range(seed_idx + 1, end_idx): # 뒤 토큰들
                # print(f"{j} ", end="")
                if attentions[0, :, j, seed_idx].mean() < self.threshold:
                    # print(f" << end!", end="")
                    end_idx = j # threshold 못 넘기 직전까지 청킹
                    break

            # Apply them to chunk_boundary
            if len(chunk_boundary) == 0 or chunk_boundary[-1][1] < start_idx: # if no overlap (prev_end < start_idx)
                chunk_boundary.append([start_idx, end_idx])
            else: # if overlap
                chunk_boundary[-1][0] = min(chunk_boundary[-1][0], start_idx) # prev_start = min(prev_start, start_idx)
                chunk_boundary[-1][1] = max(chunk_boundary[-1][1], end_idx) # prev_end = max(prev_end, end_idx)
                
            # print("")
        

        # 3. Calculate chunk_scores from scores
        chunk_scores = torch.empty(bsz, num_seeds)
        for i in range(num_seeds):
            curr_global_scores = global_scores[..., chunk_boundary[i][0]: chunk_boundary[i][1]] # chunk에 해당하는 score들을 가져다가
            chunk_scores[i] = curr_global_scores.mean(dim=-1, keepdim=True)


        # 4. Select chunks to preserve
        indices = []
        budget = kv_len * (1 - self.compression_ratio) # 남길 토큰 개수

        sorted_chunk_scores = torch.sort(chunk_scores, descending=True, dim=-1)
        i = 0

        while(i < num_seeds and budget > 0):
            chunk_idx = sorted_chunk_scores.indices[0][i]
            chunk_len = chunk_boundary[chunk_idx][1] - chunk_boundary[chunk_idx][0]
            if budget < chunk_len:
                # 청크 안에서 topk로 토큰 budget개 선택
                _, chunk_indices = global_scores[0, chunk_boundary[chunk_idx][0] : chunk_boundary[chunk_idx][1]].topk(budget, dim=-1)
                indices.append(chunk_indices)
                break
            chunk_indices = torch.arange(chunk_boundary[chunk_idx][0], chunk_boundary[chunk_idx][1], device=keys.device)
            indices.append(chunk_indices)
            budget -= chunk_len

        if budget > 0:
            pass # TODO: 뒤에서부터 토큰 budget개 선택

        indices = torch.cat(indices).sort()[0]
        indices = indices.view(1, 1, -1, 1).expand(keys.shape[0], keys.shape[1], -1, module.head_dim)

        """출력"""
        print(f"-------")
        print(f"1. chunk_boundary: ", chunk_boundary)
        print(f"2. seed_indices: {top_tokens.indices[0].shape}\n ", top_tokens.indices[0])
        print(f"3. chunk_scores: {chunk_scores.shape}\n", chunk_scores)
        print(f"4. indices: {indices.shape}\n", indices)

        # 5. Use gather to collect selected keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values