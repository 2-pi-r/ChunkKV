# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn
import numpy as np

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.snapkv_press import SnapKVPress

import matplotlib.pyplot as plt
import os

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
    max_chunk_size: int = 40
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
        seed_indices = top_tokens.indices[0] #.sort()[0]

        # attentions : torch.Tensor: Attention weights from the layer with shape (batch_size, num_heads, seq_len, seq_len) # weight니까 정규화된 거겠지?
        chunk_boundary = [] # 2차원 list of [start_idx, end_idx]
        raw_chunk_boundary = []

        for seed_idx_tensor in seed_indices:
            seed_idx = seed_idx_tensor.item()

            start_idx = max(0, seed_idx - self.max_chunk_size // 2)
            end_idx = min(kv_len, seed_idx + 1 + self.max_chunk_size // 2)
            # Get and Update start_idx
            for j in range(seed_idx - 1, start_idx - 1, -1): # 앞 토큰들 (seed_idx 미만, start_idx 이상)
                if attentions[0, :, seed_idx, j].mean() < self.threshold:
                    start_idx = j + 1 # threshold 못 넘기 직전까지 청킹
                    break
            # Get and Update end_idx
            for j in range(seed_idx + 1, end_idx): # 뒤 토큰들
                if attentions[0, :, j, seed_idx].mean() < self.threshold:
                    end_idx = j # threshold 못 넘기 직전까지 청킹
                    break
            
            raw_chunk_boundary.append([start_idx, end_idx])
        
        # Merge overlapped chunks
        raw_chunk_boundary.sort(key=lambda x: x[0]) # start_idx 기준 오름차순 정렬
        for start_idx, end_idx in raw_chunk_boundary:
            if len(chunk_boundary) == 0 or chunk_boundary[-1][1] <= start_idx: # if no overlap (prev_end <= start_idx)
                chunk_boundary.append([start_idx, end_idx])
            else: # if overlap
                chunk_boundary[-1][0] = min(chunk_boundary[-1][0], start_idx) # prev_start = min(prev_start, start_idx)
                chunk_boundary[-1][1] = max(chunk_boundary[-1][1], end_idx) # prev_end = max(prev_end, end_idx)


        # 3. Calculate chunk_scores from scores
        chunk_scores = torch.empty(bsz, len(chunk_boundary))
        for i in range(len(chunk_boundary)):
            curr_global_scores = global_scores[..., chunk_boundary[i][0]: chunk_boundary[i][1]] # chunk에 해당하는 score들을 가져다가
            chunk_scores[0][i] = curr_global_scores.mean(dim=-1, keepdim=True)


        # 4. Select chunks to preserve
        indices_bitmap = torch.zeros(kv_len, dtype=torch.bool, device=keys.device)
        budget = int(kv_len * (1 - self.compression_ratio)) # 남길 토큰 개수

        sorted_chunk_scores = torch.sort(chunk_scores, descending=True, dim=-1)

        for chunk_idx in sorted_chunk_scores.indices[0]:
            if(budget <= 0): break

            chunk_len = chunk_boundary[chunk_idx][1] - chunk_boundary[chunk_idx][0]
            if budget < chunk_len:
                # 청크 안에서 topk로 토큰 budget개 선택
                _, curr_indices = global_scores[0, chunk_boundary[chunk_idx][0] : chunk_boundary[chunk_idx][1]].topk(budget, dim=-1)
                indices_bitmap[chunk_boundary[chunk_idx][0] + curr_indices] = True # topk는 주어진 범위 안에서의 인덱스. 청크의 시작 주소를 더해줘야 함.
                budget = 0
                break
            indices_bitmap[chunk_boundary[chunk_idx][0] : chunk_boundary[chunk_idx][1]] = True
            budget -= chunk_len
        
       
        # print(f"---------")
        # print(f"1. chunk_boundary: {len(chunk_boundary)}\n", chunk_boundary)
        # # print(f"2. seed_indices: {seed_indices.shape}\n ", seed_indices)
        # # print(f"3. chunk_scores: {chunk_scores.shape}\n", chunk_scores)
        # print(f"4. remaining budget: {budget}")


        if budget > 0:
            global_scores[0, indices_bitmap] = float('-inf')
            curr_indices = global_scores[0].topk(budget, dim=-1).indices
            indices_bitmap[curr_indices] = True

            # 뒤에서부터 아직 선택되지 않은 토큰 budget개 선택
            # curr_indices = torch.where(~indices_bitmap)[0][-budget:]
            # indices_bitmap[curr_indices] = True

        indices = torch.where(indices_bitmap)[0].sort()[0]
        indices = indices.view(1, 1, -1, 1).expand(keys.shape[0], keys.shape[1], -1, module.head_dim)
        
        # print(f"4. indices: {indices.shape}\n", indices)
        # print(f"5. indices_bitmap: {indices_bitmap.shape}\n", indices_bitmap)
        # print(f"6. attentions: {attentions.shape}\n")

        # 5. Use gather to collect selected keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        layer_idx = getattr(module, "layer_idx", "unknown")
        unique_seeds = []
        for s in seed_indices:
            s_val = s.item()
            if all(abs(s_val - existing) > 100 for existing in unique_seeds):
                unique_seeds.append(s_val)
            if len(unique_seeds) >= 5: # 대표 시드 5개만 확보
                break
        unique_seeds.sort()
        if layer_idx in [2, 14, 25]: # 첫 번째, 중간, 마지막 레이어 샘플링
            self._visualize_attention_distribution(attentions, unique_seeds, layer_idx, kv_len)

        return keys, values
    
    def _visualize_attention_distribution(self, attentions, unique_seeds, layer_idx, kv_len):
        # 1. 특정 Seed 하나를 골라 주변 어텐션 분포 확인 (Decay 관찰)
        window_size = 100 # max_chunk_size보다 넓게 설정
        plt.figure(figsize=(10, 6))
        
        for i, target_seed in enumerate(unique_seeds):
            start = max(0, target_seed - window_size // 2)
            end = min(kv_len, target_seed + window_size // 2)
            
            # 해당 시드의 주변 어텐션 값 (헤드 평균)
            backward_dist = attentions[0, :, target_seed, start : target_seed + 1].mean(dim=0)
            forward_dist = attentions[0, :, target_seed + 1 : end, target_seed].mean(dim=0)
            around_attn = torch.cat([backward_dist, forward_dist]).cpu().float().numpy() # shape: (window_range,)

            # 3X축을 시드 중심의 상대적 거리로 설정 (-50 ~ +50)
            rel_x = np.arange(start - target_seed, end - target_seed)
            
            # 개별 시드 그래프 (겹쳐 보이도록 투명도 조정)
            plt.plot(rel_x, around_attn, alpha=0.4, label=f'Seed {target_seed}' if i < 5 else "")

        plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        
        # 0.01, 0.001 등 후보 threshold도 같이 그려보면 분석에 도움됩니다
        # plt.axhline(y=0.01, color='gray', linestyle='--')
        # plt.axhline(y=0.001, color='gray', linestyle='--')
        
        plt.title(f"Layer {layer_idx} - Overlaid Attention Decay (Log Scale)")
        plt.xlabel("Relative Token Index from Seed")
        plt.ylabel("Attention Weight to Seed")
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend(loc='upper right', fontsize='small')
        
        os.makedirs("attn_plots", exist_ok=True)
        plt.savefig(f"attn_plots/layer_{layer_idx}_overlaid_seeds.png")
        plt.close()