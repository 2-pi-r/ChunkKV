# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn
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
    threshold: float = 0.001 # 0.0-1.0
    chunking_window_size: int = 20
    fixed_chunk_length: int = 10
    seed_ratio: float = 0.05 # num_seeds = kv_len * ratio

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
    def compute_seeds_attention(module, hidden_states, keys, chunking_window_size, seed_indices, position_embeddings):
        """
        seed_indices를 받아서, 예를 들어 window=20이라면 q는 [s, s+10], k는 [s-10, s+10] 범위로 attention weights 계산.
        """
        bsz, _, k_len, _ = keys.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads
        
        num_seeds = seed_indices.shape[0]

        half_win = chunking_window_size // 2
        q_win_len = 1 + half_win # [s, s+half_win]
        k_win_len = 1 + chunking_window_size # [s-half_win, s+half_win]

        # 1. Prepare Full States
        query_states_raw = get_prerope_query_states(module, hidden_states)
        cos, sin = position_embeddings
        key_states = repeat_kv(keys, num_key_value_groups)

        # 2. Generate Relative Indices for Gathering
        q_offsets = torch.arange(0, q_win_len, device=seed_indices.device) # [0, 1, ..., half_win]
        k_offsets = torch.arange(-half_win, half_win + 1, device=seed_indices.device) # [-half_win, ..., half_win]

        # absolute indices: (num_seeds, win_len)
        q_idx = (seed_indices.unsqueeze(1) + q_offsets).clamp(0, k_len - 1) # q_idx: (num_seeds, q_win_len) # seed_indices: (num_seeds,) -> (num_seeds, 1) # q_offsets: (q_win_len,)
        k_idx = (seed_indices.unsqueeze(1) + k_offsets).clamp(0, k_len - 1)

        # 3. Gather States
        # (bsz, num_heads, k_len, head_dim) -> (bsz, num_heads, num_seeds, win_len, head_dim)
        def gather_states(states, idx):
            # Index expansion for gather
            idx = idx.view(1, 1, num_seeds, -1, 1).expand(bsz, num_heads, -1, -1, head_dim)
            return torch.gather(states.unsqueeze(2).expand(-1, -1, num_seeds, -1, -1), 3, idx)

        q_sliced = gather_states(query_states_raw, q_idx) # (bsz, num_heads, num_seeds, q_win, head_dim)
        k_sliced = gather_states(key_states, k_idx) # (bsz, num_heads, num_seeds, k_win, head_dim)

        # Apply RoPE to query
        cos_sliced = gather_states(cos.expand(bsz, num_heads, -1, -1), q_idx) # # cos, sin: (1, 1, seq_len, head_dim) -> (bsz, num_heads, num_seeds, q_win, head_dim)
        sin_sliced = gather_states(sin.expand(bsz, num_heads, -1, -1), q_idx)
        q_sliced = (q_sliced * cos_sliced) + (rotate_half(q_sliced) * sin_sliced)

        # attn_weights
        attn_weights = torch.matmul(q_sliced, k_sliced.transpose(-1, -2)) / math.sqrt(head_dim) # (bsz, num_heads, num_seeds, q_win, k_win)

        mask = q_idx.unsqueeze(-1) < k_idx.unsqueeze(-2) # Q의 절대 위치와 K의 절대 위치를 비교하여 mask 생성 # mask: (num_seeds, q_win, k_win)  # q_idx: (num_seeds, q_win)
        attn_weights = attn_weights.masked_fill(mask.view(1, 1, num_seeds, q_win_len, k_win_len), float("-inf"))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_sliced.dtype)

        return attn_weights # (bsz, num_heads, num_seeds, q_win, k_win)
        
        # 병렬연산 적용 전
        all_key_states = repeat_kv(keys, num_key_value_groups) # (bsz, num_heads, k_len, head_dim)
        cos, sin = position_embeddings

        all_attn_weights = []
        for seed in seed_indices:
            # Define ranges
            q_start = seed.item()
            q_end = min(seed.item() + 1 + self.chunking_window_size // 2, k_len)
            
            k_start = max(seed.item() - self.chunking_window_size // 2, 0)
            k_end = q_end

            # Get query_states (Apply RoPE)
            query_states = get_prerope_query_states(module, hidden_states[:, q_start:q_end])
            query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))
            # Get key_states
            key_states = all_key_states[:, :, k_start:k_end, :]

            # Compute attention # (bsz, num_heads, q_seg_len, k_seg_len)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim) 
            attention_mask = torch.ones_like(attn_weights) * float("-inf")
            attention_mask = torch.triu(attention_mask, diagonal= seed.item() - k_start + 1)
            attn_weights += attention_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            all_attn_weights.append(attn_weights)

        # all_attn_weights는 각 seed별로 (bsz, num_heads, q_win, k_win) 크기의 리스트임
        return all_attn_weights
    

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
        seed_attentions = self.compute_seeds_attention(
            module, hidden_states, keys, self.chunking_window_size, seed_indices, kwargs["position_embeddings"]
        ) # (bsz, num_heads, num_seeds, q_win, k_win)
        
        chunk_boundary = [] # 2차원 list of [start_idx, end_idx]
        raw_chunk_boundary = []

        half_win = self.chunking_window_size // 2

        for i in range(seed_indices.shape[0]):
            seed_idx = seed_indices[i].item()

            start_idx = max(0, seed_idx - half_win)
            end_idx = min(kv_len, seed_idx + 1 + half_win)
            
            # Get and Update start_idx
            for j in range(seed_idx - 1, start_idx - 1, -1): # 앞 토큰들 (seed_idx 미만, start_idx 이상)
                if seed_attentions[0, :, i, 0, j - start_idx].mean() < self.threshold:
                    start_idx = j + 1 # threshold 못 넘기 직전까지 청킹
                    break
            # Get and Update end_idx
            for j in range(seed_idx + 1, end_idx): # 뒤 토큰들
                if seed_attentions[0, :, i, j - seed_idx, half_win].mean() < self.threshold:
                    end_idx = j # threshold 못 넘기 직전까지 청킹
                    break
            # # Get and Update start_idx
            # for j in range(seed_idx - 1, start_idx - 1, -1): # 앞 토큰들 (seed_idx 미만, start_idx 이상)
            #     if attentions[0, :, seed_idx, j].mean() < self.threshold:
            #         start_idx = j + 1 # threshold 못 넘기 직전까지 청킹
            #         break
            # # Get and Update end_idx
            # for j in range(seed_idx + 1, end_idx): # 뒤 토큰들
            #     if attentions[0, :, j, seed_idx].mean() < self.threshold:
            #         end_idx = j # threshold 못 넘기 직전까지 청킹
            #         break
            
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

        chunk_len_sum = 0
        for boundary in chunk_boundary:
            chunk_len_sum += boundary[1] - boundary[0]

        if chunk_len_sum > budget: # 그리디로 선택
            sorted_chunk_scores = torch.sort(chunk_scores, descending=True, dim=-1)

            for chunk_id in sorted_chunk_scores.indices[0]:
                if budget <= 0: break

                chunk_len = chunk_boundary[chunk_id][1] - chunk_boundary[chunk_id][0]
                if budget >= chunk_len:
                    indices_bitmap[chunk_boundary[chunk_id][0] : chunk_boundary[chunk_id][1]] = True
                    budget -= chunk_len
                else:
                    # 청크 안에서 topk로 토큰 budget개 선택
                    _, curr_indices = global_scores[0, chunk_boundary[chunk_id][0] : chunk_boundary[chunk_id][1]].topk(budget, dim=-1)
                    indices_bitmap[chunk_boundary[chunk_id][0] + curr_indices] = True # topk는 주어진 범위 안에서의 인덱스. 청크의 시작 주소를 더해줘야 함.
                    budget = 0


        else: # 모든 가변길이 청크 선택
            for boundary in chunk_boundary:
                indices_bitmap[boundary[0]:boundary[1]] = True
            budget -= chunk_len_sum
            # print(f"---------")
            # print(f"모든 가변길이 청크 선택: {budget}")

        # print(f"---------")
        # print(f"1. chunk_boundary: {len(chunk_boundary)}\n", chunk_boundary)
        # # print(f"2. seed_indices: {seed_indices.shape}\n ", seed_indices)
        # # print(f"3. chunk_scores: {chunk_scores.shape}\n", chunk_scores)
        # print(f"4. remaining budget: {budget}, chunk_len_sum: {chunk_len_sum}")

        if budget > 0: # 나머지 토큰들로 budget 채우기
            # 방법3: 선택되지 않은 토큰을 고정 길이 청크로 분할해서 topk
            
            fixed_chunks = [] # 아직 선택되지 않은 토큰들의 평균 점수 계산 (선택된 토큰은 제외하고 계산)
            for i in range(kv_len // self.fixed_chunk_length):
                start_idx = i * self.fixed_chunk_length
                end_idx = min(kv_len, start_idx + self.fixed_chunk_length)

                mask = ~indices_bitmap[start_idx:end_idx]
                if mask.any():
                    chunk_score = global_scores[0, start_idx:end_idx][mask].mean().item()
                    fixed_chunks.append(chunk_score)
            
            # 점수 순으로 정렬 후 budget 소진 시까지 할당
            fixed_chunks.sort(reverse=True)
            # print(f"나머지 토큰들 채우기 - fixed_chunks: {len(fixed_chunks)}\n{fixed_chunks}")

            for i in range(len(fixed_chunks)):
                if budget <= 0: break
                
                start_idx = i * self.fixed_chunk_length
                end_idx = min(kv_len, start_idx + self.fixed_chunk_length)

                # 실제 새로 추가될 토큰들만 bitmap에 반영
                mask = ~indices_bitmap[start_idx:end_idx]
                actual_to_add = mask.sum().item()
                
                if budget >= actual_to_add:
                    indices_bitmap[start_idx:end_idx] = True
                    budget -= actual_to_add
                else:
                    # 마지막 남은 budget은 해당 고정 청크 내에서 top-k
                    _, sub_idx = global_scores[0, start_idx:end_idx][mask].topk(budget)
                    # 실제 인덱스 매핑 (mask가 True인 위치들 중 top-k)
                    target_indices = torch.where(mask)[0][sub_idx]
                    indices_bitmap[start_idx + target_indices] = True
                    budget = 0

            # # 방법2: 선택되지 않은 토큰 중 topk
            # global_scores[0, indices_bitmap] = float('-inf')
            # curr_indices = global_scores[0].topk(budget, dim=-1).indices
            # indices_bitmap[curr_indices] = True

            # 방법1: 뒤에서부터 아직 선택되지 않은 토큰 budget개 선택
            # curr_indices = torch.where(~indices_bitmap)[0][-budget:]
            # indices_bitmap[curr_indices] = True

        indices = torch.where(indices_bitmap)[0].sort()[0]
        indices = indices.view(1, 1, -1, 1).expand(keys.shape[0], keys.shape[1], -1, module.head_dim)
        
        # print(f"4. indices: {indices.shape}\n", indices)
        # print(f"5. indices_bitmap: {indices_bitmap.shape}\n", indices_bitmap)
        # print(f"6. attentions: {attentions.shape}\n")
        # print(f"7. budget: {budg et}")

        # 5. Use gather to collect selected keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        # # 중심토큰 주변 attention 그래프 출력
        # layer_idx = getattr(module, "layer_idx", "unknown")
        # unique_seeds = []
        # for s in seed_indices:
        #     s_val = s.item()
        #     if all(abs(s_val - existing) > 100 for existing in unique_seeds):
        #         unique_seeds.append(s_val)
        #     if len(unique_seeds) >= 5: # 대표 시드 5개만 확보
        #         break
        # unique_seeds.sort()
        # if layer_idx in [2, 14, 25]: # 첫 번째, 중간, 마지막 레이어 샘플링
        #     self._visualize_attention_distribution(attentions, unique_seeds, layer_idx, kv_len)

        return keys, values
    
    def _visualize_attention_distribution(self, attentions, unique_seeds, layer_idx, kv_len):
        # 1. 특정 Seed 하나를 골라 주변 어텐션 분포 확인 (Decay 관찰)
        window_size = 100 # chunking_window_size보다 넓게 설정
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