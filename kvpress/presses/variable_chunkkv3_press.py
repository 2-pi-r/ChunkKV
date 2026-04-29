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

from transformers import AutoTokenizer # 관찰: 경계 토큰 출력

# import matplotlib.pyplot as plt
# import os

@dataclass
class VariableChunkKVPress3(BasePress):
    """
    ChunkKV 개선 연구 (v3)
    """

    press: ScorerPress
    chunking_window: int = 5 # 고정 경계를 기준으로 ± window 토큰을 봄.
    chunk_length: int = 10 # 고정 경계 

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "VariableChunkKVPress requires a ScorerPress as input"

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path) # 관찰: 경계 토큰 출력


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
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        query_states = (query_states * cos) + (rotate_half(query_states) * sin) # Apply RoPE
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
        # pos_emb = [p.to(keys.device) for p in kwargs["position_embeddings"]]

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
        queries = self.get_postrope_queries(module, hidden_states, kwargs["position_embeddings"])[0] # batch_idx=0
        num_heads = queries.shape[0]

        fixed_boundaries = torch.arange(self.chunk_length, kv_len, self.chunk_length, device=keys.device) # (num_boundaries) # 예: kv_len=100, chunk_len=10일 때 [10, 20, 30, 40, 50, 60, 70, 80, 90] --> 0, 100 제외한 나머지 경계
        num_boundaries = fixed_boundaries.shape[0]

        offsets = torch.arange(-self.chunking_window, self.chunking_window + 1, device=keys.device) # (window_width)
        window_width = offsets.shape[0] # 예: [-3, -2, -1, 0, ..., 3] --> (7)
        candidate_idx = (fixed_boundaries.view(-1, 1) + offsets.view(1, -1)).clamp(1, kv_len - 1) # (num_boundaries, window_width) # 각 고정 경계 주변의 후보 인덱스 # .clamp()로 idx가 시퀀스 범위 벗어나지 않게 예외처리

        # 2-1. 후보 토큰의 Query (q_candidate) 구하기
        q_idx_flat = candidate_idx.view(-1)
        q_candidate = queries[:, q_idx_flat, :].view(num_heads, num_boundaries, window_width, head_dim)

        # 2-2. 앞 토큰들의 Key 평균 (k_interval_mean) 구하기 --> [chunk_starts : candidate_idx] 범위
        chunk_starts = (fixed_boundaries - self.chunk_length).view(-1, 1).expand(-1, window_width) # (num_boundaries, window_width)

        key_cumsum = torch.cumsum(keys[0].float(), dim=1) # 누적합 (청크 내 평균 Key를 빠르게 구하기 위함) # (num_kv_heads, kv_len, head_dim)
        k_cand_idx = (candidate_idx - 1).clamp(min=0).view(-1)
        k_start_idx = (chunk_starts - 1).clamp(min=0).view(-1)
        k_candidates = key_cumsum[:, k_cand_idx, :].view(num_kv_heads, num_boundaries, window_width, head_dim)
        k_starts = key_cumsum[:, k_start_idx, :].view(num_kv_heads, num_boundaries, window_width, head_dim)
        k_starts = k_starts * (chunk_starts.view(1, num_boundaries, window_width, 1) > 0)

        interval_len = (candidate_idx - chunk_starts).clamp(min=1).view(1, num_boundaries, window_width, 1)
        k_interval_mean = ((k_candidates - k_starts) / interval_len.float()).to(keys.dtype) # (num_heads, num_boundaries, window_width, head_dim)

        # 2-3. QK^T로 Similarity 계산
        if num_heads != num_kv_heads: # custom repeat_kv
            n_rep = num_heads // num_kv_heads
            k_interval_mean = k_interval_mean.repeat_interleave(n_rep, dim=0)
        
        similarity = (q_candidate * k_interval_mean).sum(dim=-1) / math.sqrt(head_dim) # (num_heads, num_boundaries, window_width)
        similarity = similarity.sum(dim=0) # head 차원으로 합산                         -> (           num_boundaries, window_width)
        
        # --- [수정 및 추가] Linguistic Constraint 필터링 ---
        # 1. 후보 인덱스의 토큰 ID들을 가져와 문자열로 변환
        flat_cand_idx = candidate_idx.view(-1)
        print(self.input_ids.shape)
        cand_ids = self.input_ids[0,flat_cand_idx].tolist()
        cand_tokens = self.tokenizer.convert_ids_to_tokens(cand_ids)

        # 2. 유효한 경계 후보인지 판단하는 마스크 생성
        # 조건: ' ' (U+2581)로 시작하거나, 알파벳/숫자가 아닌 특수문자/구두점인 경우
        valid_mask = torch.tensor(
            [(t.startswith(' ') or not t.replace(' ', '').isalnum()) for t in cand_tokens],
            device=similarity.device, dtype=torch.bool
        ).view(num_boundaries, window_width)

        # 3. 단어 중간 파편(valid_mask가 False인 곳)에 아주 큰 페널티 부여
        # float('inf')를 쓰면 모든 후보가 무효할 때 에러가 날 수 있으므로 충분히 큰 값 사용
        similarity.masked_fill_(~valid_mask, 1e9)
        # -----------------------------------------------

        best_offsets = similarity.argmin(dim=-1) # (num_boundaries)
        final_boundaries = candidate_idx.gather(1, best_offsets.view(-1, 1)).squeeze(-1) # (num_boundaries)

        # --- 관찰: 경계 토큰 출력 (디버깅용) ---
        print(f"\n{'='*30} Boundary Analysis {'='*30}")
        
        for i, b_idx in enumerate(final_boundaries):
            idx = b_idx.item()
            min_score = similarity[i, best_offsets[i]].item()
            avg_win_score = similarity[i].mean().item()
            
            # 토큰 텍스트 추출 (tokenizer가 전역 변수나 self에 있다고 가정)
            token_id = self.input_ids[0, idx].item()
            token_text = self.tokenizer.decode([token_id]) 

            # 결과 출력: 경계 번호, 인덱스, 토큰, 스코어(최소값 vs 창 평균)
            print(f"Boundary {i+1:2d} | Index: {idx:4d} | Token: ({token_text}) ID:{token_id} | "
                  f"Score: {min_score:.4f} (Win Avg: {avg_win_score:.4f})")
        
        print(f"{'='*79}\n")
        # --- 관찰 코드 끝 ---

        # 3. Calculate chunk_scores from scores
        chunk_ids = torch.zeros(kv_len, device=keys.device, dtype=torch.long)
        chunk_ids = chunk_ids.scatter_(0, final_boundaries, 1).cumsum(dim=0) # (kv_len)
        num_chunks = chunk_ids[-1] + 1

        mean_global_scores = global_scores[0].mean(dim=0)
        chunk_scores  = torch.zeros(num_chunks, device=keys.device, dtype=mean_global_scores.dtype)
        chunk_counts  = torch.zeros(num_chunks, device=keys.device, dtype=torch.long)
        chunk_scores.index_add_(0, chunk_ids, mean_global_scores)
        chunk_counts.index_add_(0, chunk_ids, torch.ones(kv_len, device=keys.device, dtype=torch.long))
        chunk_scores = chunk_scores / chunk_counts.clamp(min=1).to(chunk_scores.dtype)


        # 4. Select chunks to preserve
        budget = int(kv_len * (1 - self.compression_ratio)) # 남길 토큰 개수
        chunk_scores, chunk_indices = torch.sort(chunk_scores, descending=True) # (num_chunks,)
    
        chunk_lengths = torch.bincount(chunk_ids)[chunk_indices] # (num_chunks,) # chunk_indices 순서대로 bincount 결과를 재배치
        cumulative_lengths = torch.cumsum(chunk_lengths, dim=0)  # (num_chunks,)
        num_complete_chunks = torch.searchsorted(cumulative_lengths, budget, right=True).item() # int
        remaining_budget = budget if num_complete_chunks == 0 else budget - cumulative_lengths[num_complete_chunks - 1].item() # int

        # indices에 반영하기
        indices = []
        if num_complete_chunks > 0: # complete_chunks 처리
            chunk_rank = torch.empty_like(chunk_indices)
            chunk_rank[chunk_indices] = torch.arange(num_chunks, device=keys.device)
            mask = chunk_rank[chunk_ids] < num_complete_chunks
            indices.append(torch.where(mask)[0])

        if remaining_budget > 0: # remaining_budget 처리: 다음 순위 청크에서 global_scores 기반 topk 토큰 선택
            next_chunk_id = chunk_indices[num_complete_chunks]
            token_indices = torch.where(chunk_ids == next_chunk_id)[0] # next_chunk에 속하는 토큰들 인덱스
            token_scores = mean_global_scores[token_indices]
            _, relative_idx = token_scores.topk(remaining_budget) # 상대 인덱스. 즉 next_chunk 안에서 몇 번째 토큰인지
            indices.append(token_indices[relative_idx])

        # 이후 처리
        indices = torch.cat(indices).sort()[0]
        indices = indices.view(1, 1, -1, 1).expand(bsz, num_kv_heads, -1, head_dim)

        return keys.gather(2, indices).contiguous(), values.gather(2, indices).contiguous() # return key, value