# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# Copyright 2026 Tencent Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from transformers.models.qwen3_vl.modeling_qwen3_vl (transformers 5.3.0).
# Only the Vision Encoder part is copied here.
#
# Optimizations over the upstream HuggingFace implementation:
#   1. Pre-compute max_seqlen as int ONCE before the layer loop, avoiding
#      a per-layer cudaStreamSynchronize caused by implicit GPU tensor → int
#      conversion inside flash_attn_varlen_func.
#      (27 layers × ~1.2ms sync = ~33ms saved on Qwen3-VL-32B)
#   2. torch.compile RoPE: fused Triton kernel via inductor, replacing multiple per-layer PyTorch ops.
#   3. Contiguous QKV split: .contiguous() ensures FA uses aligned memory copies.
#   4. Pre-cast cos/sin to model dtype once, avoiding per-layer implicit casts.

from collections.abc import Callable
from dataclasses import dataclass
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

try:
    from transformers.utils.generic import is_flash_attention_requested
except ImportError:
    # Fallback for transformers < 5.x.
    def is_flash_attention_requested(config):
        attn_impl = getattr(config, "_attn_implementation", None)
        return attn_impl in ("flash_attention_2", "flash_attention_3")


from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig


if os.environ.get("KLLM_DISABLE_SGL_ATTN", "0") == "1":
    _HAS_VIT_FA3 = False
    print("[I] sgl_kernel FA3 disabled via KLLM_DISABLE_SGL_ATTN=1.")
else:
    try:
        from sgl_kernel.flash_attn import flash_attn_varlen_func as _vit_fa3_varlen

        _HAS_VIT_FA3 = True
        print("[I] sgl_kernel FA3 available.")
    except ImportError:
        _HAS_VIT_FA3 = False
        print("[I] sgl_kernel not found, will use transformers attention dispatch.")


@dataclass
class BaseModelOutputWithDeepstackFeatures(BaseModelOutputWithPooling):
    deepstack_features: list[torch.FloatTensor] | None = None


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(
            x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x
        ).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.compile(dynamic=True)
def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        max_seqlen: int | None = None,  # pre-computed, avoids per-layer sync
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]

        # Contiguous split ensures FA uses aligned memory (CatArrayBatchedCopy).
        qkv = self.qkv(hidden_states).reshape(
            seq_length, 3, self.num_heads, self.head_dim
        )
        query_states = qkv[:, 0].contiguous()
        key_states = qkv[:, 1].contiguous()
        value_states = qkv[:, 2].contiguous()

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        if _HAS_VIT_FA3:
            # sgl_kernel FA3: bypass transformers dispatch, q/k/v are [seq_len, num_heads, head_dim].
            if max_seqlen is None:
                max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

            attn_output = _vit_fa3_varlen(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=False,
            )
        elif is_flash_attention_requested(self.config):
            # transformers FA3/FA2 dispatch.
            if max_seqlen is None:
                max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

            query_states = query_states.transpose(0, 1).unsqueeze(0)
            key_states = key_states.transpose(0, 1).unsqueeze(0)
            value_states = value_states.transpose(0, 1).unsqueeze(0)

            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # SDPA/eager: split by cu_seqlens.
            query_states = query_states.transpose(0, 1).unsqueeze(0)
            key_states = key_states.transpose(0, 1).unsqueeze(0)
            value_states = value_states.transpose(0, 1).unsqueeze(0)

            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2)
                for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        max_seqlen: int | None = None,  # passed through to attention
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            max_seqlen=max_seqlen,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class KsanaQwen3VLVisionModel(PreTrainedModel):
    """Qwen3-VL Vision Encoder (adapted from HuggingFace transformers).

    Key optimization: ``max_seqlen`` is computed once as a Python int before
    the layer loop, eliminating per-layer ``cudaStreamSynchronize`` overhead.
    Weight names are identical to the upstream model.
    """

    config_class = Qwen3VLVisionConfig
    _supports_flash_attn = True  # for FA3 path (_flash_attn_3_can_dispatch)
    _supports_flash_attn_2 = True
    _supports_flash_attn_3 = True
    _supports_sdpa = True

    def __init__(self, config: Qwen3VLVisionConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen3VLVisionBlock(config) for _ in range(config.depth)]
        )
        self.merger = Qwen3VLVisionPatchMerger(
            config=config, use_postshuffle_norm=False
        )

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=True)
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

        self.gradient_checkpointing = False

        # Dedicated CUDA stream for async D2H copy of deepstack features,
        # overlapping transfers with subsequent layer computation.
        self._copy_stream: torch.cuda.Stream | None = None

        self.post_init()

    # ---- Position embedding helpers ----

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()

        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = (
                block_rows[:, None, None, None] * merge_size
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size
                + intra_col[None, None, None, :]
            )

            row_idx = row_idx.expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)
            col_idx = col_idx.expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        """Bilinear interpolation of position embeddings on GPU (avoids CPU/H2D overhead)."""
        grid_thw_list = grid_thw.tolist()
        side = self.num_grid_per_side
        merge_size = self.spatial_merge_size
        hidden_dim = self.pos_embed.embedding_dim
        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype

        outputs = []
        for t, h, w in grid_thw_list:
            t, h, w = int(t), int(h), int(w)

            h_idxs = torch.linspace(0, side - 1, h, dtype=torch.float32, device=device)
            w_idxs = torch.linspace(0, side - 1, w, dtype=torch.float32, device=device)

            h_floor = h_idxs.to(torch.long)
            w_floor = w_idxs.to(torch.long)
            h_ceil = torch.clamp(h_floor + 1, max=side - 1)
            w_ceil = torch.clamp(w_floor + 1, max=side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            # Meshgrid for bilinear interpolation.
            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

            # Bilinear weights.
            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - w01

            # Flatten indices and weights for embedding lookup.
            h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            indices = (h_grid * side + w_grid).reshape(4, -1)
            weights = (
                torch.stack([w00, w01, w10, w11], dim=0)
                .reshape(4, -1, 1)
                .to(dtype=dtype)
            )

            # Weighted sum of embeddings.
            embeds = self.pos_embed(indices)
            embeds *= weights
            combined = embeds.sum(dim=0)

            # Reorganize by spatial_merge_size and tile over temporal dim.
            combined = (
                combined.reshape(
                    h // merge_size, merge_size, w // merge_size, merge_size, hidden_dim
                )
                .permute(0, 2, 1, 3, 4)
                .reshape(1, -1, hidden_dim)
            )
            repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

    # ---- Forward ----

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        on_deepstack_ready=None,
        **kwargs,
    ) -> BaseModelOutputWithDeepstackFeatures:
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        # Pre-cast cos/sin to model dtype once, avoiding per-layer implicit casts.
        position_embeddings = (
            emb.cos().to(hidden_states.dtype),
            emb.sin().to(hidden_states.dtype),
        )

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            dtype=torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # Pre-compute max_seqlen once as Python int to avoid per-layer GPU sync.
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        # Async D2H for deepstack features on a dedicated stream,
        # overlapping transfers with subsequent layer computation.
        if self.deepstack_visual_indexes:
            if self._copy_stream is None:
                self._copy_stream = torch.cuda.Stream(device=hidden_states.device)
            copy_stream = self._copy_stream
            compute_stream = torch.cuda.current_stream()
        else:
            copy_stream = None
            compute_stream = None

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                max_seqlen=max_seqlen,
            )
            if layer_num in self.deepstack_visual_indexes:
                ds_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[ds_idx](hidden_states)
                # Async D2H on separate stream to avoid blocking layer compute.
                with torch.cuda.stream(copy_stream):
                    copy_stream.wait_stream(compute_stream)
                    cpu_feature = deepstack_feature.to("cpu", non_blocking=True)
                deepstack_feature_lists.append(cpu_feature)

                if on_deepstack_ready is not None:
                    copy_stream.synchronize()
                    on_deepstack_ready(ds_idx, cpu_feature)

        if copy_stream is not None and on_deepstack_ready is None:
            copy_stream.synchronize()

        merged_hidden_states = self.merger(hidden_states)

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists,
        )
