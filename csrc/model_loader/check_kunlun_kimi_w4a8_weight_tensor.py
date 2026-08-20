# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
#
# Kunlun XPU Kimi K2.6 W4A8 model weight correctness verification script.
#
# Verifies that weights dumped from Ksana LLM engine (Kunlun XPU) match the
# original safetensors model weights after applying the same transformations
# (TP split, permute, gate/up merge, fused_lora_a, kv_b split, INT4 pack, scale*7, etc.)
#
# Usage:
#   python check_kunlun_kimi_w4a8_weight_tensor.py <model_dir> <tp_size> <ep_size> [dump_path]
#
# Platform: Kunlun XPU
# Model: Kimi K2.6 (DeepSeek-V3 architecture with W4A8_KLX quantization)

import os
import sys
import re
import json

import torch
import numpy as np
from safetensors import safe_open


def convert_uint16_bf16_to_fp32(u16_arr):
    uint32_view = u16_arr.astype(np.uint32)
    bf16_padded = uint32_view << 16
    arr_float32 = bf16_padded.view(np.float32)
    return arr_float32.copy()


def check_torch_tensor_equal(tensor_name, torch_tensor, rank, dump_path,
                             rtol=1e-5, atol=1e-8, check_dtype=None, int8_max_mismatch_ratio=0.0):
    tensor_data_file = os.path.join(dump_path, str(rank), tensor_name + ".npy")
    if not os.path.isfile(tensor_data_file):
        print("  [SKIP] Tensor file {} does not exist.".format(tensor_data_file))
        return True

    read_data_arr = np.load(tensor_data_file)

    if read_data_arr.dtype == np.uint16:
        read_data_arr = convert_uint16_bf16_to_fp32(read_data_arr)

    if isinstance(torch_tensor, torch.Tensor):
        base_data_arr = torch_tensor.to(torch.float32).numpy()
    else:
        base_data_arr = np.array(torch_tensor, dtype=np.float32)

    if read_data_arr.dtype == np.int8 or (check_dtype is not None and check_dtype == np.int8):
        if isinstance(torch_tensor, torch.Tensor):
            base_data_arr = torch_tensor.numpy().astype(np.int8)
        if not read_data_arr.shape == base_data_arr.shape:
            print("  [FAIL] Tensor {} shape mismatch: actual={}, expected={}".format(
                tensor_name, read_data_arr.shape, base_data_arr.shape))
            return False
        if not np.array_equal(read_data_arr, base_data_arr):
            diff_count = np.sum(read_data_arr != base_data_arr)
            mismatch_ratio = diff_count / read_data_arr.size
            if int8_max_mismatch_ratio > 0 and mismatch_ratio <= int8_max_mismatch_ratio:
                print("  [OK] Tensor {} approx matched (INT8 mismatch {}/{} = {:.4f}% <= {:.2f}%), shape: {}".format(
                    tensor_name, diff_count, read_data_arr.size,
                    mismatch_ratio * 100, int8_max_mismatch_ratio * 100, base_data_arr.shape))
                return True
            print("  [FAIL] Tensor {} (INT8) {} mismatched of {} ({:.4f}%)".format(
                tensor_name, diff_count, read_data_arr.size, mismatch_ratio * 100))
            return False
        print("  [OK] Tensor {} exact matched, shape: {}, dtype: int8".format(
            tensor_name, base_data_arr.shape))
        return True

    if not read_data_arr.shape == base_data_arr.shape:
        print("  [FAIL] Tensor {} shape mismatch: actual={}, expected={}".format(
            tensor_name, read_data_arr.shape, base_data_arr.shape))
        return False

    if not np.allclose(read_data_arr, base_data_arr, rtol=rtol, atol=atol):
        abs_diff = np.abs(read_data_arr - base_data_arr)
        print("  [FAIL] Tensor {} not matched. max_diff={}, mean_diff={}".format(
            tensor_name, np.max(abs_diff), np.mean(abs_diff)))
        return False

    print("  [OK] Tensor {} matched, shape: {}, dtype: {}".format(
        tensor_name, base_data_arr.shape, read_data_arr.dtype))
    return True


# ============ Global weights ============

def check_embed_tokens(state_dict, tensor_name, tp_size, rank, dump_path):
    """embed_tokens: TransSplitOptTrans(transpose=true) → permute, split axis=0, permute back.
    Original: [vocab, hidden]. After: [vocab, hidden/tp]"""
    weight = state_dict[tensor_name]
    # TransSplitOptTrans: permute → [hidden, vocab], split axis=0 → [hidden/tp, vocab], permute → [vocab, hidden/tp]
    weight_t = weight.permute(1, 0)
    slice_size = (weight_t.shape[0] + tp_size - 1) // tp_size
    sliced = weight_t[rank * slice_size:(rank + 1) * slice_size, :]
    result = sliced.permute(1, 0)
    return check_torch_tensor_equal(tensor_name, result, rank, dump_path)


def check_lm_head(state_dict, tensor_name, tp_size, rank, dump_path):
    """lm_head: SplitOptTrans(transpose=false) → split axis=0, no permute.
    Original: [vocab, hidden]. After: [vocab/tp, hidden]"""
    weight = state_dict[tensor_name]
    slice_size = (weight.shape[0] + tp_size - 1) // tp_size
    sliced = weight[rank * slice_size:(rank + 1) * slice_size, :]
    result = sliced
    return check_torch_tensor_equal(tensor_name, result, rank, dump_path)


def check_norm(state_dict, tensor_name, rank, dump_path):
    """Norm weights: no split, no permute."""
    weight = state_dict[tensor_name]
    return check_torch_tensor_equal(tensor_name, weight, rank, dump_path)


# ============ Attention weights ============

def check_fused_lora_a_proj(state_dict, layer_prefix, model_config, rank, dump_path):
    """fused_lora_a_proj = concat(q_a_proj, kv_a_proj_with_mqa), no PostProcess permute (trans_b=true).
    q_a_proj: [q_lora_rank, hidden] = [1536, 7168]
    kv_a_proj_with_mqa: [kv_lora_rank + rope, hidden] = [576, 7168]
    Result: [q_lora + kv_lora + rope, hidden] = [2112, 7168]"""
    q_a_name = layer_prefix + "self_attn.q_a_proj.weight"
    kv_a_name = layer_prefix + "self_attn.kv_a_proj_with_mqa.weight"
    if q_a_name not in state_dict or kv_a_name not in state_dict:
        return True

    q_a = state_dict[q_a_name]
    kv_a = state_dict[kv_a_name]
    fused = torch.cat([q_a, kv_a], dim=0)  # [2112, 7168]
    result = fused  # trans_b=true: no PostProcess permute

    dump_name = layer_prefix + "self_attn.fused_lora_a_proj.weight"
    return check_torch_tensor_equal(dump_name, result, rank, dump_path, check_dtype=np.int8)


def check_fused_lora_a_proj_scale(state_dict, layer_prefix, rank, dump_path):
    """fused_lora_a_proj.weight_scale = concat(q_a_proj.weight_scale, kv_a.weight_scale) * 127"""
    q_a_scale = layer_prefix + "self_attn.q_a_proj.weight_scale"
    kv_a_scale = layer_prefix + "self_attn.kv_a_proj_with_mqa.weight_scale"
    if q_a_scale not in state_dict or kv_a_scale not in state_dict:
        return True
    fused_scale = torch.cat([state_dict[q_a_scale], state_dict[kv_a_scale]], dim=0)
    fused_scale = fused_scale.to(torch.float32) * 127.0
    dump_name = layer_prefix + "self_attn.fused_lora_a_proj.weight_scale"
    return check_torch_tensor_equal(dump_name, fused_scale, rank, dump_path)


def check_q_b_nope_rope_proj(state_dict, layer_prefix, model_config, tp_size, rank, dump_path):
    """q_b_proj → TP-shard rows (head-contiguous), no permute (trans_b=true).
    Original: [heads*(nope+rope), q_lora] = [12288, 1536]
    After TP: [heads_tp*(nope+rope), q_lora]"""
    q_b_name = layer_prefix + "self_attn.q_b_proj.weight"
    if q_b_name not in state_dict:
        return True
    weight = state_dict[q_b_name]
    head_num = model_config["num_attention_heads"]
    nope = model_config["qk_nope_head_dim"]
    rope = model_config["qk_rope_head_dim"]
    head_dim = nope + rope

    heads_per_rank = (head_num + tp_size - 1) // tp_size
    row_start = rank * heads_per_rank * head_dim
    row_end = row_start + heads_per_rank * head_dim
    sliced = weight[row_start:row_end, :]  # [heads_tp*(nope+rope), q_lora]
    result = sliced  # trans_b=true: no permute

    dump_name = layer_prefix + "self_attn.q_b_nope_rope_proj.weight"
    return check_torch_tensor_equal(dump_name, result, rank, dump_path, check_dtype=np.int8)


def check_kv_b_nope_proj(state_dict, layer_prefix, model_config, tp_size, rank, dump_path):
    """kv_b_proj → extract nope from each head (2D copy), no permute (trans_b=true).
    Original: [heads*(nope+v), kv_lora] = [16384, 512]
    TP shard heads, extract nope → [heads_tp*nope, kv_lora]"""
    kv_b_name = layer_prefix + "self_attn.kv_b_proj.weight"
    if kv_b_name not in state_dict:
        return True
    weight = state_dict[kv_b_name]
    head_num = model_config["num_attention_heads"]
    nope = model_config["qk_nope_head_dim"]
    v_dim = model_config["v_head_dim"]
    head_dim = nope + v_dim

    heads_per_rank = (head_num + tp_size - 1) // tp_size
    head_start = rank * heads_per_rank

    parts = []
    for h in range(heads_per_rank):
        global_h = head_start + h
        row_start = global_h * head_dim
        parts.append(weight[row_start:row_start + nope, :])
    nope_tensor = torch.cat(parts, dim=0)  # [heads_tp*nope, kv_lora]
    result = nope_tensor  # trans_b=true: no permute

    dump_name = layer_prefix + "self_attn.kv_b_nope_proj.weight"
    return check_torch_tensor_equal(dump_name, result, rank, dump_path, check_dtype=np.int8)


def check_v_head_proj(state_dict, layer_prefix, model_config, tp_size, rank, dump_path):
    """kv_b_proj → extract v_head from each head, no permute (trans_b=true).
    Result: [heads_tp*v_dim, kv_lora]"""
    kv_b_name = layer_prefix + "self_attn.kv_b_proj.weight"
    if kv_b_name not in state_dict:
        return True
    weight = state_dict[kv_b_name]
    head_num = model_config["num_attention_heads"]
    nope = model_config["qk_nope_head_dim"]
    v_dim = model_config["v_head_dim"]
    head_dim = nope + v_dim

    heads_per_rank = (head_num + tp_size - 1) // tp_size
    head_start = rank * heads_per_rank

    parts = []
    for h in range(heads_per_rank):
        global_h = head_start + h
        row_start = global_h * head_dim + nope
        parts.append(weight[row_start:row_start + v_dim, :])
    v_tensor = torch.cat(parts, dim=0)  # [heads_tp*v_dim, kv_lora]
    result = v_tensor  # trans_b=true: no permute

    dump_name = layer_prefix + "self_attn.v_head_proj.weight"
    return check_torch_tensor_equal(dump_name, result, rank, dump_path, check_dtype=np.int8)


def check_kv_b_combined_proj(state_dict, layer_prefix, model_config, tp_size, rank, dump_path):
    """kv_b_proj → TP-shard contiguous rows, no permute (trans_b=true).
    Original: [heads*(nope+v), kv_lora] = [16384, 512]
    Shard: [heads_tp*(nope+v), kv_lora]"""
    kv_b_name = layer_prefix + "self_attn.kv_b_proj.weight"
    if kv_b_name not in state_dict:
        return True
    weight = state_dict[kv_b_name]
    head_num = model_config["num_attention_heads"]
    nope = model_config["qk_nope_head_dim"]
    v_dim = model_config["v_head_dim"]
    head_dim = nope + v_dim

    heads_per_rank = (head_num + tp_size - 1) // tp_size
    row_start = rank * heads_per_rank * head_dim
    row_end = row_start + heads_per_rank * head_dim
    sliced = weight[row_start:row_end, :]  # [heads_tp*(nope+v), kv_lora]
    result = sliced  # trans_b=true: no permute

    dump_name = layer_prefix + "self_attn.kv_b_combined_proj.weight"
    return check_torch_tensor_equal(dump_name, result, rank, dump_path, check_dtype=np.int8)


def check_w_uk_t(state_dict, layer_prefix, model_config, tp_size, rank, dump_path):
    """w_uk_t: kv_b_nope data reshaped to [heads_tp, nope, kv_lora],
    then permute{0,2,1} → [heads_tp, kv_lora, nope].
    Then dequant(per-K-channel) → quant2d(per-N-channel) requantization.
    Final weight: [heads_tp, kv_lora, nope] INT8 (re-quantized).
    Final scale: [heads_tp*kv_lora, 1] FP32 (max_abs format, not divided by 127)."""
    kv_b_name = layer_prefix + "self_attn.kv_b_proj.weight"
    kv_b_scale_name = layer_prefix + "self_attn.kv_b_proj.weight_scale"
    if kv_b_name not in state_dict or kv_b_scale_name not in state_dict:
        return True
    weight = state_dict[kv_b_name]
    orig_scale = state_dict[kv_b_scale_name].to(torch.float32)
    head_num = model_config["num_attention_heads"]
    nope = model_config["qk_nope_head_dim"]
    v_dim = model_config["v_head_dim"]
    kv_lora = model_config["kv_lora_rank"]
    head_dim = nope + v_dim

    heads_per_rank = (head_num + tp_size - 1) // tp_size
    head_start = rank * heads_per_rank

    # Extract nope for each head
    parts = []
    for h in range(heads_per_rank):
        global_h = head_start + h
        row_start = global_h * head_dim
        parts.append(weight[row_start:row_start + nope, :])
    nope_tensor = torch.cat(parts, dim=0)  # [heads_tp*nope, kv_lora]

    # Reshape to [heads_tp, nope, kv_lora], permute to [heads_tp, kv_lora, nope]
    reshaped = nope_tensor.reshape(heads_per_rank, nope, kv_lora)
    permuted = reshaped.permute(0, 2, 1).contiguous()  # [B, N, K] = [heads_tp, kv_lora, nope]

    # Extract per-K-channel scale (original kv_b_nope scale)
    nope_scale_parts = []
    for h in range(heads_per_rank):
        global_h = head_start + h
        row_start = global_h * head_dim
        nope_scale_parts.append(orig_scale[row_start:row_start + nope])
    # old_scale: [heads_tp*nope, 1] → reshape to [heads_tp, nope]
    old_scale_flat = torch.cat(nope_scale_parts, dim=0).squeeze()  # [heads_tp*nope]
    old_scale_reshaped = old_scale_flat.reshape(heads_per_rank, nope)  # [B, K]

    # Dequant: real = int8 * scale (per-K-channel: w_uk_t[b,:,k] shares scale[b,k])
    permuted_float = permuted.to(torch.float32)
    for b in range(heads_per_rank):
        for k in range(nope):
            permuted_float[b, :, k] *= old_scale_reshaped[b, k]

    # Cast to BF16 then back (matching C++ bf16 intermediate buffer)
    permuted_float = permuted_float.to(torch.bfloat16).to(torch.float32)

    # Quant2d per-N-channel: for each row (b*N+n), find max_abs, quantize
    flat = permuted_float.reshape(heads_per_rank * kv_lora, nope)  # [B*N, K]
    new_scale = flat.abs().max(dim=1, keepdim=True).values  # [B*N, 1]
    new_scale = new_scale.clamp(min=1e-10)
    new_int8 = torch.round(flat / new_scale * 127.0).clamp(-128, 127).to(torch.int8)
    new_int8 = new_int8.reshape(heads_per_rank, kv_lora, nope)  # [B, N, K]

    dump_name = layer_prefix + "self_attn.w_uk_t.weight"
    passed = check_torch_tensor_equal(dump_name, new_int8, rank, dump_path, check_dtype=np.int8,
                                      int8_max_mismatch_ratio=0.001)

    # Check scale: [B*N, 1] max_abs format
    dump_scale_name = layer_prefix + "self_attn.w_uk_t.weight_scale"
    passed2 = check_torch_tensor_equal(dump_scale_name, new_scale, rank, dump_path, rtol=1e-2, atol=1e-4)
    return passed and passed2


def check_w_uv(state_dict, layer_prefix, model_config, tp_size, rank, dump_path):
    """w_uv: v_head data (before permute), reshaped to [heads_tp, v_dim, kv_lora].
    Kunlun: no additional permute (keep [B, N, K] for trans_b=true).
    Scale: v_head_proj.weight_scale * 127 (NdScaleTensor max_abs format)."""
    kv_b_name = layer_prefix + "self_attn.kv_b_proj.weight"
    if kv_b_name not in state_dict:
        return True
    weight = state_dict[kv_b_name]
    head_num = model_config["num_attention_heads"]
    nope = model_config["qk_nope_head_dim"]
    v_dim = model_config["v_head_dim"]
    kv_lora = model_config["kv_lora_rank"]
    head_dim = nope + v_dim

    heads_per_rank = (head_num + tp_size - 1) // tp_size
    head_start = rank * heads_per_rank

    parts = []
    for h in range(heads_per_rank):
        global_h = head_start + h
        row_start = global_h * head_dim + nope
        parts.append(weight[row_start:row_start + v_dim, :])
    v_tensor = torch.cat(parts, dim=0)  # [heads_tp*v_dim, kv_lora]

    # Reshape to [heads_tp, v_dim, kv_lora]
    result = v_tensor.reshape(heads_per_rank, v_dim, kv_lora)

    dump_name = layer_prefix + "self_attn.w_uv.weight"
    passed = check_torch_tensor_equal(dump_name, result, rank, dump_path, check_dtype=np.int8)

    # Check scale: v_head_proj.weight_scale * 127
    kv_b_scale_name = layer_prefix + "self_attn.kv_b_proj.weight_scale"
    if kv_b_scale_name in state_dict:
        orig_scale = state_dict[kv_b_scale_name].to(torch.float32)
        v_scale_parts = []
        for h in range(heads_per_rank):
            global_h = head_start + h
            row_start = global_h * head_dim + nope
            v_scale_parts.append(orig_scale[row_start:row_start + v_dim])
        v_scale = torch.cat(v_scale_parts, dim=0)  # [heads_tp*v_dim, 1]
        w_uv_scale = v_scale * 127.0

        dump_scale_name = layer_prefix + "self_attn.w_uv.weight_scale"
        passed2 = check_torch_tensor_equal(dump_scale_name, w_uv_scale, rank, dump_path, rtol=1e-4, atol=1e-6)
        passed = passed and passed2

    return passed


def check_o_proj(state_dict, layer_prefix, model_config, tp_size, rank, dump_path):
    """o_proj: TransSplitOptTrans(transpose=true) without PostProcess permute (trans_b=true).
    Original: [hidden, heads*(nope+v)] = [7168, 8192]
    TransSplitOptTrans: permute → [8192, 7168], split → [1024, 7168], permute → [7168, 1024]
    No PostProcess permute. Final: [7168, 1024]"""
    o_name = layer_prefix + "self_attn.o_proj.weight"
    if o_name not in state_dict:
        return True
    weight = state_dict[o_name]  # [hidden, heads*head_dim] = [7168, 8192]
    # TransSplitOptTrans: permute → split → permute (transpose=true)
    wt = weight.permute(1, 0)  # [8192, 7168]
    slice_size = (wt.shape[0] + tp_size - 1) // tp_size
    sliced = wt[rank * slice_size:(rank + 1) * slice_size, :]  # [1024, 7168]
    after_trans = sliced.permute(1, 0)  # [7168, 1024]
    # trans_b=true: no PostProcess permute
    result = after_trans  # [7168, 1024]

    return check_torch_tensor_equal(o_name, result, rank, dump_path, check_dtype=np.int8)


# ============ Dense MLP / Shared Expert weights ============

def check_gate_up_proj(state_dict, layer_prefix, proj_prefix, tp_size, rank, dump_path):
    """gate+up merged: SplitOptTrans(trans=false) splits each,
    ProcessGateUpProjWeight concatenates, no PostProcess permute (trans_b=true).
    gate_proj: [inter, hidden] → split axis=0 → [inter/tp, hidden]
    up_proj: [inter, hidden] → split axis=0 → [inter/tp, hidden]
    Concat: [inter/tp*2, hidden]
    No PostProcess permute. Final: [inter/tp*2, hidden]"""
    gate_name = layer_prefix + proj_prefix + "gate_proj.weight"
    up_name = layer_prefix + proj_prefix + "up_proj.weight"
    if gate_name not in state_dict or up_name not in state_dict:
        return True

    gate = state_dict[gate_name]  # [inter, hidden]
    up = state_dict[up_name]

    gate_slice_size = (gate.shape[0] + tp_size - 1) // tp_size
    gate_sliced = gate[rank * gate_slice_size:(rank + 1) * gate_slice_size, :]

    up_slice_size = (up.shape[0] + tp_size - 1) // tp_size
    up_sliced = up[rank * up_slice_size:(rank + 1) * up_slice_size, :]

    merged = torch.cat([gate_sliced, up_sliced], dim=0)  # [inter/tp*2, hidden]
    result = merged  # trans_b=true: no PostProcess permute

    # C++ uses "shared_expert" (singular), safetensors uses "shared_experts" (plural)
    dump_prefix = proj_prefix.replace("shared_experts.", "shared_expert.")
    dump_name = layer_prefix + dump_prefix + "gate_up_proj.weight"
    return check_torch_tensor_equal(dump_name, result, rank, dump_path, check_dtype=np.int8)


def check_gate_up_proj_scale(state_dict, layer_prefix, proj_prefix, tp_size, rank, dump_path):
    """gate_up_proj.weight_scale = concat(TP-shard of gate scale, TP-shard of up scale) * 127"""
    gate_scale_name = layer_prefix + proj_prefix + "gate_proj.weight_scale"
    up_scale_name = layer_prefix + proj_prefix + "up_proj.weight_scale"
    if gate_scale_name not in state_dict or up_scale_name not in state_dict:
        return True

    gate_scale = state_dict[gate_scale_name].to(torch.float32)
    up_scale = state_dict[up_scale_name].to(torch.float32)

    gate_slice_size = (gate_scale.shape[0] + tp_size - 1) // tp_size
    gate_sliced = gate_scale[rank * gate_slice_size:(rank + 1) * gate_slice_size]

    up_slice_size = (up_scale.shape[0] + tp_size - 1) // tp_size
    up_sliced = up_scale[rank * up_slice_size:(rank + 1) * up_slice_size]

    result = torch.cat([gate_sliced, up_sliced], dim=0) * 127.0

    dump_prefix = proj_prefix.replace("shared_experts.", "shared_expert.")
    dump_name = layer_prefix + dump_prefix + "gate_up_proj.weight_scale"
    return check_torch_tensor_equal(dump_name, result, rank, dump_path)


def check_down_proj(state_dict, layer_prefix, proj_prefix, tp_size, rank, dump_path):
    """down_proj: TransSplitOptTrans(transpose=true) without PostProcess permute (trans_b=true).
    Original: [hidden, inter] = [7168, 18432]
    Trans: permute → [inter, hidden], split → [inter/tp, hidden], permute → [hidden, inter/tp]
    No PostProcess permute. Final: [hidden, inter/tp]"""
    down_name = layer_prefix + proj_prefix + "down_proj.weight"
    if down_name not in state_dict:
        return True
    weight = state_dict[down_name]  # [hidden, inter]
    wt = weight.permute(1, 0)  # [inter, hidden]
    slice_size = (wt.shape[0] + tp_size - 1) // tp_size
    sliced = wt[rank * slice_size:(rank + 1) * slice_size, :]  # [inter/tp, hidden]
    after_trans = sliced.permute(1, 0)  # [hidden, inter/tp]
    result = after_trans  # trans_b=true: no PostProcess permute

    dump_prefix = proj_prefix.replace("shared_experts.", "shared_expert.")
    dump_name = layer_prefix + dump_prefix + "down_proj.weight"
    return check_torch_tensor_equal(dump_name, result, rank, dump_path, check_dtype=np.int8)


def check_down_proj_scale(state_dict, layer_prefix, proj_prefix, tp_size, rank, dump_path):
    """down_proj.weight_scale: per-output-channel, no TP split (output dim=hidden is not split). *127 applied."""
    scale_name = layer_prefix + proj_prefix + "down_proj.weight_scale"
    if scale_name not in state_dict:
        return True
    result = state_dict[scale_name].to(torch.float32) * 127.0
    dump_prefix = proj_prefix.replace("shared_experts.", "shared_expert.")
    dump_scale_name = layer_prefix + dump_prefix + "down_proj.weight_scale"
    return check_torch_tensor_equal(dump_scale_name, result, rank, dump_path)


# ============ MoE Router ============

def check_mlp_gate(state_dict, layer_prefix, rank, dump_path):
    """mlp.gate.weight: no permute (trans_b=true), kept as original BF16 layout.
    Original: [n_experts, hidden] = [384, 7168] → stays [384, 7168]"""
    gate_name = layer_prefix + "mlp.gate.weight"
    if gate_name not in state_dict:
        return True
    weight = state_dict[gate_name]
    result = weight
    return check_torch_tensor_equal(gate_name, result, rank, dump_path)


def check_mlp_gate_bias(state_dict, layer_prefix, rank, dump_path):
    """mlp.gate.e_score_correction_bias: cast to FP32 on Kunlun, no other transform."""
    bias_name = layer_prefix + "mlp.gate.e_score_correction_bias"
    if bias_name not in state_dict:
        return True
    result = state_dict[bias_name].to(torch.float32)
    return check_torch_tensor_equal(bias_name, result, rank, dump_path)


# ============ W4 Routed Expert weights ============

def check_experts_up_gate_weight(state_dict, model_config, layer_idx, ep_size, rank, dump_path):
    """Expert up_gate_proj: stacked from individual expert gate+up, EP-split.
    Shape on device: [experts_per_rank, moe_inter*2, in_dim_packed]
    gate goes to offset 0, up goes to offset moe_inter."""
    n_experts = model_config["n_routed_experts"]
    moe_inter = model_config["moe_intermediate_size"]

    experts_per_rank = (n_experts + ep_size - 1) // ep_size
    expert_start = rank * experts_per_rank

    tensor_name = "model.layers.{}.mlp.experts.up_gate_proj.weight".format(layer_idx)

    gate_template = "model.layers.{}.mlp.experts.{{}}.gate_proj.weight".format(layer_idx)
    up_template = "model.layers.{}.mlp.experts.{{}}.up_proj.weight".format(layer_idx)

    sample_name = gate_template.format(expert_start)
    if sample_name not in state_dict:
        print("  [SKIP] Expert {} not in state_dict".format(sample_name))
        return True

    sample = state_dict[sample_name]
    in_dim = sample.shape[1]

    expected = np.zeros((experts_per_rank, moe_inter * 2, in_dim), dtype=np.int8)

    for local_idx in range(experts_per_rank):
        global_idx = expert_start + local_idx
        if global_idx >= n_experts:
            break
        gate_w = state_dict[gate_template.format(global_idx)].numpy()
        up_w = state_dict[up_template.format(global_idx)].numpy()
        expected[local_idx, :moe_inter, :] = gate_w
        expected[local_idx, moe_inter:, :] = up_w

    return check_torch_tensor_equal(tensor_name, torch.from_numpy(expected), rank, dump_path,
                                    check_dtype=np.int8)


def check_experts_down_weight(state_dict, model_config, layer_idx, ep_size, rank, dump_path):
    """Expert down_proj: stacked, EP-split. No TP split on columns for moe_tp=1.
    Shape: [experts_per_rank, hidden, in_dim_packed]"""
    n_experts = model_config["n_routed_experts"]
    hidden = model_config["hidden_size"]

    experts_per_rank = (n_experts + ep_size - 1) // ep_size
    expert_start = rank * experts_per_rank

    tensor_name = "model.layers.{}.mlp.experts.down_proj.weight".format(layer_idx)
    down_template = "model.layers.{}.mlp.experts.{{}}.down_proj.weight".format(layer_idx)

    sample_name = down_template.format(expert_start)
    if sample_name not in state_dict:
        print("  [SKIP] Expert {} not in state_dict".format(sample_name))
        return True

    sample = state_dict[sample_name]
    in_dim = sample.shape[1]

    expected = np.zeros((experts_per_rank, hidden, in_dim), dtype=np.int8)

    for local_idx in range(experts_per_rank):
        global_idx = expert_start + local_idx
        if global_idx >= n_experts:
            break
        down_w = state_dict[down_template.format(global_idx)].numpy()
        expected[local_idx, :, :] = down_w

    return check_torch_tensor_equal(tensor_name, torch.from_numpy(expected), rank, dump_path,
                                    check_dtype=np.int8)


def check_experts_up_gate_scale(state_dict, model_config, layer_idx, ep_size, rank, dump_path):
    """Expert up_gate_proj.weight_scale: stacked, EP-split. Scale *7 for W4 INT4 range."""
    n_experts = model_config["n_routed_experts"]
    moe_inter = model_config["moe_intermediate_size"]

    experts_per_rank = (n_experts + ep_size - 1) // ep_size
    expert_start = rank * experts_per_rank

    tensor_name = "model.layers.{}.mlp.experts.up_gate_proj.weight_scale".format(layer_idx)
    gate_scale_template = "model.layers.{}.mlp.experts.{{}}.gate_proj.weight_scale".format(layer_idx)
    up_scale_template = "model.layers.{}.mlp.experts.{{}}.up_proj.weight_scale".format(layer_idx)

    sample_name = gate_scale_template.format(expert_start)
    if sample_name not in state_dict:
        print("  [SKIP] Expert scale {} not in state_dict".format(sample_name))
        return True

    sample = state_dict[sample_name]
    scale_cols = sample.shape[-1] if len(sample.shape) > 1 else 1

    if scale_cols == 1:
        expected = np.zeros((experts_per_rank, moe_inter * 2, 1), dtype=np.float32)
        for local_idx in range(experts_per_rank):
            global_idx = expert_start + local_idx
            if global_idx >= n_experts:
                break
            gs = state_dict[gate_scale_template.format(global_idx)].to(torch.float32).numpy()
            us = state_dict[up_scale_template.format(global_idx)].to(torch.float32).numpy()
            expected[local_idx, :moe_inter, :] = gs
            expected[local_idx, moe_inter:, :] = us
    else:
        expected = np.zeros((experts_per_rank, moe_inter * 2, scale_cols), dtype=np.float32)
        for local_idx in range(experts_per_rank):
            global_idx = expert_start + local_idx
            if global_idx >= n_experts:
                break
            gs = state_dict[gate_scale_template.format(global_idx)].to(torch.float32).numpy()
            us = state_dict[up_scale_template.format(global_idx)].to(torch.float32).numpy()
            expected[local_idx, :moe_inter, :] = gs
            expected[local_idx, moe_inter:, :] = us

    expected *= 7.0

    return check_torch_tensor_equal(tensor_name, torch.from_numpy(expected), rank, dump_path,
                                    rtol=1e-4, atol=1e-6)


def check_experts_down_scale(state_dict, model_config, layer_idx, ep_size, rank, dump_path):
    """Expert down_proj.weight_scale: stacked, EP-split. Scale *7 for W4 INT4 range."""
    n_experts = model_config["n_routed_experts"]
    hidden = model_config["hidden_size"]

    experts_per_rank = (n_experts + ep_size - 1) // ep_size
    expert_start = rank * experts_per_rank

    tensor_name = "model.layers.{}.mlp.experts.down_proj.weight_scale".format(layer_idx)
    down_scale_template = "model.layers.{}.mlp.experts.{{}}.down_proj.weight_scale".format(layer_idx)

    sample_name = down_scale_template.format(expert_start)
    if sample_name not in state_dict:
        print("  [SKIP] Expert scale {} not in state_dict".format(sample_name))
        return True

    sample = state_dict[sample_name]
    scale_cols = sample.shape[-1] if len(sample.shape) > 1 else 1

    if scale_cols == 1:
        expected = np.zeros((experts_per_rank, hidden, 1), dtype=np.float32)
        for local_idx in range(experts_per_rank):
            global_idx = expert_start + local_idx
            if global_idx >= n_experts:
                break
            ds = state_dict[down_scale_template.format(global_idx)].to(torch.float32).numpy()
            expected[local_idx, :, :] = ds
    else:
        expected = np.zeros((experts_per_rank, hidden, scale_cols), dtype=np.float32)
        for local_idx in range(experts_per_rank):
            global_idx = expert_start + local_idx
            if global_idx >= n_experts:
                break
            ds = state_dict[down_scale_template.format(global_idx)].to(torch.float32).numpy()
            expected[local_idx, :, :] = ds

    expected *= 7.0

    return check_torch_tensor_equal(tensor_name, torch.from_numpy(expected), rank, dump_path,
                                    rtol=1e-4, atol=1e-6)


# ============ Scale checks for attention weights ============

def check_q_b_proj_scale(state_dict, layer_prefix, model_config, tp_size, rank, dump_path):
    """q_b_proj.weight_scale: stored as original full scale (not TP-sharded)."""
    scale_name = layer_prefix + "self_attn.q_b_proj.weight_scale"
    if scale_name not in state_dict:
        return True
    result = state_dict[scale_name].to(torch.float32)
    return check_torch_tensor_equal(scale_name, result, rank, dump_path)


def check_kv_b_proj_scale(state_dict, layer_prefix, model_config, tp_size, rank, dump_path):
    """kv_b_proj.weight_scale: stored as original full scale (not TP-sharded)."""
    scale_name = layer_prefix + "self_attn.kv_b_proj.weight_scale"
    if scale_name not in state_dict:
        return True
    result = state_dict[scale_name].to(torch.float32)
    return check_torch_tensor_equal(scale_name, result, rank, dump_path)


def check_kv_b_nope_proj_scale(state_dict, layer_prefix, model_config, tp_size, rank, dump_path):
    """kv_b_nope_proj.weight_scale: extracted per-head nope scales from kv_b_proj scale, *127 applied."""
    orig_scale_name = layer_prefix + "self_attn.kv_b_proj.weight_scale"
    if orig_scale_name not in state_dict:
        return True
    scale = state_dict[orig_scale_name].to(torch.float32)
    head_num = model_config["num_attention_heads"]
    nope = model_config["qk_nope_head_dim"]
    v_dim = model_config["v_head_dim"]
    head_dim = nope + v_dim

    heads_per_rank = (head_num + tp_size - 1) // tp_size
    head_start = rank * heads_per_rank

    parts = []
    for h in range(heads_per_rank):
        global_h = head_start + h
        row_start = global_h * head_dim
        parts.append(scale[row_start:row_start + nope])
    result = torch.cat(parts, dim=0) * 127.0

    dump_name = layer_prefix + "self_attn.kv_b_nope_proj.weight_scale"
    return check_torch_tensor_equal(dump_name, result, rank, dump_path)


def check_kv_b_combined_proj_scale(state_dict, layer_prefix, model_config, tp_size, rank, dump_path):
    """kv_b_combined_proj.weight_scale: contiguous TP-shard of kv_b_proj scale, *127 applied."""
    orig_scale_name = layer_prefix + "self_attn.kv_b_proj.weight_scale"
    if orig_scale_name not in state_dict:
        return True
    scale = state_dict[orig_scale_name].to(torch.float32)
    head_num = model_config["num_attention_heads"]
    nope = model_config["qk_nope_head_dim"]
    v_dim = model_config["v_head_dim"]
    head_dim = nope + v_dim
    total = head_num * head_dim
    slice_size = (total + tp_size - 1) // tp_size
    sliced = scale[rank * slice_size:(rank + 1) * slice_size] * 127.0

    dump_name = layer_prefix + "self_attn.kv_b_combined_proj.weight_scale"
    return check_torch_tensor_equal(dump_name, sliced, rank, dump_path)


def check_o_proj_scale(state_dict, layer_prefix, rank, dump_path):
    """o_proj.weight_scale: per-output-channel (hidden dim), no TP split. *127 applied."""
    scale_name = layer_prefix + "self_attn.o_proj.weight_scale"
    if scale_name not in state_dict:
        return True
    result = state_dict[scale_name].to(torch.float32) * 127.0
    return check_torch_tensor_equal(scale_name, result, rank, dump_path)


def check_q_a_proj_scale(state_dict, layer_prefix, rank, dump_path):
    """q_a_proj.weight_scale: no TP split."""
    scale_name = layer_prefix + "self_attn.q_a_proj.weight_scale"
    if scale_name not in state_dict:
        return True
    result = state_dict[scale_name].to(torch.float32)
    return check_torch_tensor_equal(scale_name, result, rank, dump_path)


def check_kv_a_proj_scale(state_dict, layer_prefix, rank, dump_path):
    """kv_a_proj_with_mqa.weight_scale: no TP split."""
    scale_name = layer_prefix + "self_attn.kv_a_proj_with_mqa.weight_scale"
    if scale_name not in state_dict:
        return True
    result = state_dict[scale_name].to(torch.float32)
    return check_torch_tensor_equal(scale_name, result, rank, dump_path)


# ============ Main ============

def main():
    if len(sys.argv) < 4:
        print("Usage: python check_kunlun_kimi_w4a8_weight_tensor.py <model_dir> <tp_size> <ep_size> [dump_path]")
        sys.exit(1)

    model_dir = sys.argv[1]
    tp_size = int(sys.argv[2])
    ep_size = int(sys.argv[3])
    if not os.path.isdir(model_dir):
        print("Model dir {} does not exist.".format(model_dir))
        sys.exit(1)

    if len(sys.argv) >= 5:
        dump_path = sys.argv[4]
    else:
        dump_path = os.path.join(os.getcwd(), os.path.basename(model_dir.rstrip("/")))
    if not os.path.isdir(dump_path):
        print("Dump path {} does not exist.".format(dump_path))
        sys.exit(1)

    config_file = os.path.join(model_dir, "config.json")
    with open(config_file) as f:
        model_config = json.load(f)
    if "text_config" in model_config:
        model_config = model_config["text_config"]

    first_k_dense_replace = model_config.get("first_k_dense_replace", 1)

    print("Loading model weights from {}...".format(model_dir))
    state_dict = {}
    model_files = [os.path.join(os.path.abspath(model_dir), f) for f in os.listdir(model_dir)
                   if not f.startswith('.')]
    for model_file in model_files:
        if not model_file.endswith(".safetensors"):
            continue
        print("  Loading {}".format(model_file))
        with safe_open(model_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    print("Loaded {} tensors from safetensors.".format(len(state_dict)))

    # Normalize key names: remove "language_model." prefix if present
    normalized = {}
    for key, value in state_dict.items():
        nk = key[len("language_model."):] if key.startswith("language_model.") else key
        normalized[nk] = value
    state_dict = normalized
    print("Normalized key names ({} tensors).".format(len(state_dict)))

    # Determine which layers were dumped
    dumped_layers = set()
    for rank in range(tp_size):
        rank_dir = os.path.join(dump_path, str(rank))
        if not os.path.isdir(rank_dir):
            continue
        for fname in os.listdir(rank_dir):
            match = re.search(r"layers\.(\d+)", fname)
            if match:
                dumped_layers.add(int(match.group(1)))
    print("Detected dumped layers: {}".format(sorted(dumped_layers)))
    print("Config: TP={}, EP={}, heads={}, q_lora={}, kv_lora={}, nope={}, rope={}, v={}".format(
        tp_size, ep_size, model_config["num_attention_heads"],
        model_config["q_lora_rank"], model_config["kv_lora_rank"],
        model_config["qk_nope_head_dim"], model_config["qk_rope_head_dim"], model_config["v_head_dim"]))

    all_passed = True

    # Auto-detect available rank directories under dump_path (within tp/ep size)
    max_rank = max(tp_size, ep_size)
    available_ranks = []
    for i in range(max_rank + 1):
        rank_dir = os.path.join(dump_path, str(i))
        if os.path.isdir(rank_dir):
            available_ranks.append(i)
    if not available_ranks:
        print("No rank directories (0-{}) found under {}".format(max_rank, dump_path))
        sys.exit(1)
    print("Detected rank directories: {}".format(available_ranks))

    for rank in available_ranks:
        print("\n===== Checking rank {} =====".format(rank))

        # 1. Global weights
        print("\n--- Global weights ---")
        if "model.embed_tokens.weight" in state_dict:
            if not check_embed_tokens(state_dict, "model.embed_tokens.weight", tp_size, rank, dump_path):
                all_passed = False
        if "lm_head.weight" in state_dict:
            if not check_lm_head(state_dict, "lm_head.weight", tp_size, rank, dump_path):
                all_passed = False
        if "model.norm.weight" in state_dict:
            if not check_norm(state_dict, "model.norm.weight", rank, dump_path):
                all_passed = False

        # 2. Per-layer weights
        for layer_idx in sorted(dumped_layers):
            print("\n--- Layer {} ---".format(layer_idx))
            prefix = "model.layers.{}.".format(layer_idx)

            # Norms (no split)
            for norm_name in ["input_layernorm.weight", "post_attention_layernorm.weight",
                              "self_attn.kv_a_layernorm.weight", "self_attn.q_a_layernorm.weight"]:
                full_name = prefix + norm_name
                if full_name in state_dict:
                    if not check_norm(state_dict, full_name, rank, dump_path):
                        all_passed = False

            # MoE router gate
            if layer_idx >= first_k_dense_replace:
                if not check_mlp_gate(state_dict, prefix, rank, dump_path):
                    all_passed = False
                if not check_mlp_gate_bias(state_dict, prefix, rank, dump_path):
                    all_passed = False

            # Attention
            print("  -- Attention --")
            if not check_fused_lora_a_proj(state_dict, prefix, model_config, rank, dump_path):
                all_passed = False
            if not check_fused_lora_a_proj_scale(state_dict, prefix, rank, dump_path):
                all_passed = False
            if not check_q_a_proj_scale(state_dict, prefix, rank, dump_path):
                all_passed = False
            if not check_kv_a_proj_scale(state_dict, prefix, rank, dump_path):
                all_passed = False
            if not check_q_b_nope_rope_proj(state_dict, prefix, model_config, tp_size, rank, dump_path):
                all_passed = False
            if not check_q_b_proj_scale(state_dict, prefix, model_config, tp_size, rank, dump_path):
                all_passed = False
            if not check_kv_b_nope_proj(state_dict, prefix, model_config, tp_size, rank, dump_path):
                all_passed = False
            if not check_v_head_proj(state_dict, prefix, model_config, tp_size, rank, dump_path):
                all_passed = False
            if not check_kv_b_combined_proj(state_dict, prefix, model_config, tp_size, rank, dump_path):
                all_passed = False
            if not check_kv_b_nope_proj_scale(state_dict, prefix, model_config, tp_size, rank, dump_path):
                all_passed = False
            if not check_kv_b_combined_proj_scale(state_dict, prefix, model_config, tp_size, rank, dump_path):
                all_passed = False
            if not check_kv_b_proj_scale(state_dict, prefix, model_config, tp_size, rank, dump_path):
                all_passed = False
            if not check_w_uk_t(state_dict, prefix, model_config, tp_size, rank, dump_path):
                all_passed = False
            if not check_w_uv(state_dict, prefix, model_config, tp_size, rank, dump_path):
                all_passed = False
            if not check_o_proj(state_dict, prefix, model_config, tp_size, rank, dump_path):
                all_passed = False
            if not check_o_proj_scale(state_dict, prefix, rank, dump_path):
                all_passed = False

            # Dense MLP (layer 0) or Shared Expert (layer >= first_k_dense_replace)
            print("  -- MLP / Shared Expert --")
            if layer_idx < first_k_dense_replace:
                if not check_gate_up_proj(state_dict, prefix, "mlp.", tp_size, rank, dump_path):
                    all_passed = False
                if not check_gate_up_proj_scale(state_dict, prefix, "mlp.", tp_size, rank, dump_path):
                    all_passed = False
                if not check_down_proj(state_dict, prefix, "mlp.", tp_size, rank, dump_path):
                    all_passed = False
                if not check_down_proj_scale(state_dict, prefix, "mlp.", tp_size, rank, dump_path):
                    all_passed = False
            else:
                # shared_experts in safetensors → shared_expert in C++ dump
                if not check_gate_up_proj(state_dict, prefix, "mlp.shared_experts.", tp_size, rank, dump_path):
                    all_passed = False
                if not check_gate_up_proj_scale(state_dict, prefix, "mlp.shared_experts.", tp_size, rank, dump_path):
                    all_passed = False
                if not check_down_proj(state_dict, prefix, "mlp.shared_experts.", tp_size, rank, dump_path):
                    all_passed = False
                if not check_down_proj_scale(state_dict, prefix, "mlp.shared_experts.", tp_size, rank, dump_path):
                    all_passed = False

            # Routed Experts (W4)
            if layer_idx >= first_k_dense_replace:
                print("  -- Routed Experts (W4) --")
                if not check_experts_up_gate_weight(state_dict, model_config, layer_idx, ep_size, rank, dump_path):
                    all_passed = False
                if not check_experts_down_weight(state_dict, model_config, layer_idx, ep_size, rank, dump_path):
                    all_passed = False
                if not check_experts_up_gate_scale(state_dict, model_config, layer_idx, ep_size, rank, dump_path):
                    all_passed = False
                if not check_experts_down_scale(state_dict, model_config, layer_idx, ep_size, rank, dump_path):
                    all_passed = False

    if all_passed:
        print("\n========================================")
        print("All checks PASSED!")
        print("========================================")
        success_file = os.path.join(dump_path, "SUCCESS")
        open(success_file, 'a').close()
        sys.exit(0)
    else:
        print("\n========================================")
        print("Some checks FAILED! See details above.")
        print("========================================")
        sys.exit(1)


if __name__ == "__main__":
    main()
