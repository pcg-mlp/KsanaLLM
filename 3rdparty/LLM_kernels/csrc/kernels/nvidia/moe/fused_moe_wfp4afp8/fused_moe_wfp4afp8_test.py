# Copyright 2025 Tencent Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ==============================================================================
#
# fused_moe_wfp4afp8 unit test
#
# 与 fused_moe_wfp4afp8.py 同目录，独立可执行：
#     python fused_moe_wfp4afp8_test.py
#
# 验证 kernel 输出与"高精度反量化 + 模拟 FP8 round-trip"参考实现是否数值匹配。
# 测试矩阵覆盖：quant_a_per_tensor、MUL_ROUTED_WEIGHT、block_k_diviable、
# compute_type ∈ {FP16, BF16}。
#
# 失败排查指引：
#   - 误差 ≈ 2^6 (=64) 倍：E8M0 的 +6 补偿没加（kernel 中 `b_scale_u8.to(uint32) + 6`）。
#   - 仅在 nibble 奇数位错：b_shifter 与参考侧 nibble 顺序不一致。
#   - 仅在 FP4 子规格 (nibble 0001/1001) 输入上错：误用了"normal-only"位重构而未走
#     marlin trick `((nibble & 0x8) << 4) | ((nibble & 0x7) << 2)`。
#   - 输出大面积出现 NaN：B_scale 取得太大导致 fp4 * 2^k 超出 FP8 E4M3 表示范围
#     (~448)。Triton 的 .to(float8e4nv) 对超界值不 saturate 而是产 NaN，单个 NaN
#     在 K 维累加后会污染整行。需要在 make_inputs 中限制 B_scale 上限。
# ==============================================================================

import os
import sys

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
common_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, current_dir)
sys.path.insert(0, common_dir)

import triton.language as tl
from common import set_global_seed, moe_align_block_cpu
from fused_moe_wfp4afp8 import fused_moe_wfp4afp8


# FP4 E2M1 的 16 个值（nibble 索引 → fp32）
# 顺序与 (sign << 3) | (exp << 1) | mantissa 完全一致：
#   0x0=+0, 0x1=+0.5, 0x2=+1, 0x3=+1.5, 0x4=+2, 0x5=+3, 0x6=+4, 0x7=+6
#   0x8=-0, 0x9=-0.5, 0xA=-1, 0xB=-1.5, 0xC=-2, 0xD=-3, 0xE=-4, 0xF=-6
FP4_LUT = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def unpack_mxfp4_to_fp32(b_packed: torch.Tensor, k_dim: int) -> torch.Tensor:
    """
    解码 MXFP4 packed uint8 -> fp32。
    b_packed: [..., K/2] uint8（low nibble = 偶数 k，high nibble = 奇数 k；与 kernel
              的 b_shifter = (offs_k % 2) * 4 一致）。
    返回:     [..., K] fp32
    """
    assert k_dim % 2 == 0, "K 必须是偶数（mxfp4 一字节 2 个 fp4）"
    lut = FP4_LUT.to(b_packed.device)
    low = (b_packed & 0xF).to(torch.long)        # 偶数 k
    high = ((b_packed >> 4) & 0xF).to(torch.long)  # 奇数 k
    out = torch.empty(b_packed.shape[:-1] + (k_dim,),
                      dtype=torch.float32,
                      device=b_packed.device)
    out[..., 0::2] = lut[low]
    out[..., 1::2] = lut[high]
    return out


def decode_e8m0_to_fp32(scale_u8: torch.Tensor) -> torch.Tensor:
    """
    E8M0 (uint8) -> fp32 乘子：value = 2^(e - 127)。
    e=0 退化为 0；e=255 在 OCP MX 中是 NaN；测试输入会避开这两个值。
    """
    e = scale_u8.to(torch.int32)
    return torch.pow(torch.tensor(2.0, dtype=torch.float32, device=scale_u8.device),
                     (e - 127).to(torch.float32))


def reference_moe_wfp4afp8(
    A_fp8: torch.Tensor,           # [m, K] fp8_e4m3
    B_packed: torch.Tensor,        # [E, N, K/2] uint8
    B_scale_u8: torch.Tensor,      # [E, N, K/group_size] uint8 (E8M0)
    A_scale: torch.Tensor,         # 标量 [1] 或 [m, K/128] fp32
    topk_weights: torch.Tensor,    # [m, top_k] fp32
    topk_ids: torch.Tensor,        # [m, top_k] int32
    mul_routed_weight: bool,
    quant_a_per_tensor: bool,
    group_size: int,               # MXFP4 = 32
    out_dtype: torch.dtype,        # fp16 / bf16
    a_scale_block: int = 128,
) -> torch.Tensor:
    """
    高精度参考实现：
      1) FP4 LUT 精确解码 + E8M0 精确解码，得到 fp32 真值 B_dq。
      2) 模拟 kernel 内 `(fp4*2^-6 -> fp8) * (e8m0*2^6) -> fp32 -> fp8` 这条路径：
         直接对 B_dq 做 fp32 -> fp8 -> fp32 round-trip。
         数学上与 kernel 路径等价（2^6 的尺度变化只影响中间表示，cast 回 fp8 后
         有效精度由 fp8 mantissa 决定，与 2^k 缩放无关，因 fp8 是浮点表示）。
      3) A 已是 fp8，直接 fp8 -> fp32 即可，无额外量化。
      4) fp32 matmul 累加，再后乘 a_scale 与（可选）topk_weights，cast 到 out_dtype。
    """
    device = A_fp8.device
    m, K = A_fp8.shape
    E, N, _ = B_packed.shape
    top_k = topk_ids.shape[1]

    # ---- B 反量化 + 模拟 fp8 round-trip ----
    B_fp32 = unpack_mxfp4_to_fp32(B_packed, K)        # [E, N, K] fp32（精确）
    B_scale_fp32 = decode_e8m0_to_fp32(B_scale_u8)    # [E, N, K/group_size] fp32
    # 把 scale 沿 K 维 repeat-interleave 到 [E, N, K]
    B_scale_full = B_scale_fp32.repeat_interleave(group_size, dim=-1)
    B_dq_fp32 = B_fp32 * B_scale_full                 # [E, N, K] fp32 真值
    # 模拟 kernel 的 fp8 量化：fp32 -> fp8_e4m3 -> fp32
    B_for_dot = B_dq_fp32.to(torch.float8_e4m3fn).to(torch.float32)

    # ---- A 反量化 ----
    A_fp32 = A_fp8.to(torch.float32)                  # [m, K]

    # ---- per-token / per-expert matmul（fp32 累加，按 a_scale_block 分块） ----
    #   per-tensor:  把整个 K 当作 1 个 block，scale 是标量
    #   per-block:   每 a_scale_block 个 k 一个 scale，分块 matmul 后乘对应 scale
    # kernel 内 a_scale 在 dot 之后乘到 partial sum 上，BLOCK_SIZE_K 与
    # a_scale_block 必须一致（约定 BLOCK_SIZE_K=128，a_scale_block=128）。
    out = torch.zeros((m, top_k, N), dtype=torch.float32, device=device)
    if quant_a_per_tensor:
        a_scale_scalar = A_scale.view(()).to(torch.float32)
        for t in range(top_k):
            expert_per_token = topk_ids[:, t].long()
            B_sel = B_for_dot[expert_per_token]                  # [m, N, K]
            out[:, t, :] = torch.einsum("mk,mnk->mn",
                                        A_fp32, B_sel) * a_scale_scalar
    else:
        BK = a_scale_block
        assert K % BK == 0, "K 必须是 a_scale block 大小的整数倍"
        num_a_blocks = K // BK
        for t in range(top_k):
            expert_per_token = topk_ids[:, t].long()
            B_sel = B_for_dot[expert_per_token]                  # [m, N, K]
            for kb in range(num_a_blocks):
                k0, k1 = kb * BK, (kb + 1) * BK
                scale_block = A_scale[:, kb].view(m, 1).to(torch.float32)
                partial = torch.einsum("mk,mnk->mn",
                                       A_fp32[:, k0:k1], B_sel[:, :, k0:k1])
                out[:, t, :] += partial * scale_block

    # ---- topk_weights ----
    if mul_routed_weight:
        out = out * topk_weights.unsqueeze(-1).to(torch.float32)

    return out.to(out_dtype)


def make_inputs(m, k, n, num_experts, top_k, quant_a_per_tensor, dtype, device='cuda'):
    """
    构造测试输入：
      A: fp8_e4m3 [m, k]
      B: uint8 [E, N, K/2] (随机全 nibble 范围，覆盖正负、零、子规格)
      B_scale: uint8 [E, N, K/32]，限制在 [120, 132)，对应 2^[-7, +4]
      A_scale: 标量或 [m, K/128]

    B_scale 上限收紧到 2^4=16 的原因：FP4 最大 |val|=6，则 |B_dq|=6*2^k 在 cast 到
    FP8 E4M3 (max ~448) 之前必须保持在 [-448, 448] 内，否则 Triton 的 .to(float8e4nv)
    会产生 NaN（不像 PyTorch 的 float8_e4m3fn 那样 saturate），导致 K 维累加被污染。
    所以 e <= 131 → max(|B_dq|) = 6*16 = 96 << 448，并且 FP16 输出也不会溢出。
    """
    A_fp32 = torch.randn([m, k], device=device, dtype=torch.float32).clamp(-3.0, 3.0)
    A = A_fp32.to(torch.float8_e4m3fn)

    B = torch.randint(0, 256, [num_experts, n, k // 2],
                      device=device, dtype=torch.uint8)
    # E8M0 限定区间：[120, 132) -> 2^[-7, +4]，避开 0/255，且不让 B_dq 溢出 FP8
    B_scale = torch.randint(120, 132, [num_experts, n, k // 32],
                            device=device, dtype=torch.uint8)

    if quant_a_per_tensor:
        A_scale = torch.tensor([0.1], device=device, dtype=torch.float32)
    else:
        A_scale = (torch.rand([m, k // 128], device=device, dtype=torch.float32)
                   * 0.2 + 0.05)

    topk_weights = torch.rand([m, top_k], device=device, dtype=torch.float32)
    topk_ids = torch.randint(0, num_experts, [m, top_k],
                             device=device, dtype=torch.int32)

    return A, B, B_scale, A_scale, topk_weights, topk_ids


def run_moe_align(topk_ids, num_experts, block_size_m, m, top_k):
    """调用 common.moe_align_block_cpu 生成 sorted_token_ids / expert_ids /
       num_tokens_post_padded。"""
    numel = m * top_k
    em = numel + num_experts * (block_size_m - 1)
    max_num_m_blocks = (em + block_size_m - 1) // block_size_m

    sorted_token_ids = torch.empty([em], dtype=torch.int32, device='cuda')
    expert_ids = torch.empty([max_num_m_blocks], dtype=torch.int32, device='cuda')
    num_tokens_post_padded = torch.zeros([1], dtype=torch.int32, device='cuda')

    # moe_align_block_cpu 是 CPU 实现，需要把 tensor 搬到 CPU
    topk_ids_cpu = topk_ids.flatten().cpu()
    sorted_cpu = sorted_token_ids.cpu()
    expert_cpu = expert_ids.cpu()
    num_post_cpu = num_tokens_post_padded.cpu()
    moe_align_block_cpu(topk_ids_cpu, expert_cpu, sorted_cpu, num_post_cpu,
                        m, top_k, num_experts, block_size_m)
    sorted_token_ids.copy_(sorted_cpu)
    expert_ids.copy_(expert_cpu)
    num_tokens_post_padded.copy_(num_post_cpu)
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def torch_dtype_of(compute_type_str):
    return {"FP16": torch.float16, "BF16": torch.bfloat16}[compute_type_str]


def triton_dtype_of(compute_type_str):
    return {"FP16": tl.float16, "BF16": tl.bfloat16}[compute_type_str]


def run_one_case(name, m, k, n, num_experts, top_k, mul_routed_weight,
                 quant_a_per_tensor, compute_type, config, group_size=32,
                 rtol=5e-2, atol=5e-2):
    print(f"\n[CASE] {name}: m={m} k={k} n={n} E={num_experts} topk={top_k} "
          f"mul_routed={mul_routed_weight} per_tensor_a={quant_a_per_tensor} "
          f"ctype={compute_type} BLOCK_K={config['BLOCK_SIZE_K']}")
    out_dtype = torch_dtype_of(compute_type)
    triton_compute_type = triton_dtype_of(compute_type)

    A, B, B_scale, A_scale, topk_weights, topk_ids = make_inputs(
        m, k, n, num_experts, top_k, quant_a_per_tensor, out_dtype)
    sorted_ids, expert_ids, num_post = run_moe_align(
        topk_ids, num_experts, config["BLOCK_SIZE_M"], m, top_k)

    # ---- 参考实现 ----
    ref = reference_moe_wfp4afp8(
        A, B, B_scale, A_scale, topk_weights, topk_ids,
        mul_routed_weight, quant_a_per_tensor, group_size, out_dtype)

    # ---- kernel ----
    C = torch.empty([m, top_k, n], dtype=out_dtype, device='cuda')
    fused_moe_wfp4afp8(A, B, C, A_scale, B_scale, topk_weights, topk_ids,
                       sorted_ids, expert_ids, num_post,
                       mul_routed_weight, top_k, config,
                       triton_compute_type, quant_a_per_tensor,
                       [0, group_size])
    torch.cuda.synchronize()

    # ---- NaN/Inf sanity 检查（早失败 + 明确诊断） ----
    n_nan_kernel = torch.isnan(C.float()).sum().item()
    n_nan_ref = torch.isnan(ref.float()).sum().item()
    n_inf_kernel = torch.isinf(C.float()).sum().item()
    n_inf_ref = torch.isinf(ref.float()).sum().item()
    if n_nan_kernel or n_nan_ref or n_inf_kernel or n_inf_ref:
        raise AssertionError(
            f"NaN/Inf detected: kernel NaN={n_nan_kernel} Inf={n_inf_kernel}, "
            f"ref NaN={n_nan_ref} Inf={n_inf_ref}. "
            f"很可能是 B_scale 上限取得过大导致 FP8 cast 溢出，参见文件顶部失败排查指引。")

    # ---- 比对 ----
    diff = (C.float() - ref.float()).abs()
    ref_abs = ref.float().abs().clamp_min(1e-4)
    rel = diff / ref_abs
    print(f"        max_abs={diff.max().item():.4e} "
          f"mean_abs={diff.mean().item():.4e} "
          f"max_rel={rel.max().item():.4e} "
          f"mean_rel={rel.mean().item():.4e}")
    torch.testing.assert_close(C, ref, rtol=rtol, atol=atol)
    print(f"        [PASS]")


TEST_CASES = [
    # 配置 A：top_k=8 + per-block A_scale + BF16
    dict(name="A_topk8_perblock_BF16",
         m=64, k=512, n=256, num_experts=8, top_k=8,
         mul_routed_weight=False, quant_a_per_tensor=False,
         compute_type="BF16",
         config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128,
                 "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1}),

    # 配置 B：top_k=8 + per-tensor A_scale + FP16
    dict(name="B_topk8_pertensor_FP16",
         m=64, k=512, n=256, num_experts=8, top_k=8,
         mul_routed_weight=False, quant_a_per_tensor=True,
         compute_type="FP16",
         config={"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128,
                 "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1}),

    # 配置 C：top_k=1 + MUL_ROUTED_WEIGHT=True + per-tensor + BF16
    dict(name="C_topk1_mul_routed_BF16",
         m=64, k=512, n=256, num_experts=8, top_k=1,
         mul_routed_weight=True, quant_a_per_tensor=True,
         compute_type="BF16",
         config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128,
                 "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1}),

    # 配置 D：K 不能整除 BLOCK_SIZE_K（block_k_diviable=False 路径，验证 mask 分支）
    # 约束：
    #   - Triton 要求 BLOCK_SIZE_K 是 2 的幂（tl.arange 限制）
    #   - K 必须是 group_size=32 的整数倍（B_scale 索引），per-tensor A_scale 下
    #     对 K 是否为 128 的倍数无要求
    # 选 BLOCK_SIZE_K=256, K=384：384 % 256 = 128 ≠ 0 → block_k_diviable=False，
    # 且 384/32=12 满足 group_size 整除。第一次迭代 k=[0, 256) 全有效，第二次
    # k=[256, 512) 只有 [256, 384) 有效，正好走 mask 分支。
    dict(name="D_block_k_indivisible_BF16",
         m=32, k=384, n=256, num_experts=4, top_k=4,
         mul_routed_weight=False, quant_a_per_tensor=True,
         compute_type="BF16",
         config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128,
                 "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1}),
]


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return 0

    set_global_seed(0)

    for cfg in TEST_CASES:
        run_one_case(**cfg)

    print("\nAll cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
