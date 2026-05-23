# Copyright 2025 Tencent Inc. All rights reserved.
# Copyright 2025 vLLM Team
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
# ==============================================================================
#
# Adapted from
# [vLLM Project] https://github.com/vllm-project/vllm/blob/v0.8.2/vllm/model_executor/layers/fused_moe/fused_moe.py#L36
#
# ==============================================================================

import argparse
import os
import sys
import json
import torch
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

os.environ["PATH"] = "/usr/local/nvidia/lib64:" + os.environ["PATH"]

from typing import Any, Dict, List, Optional, Tuple, Union

import triton
import triton.language as tl

current_dir = os.path.dirname(os.path.abspath(__file__))
common_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, current_dir)
sys.path.insert(0, common_dir)
from common import dump_kernel, str_to_bool, set_global_seed, moe_align_block_cpu, performance_kernel

"""
    Fused MoE WFP4AFP8 Kernel - W=MXFP4 / A=FP8 混合精度计算内核

    适用平台：NVIDIA Hopper H20（SM90 / SM90A），无原生 FP4 tensor core，
    因此在 kernel 内部把 MXFP4 权重 on-the-fly 转换为 FP8 E4M3，再用 FP8
    tensor core 做 GEMM、FP32 累加。

    设计要点：
    1. B 为 MXFP4：每 byte 打包 2 个 FP4 E2M1（low nibble = 偶数 k，high
       nibble = 奇数 k；与 fused_moe_kernel_gptq_awq 的打包一致）。
       shape = [num_experts, N, K/2]，dtype = uint8。
    2. B_scale 为 E8M0（OCP MX 标准），group_size 固定 32。
       shape = [num_experts, N, K/32]，dtype = uint8。
    3. A 为 FP8 E4M3，A_scale 与 fused_moe_gptq_int4_fp8_kernel 一致：
       quant_a_per_tensor=True  -> 标量；
       quant_a_per_tensor=False -> [m, K/128] fp32。
    4. FP4→FP8 转换采用 marlin 风格的位重排：
           fp8_byte = ((nibble & 0x8) << 4) | ((nibble & 0x7) << 2)
       该重排对全部 16 个 FP4 值（含子规格）满足
           fp8_value = fp4_value * 2^-6
       因此通过把 E8M0 scale 的指数字段 +6 直接补偿掉这个 1/64 因子，
       完全无需分支。
    5. b_scale 必须在 dot 之前融入 b（因为同一 K-block 内不同 32-子块的
       scale 不同），与 fused_moe_gptq_int4_fp8 的事后乘语义不同。
"""


@triton.jit
def fused_moe_wfp4afp8_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        N,
        K,
        EM,
        num_valid_tokens,
        quant_a_per_tensor: tl.constexpr,
        block_k_diviable: tl.constexpr,
        group_size: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + N * offs_token[:, None] + offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N

    # MXFP4 一字节打包 2 个 nibble：low (偶数 k) / high (奇数 k)，与 gptq_awq 一致
    pack_factor = 8 // 4  # = 2

    if not quant_a_per_tensor:
        a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * K // 128
    else:
        a_scale = tl.load(a_scale_ptr).to(tl.float32)

    stride_bse = N * K // group_size
    stride_bsn = K // group_size

    NUM_SUB_K = BLOCK_SIZE_K // group_size  # constexpr 算术 -> Python int
    SUB_K = group_size                       # constexpr 别名

    # 不含 K offset 的 base 量，每 (k, sk) 内只需做加法
    a_row_offs = (offs_token[:, None] // top_k) * K          # [BSM, 1]
    b_n_byte_offs = offs_bn[None, :] * (K // pack_factor)    # [1, BSN]
    b_expert_byte_offs = off_experts * (N * K // pack_factor)
    b_scale_expert_n_offs = off_experts * stride_bse + offs_bn * stride_bsn  # [BSN]
    sub_k_arange = tl.arange(0, group_size)                  # [SUB_K]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # A scale per 128-K-block：每 K-block 加载一次（K-block 内所有 sub 共用）
        if not quant_a_per_tensor:
            k_start = k * BLOCK_SIZE_K
            offs_ks = k_start // 128
            a_scale = tl.load(a_scale_ptrs + offs_ks,
                              mask=token_mask,
                              other=0.0)

        # K-block 内的 sub-acc：让 4 个 sub-dot 的 wgmma fold 成一个 chain，
        # 模拟原 kernel 单大 dot 的 wgmma 累加路径（fp16 acc + end-of-K-block flush）。
        sub_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for sk in range(NUM_SUB_K):
            # 本 sub-block 的 K 位置 [SUB_K]
            sub_k_pos = k * BLOCK_SIZE_K + sk * group_size + sub_k_arange
            sub_k_in_K = sub_k_pos < K  # [SUB_K]

            # A [BSM, SUB_K]：padding 行和越界 K 元素 mask 为 0
            a_sub = tl.load(
                a_ptr + a_row_offs + sub_k_pos[None, :],
                mask=token_mask[:, None] & sub_k_in_K[None, :],
                other=0.0)

            # B [SUB_K, BSN]：FP4 packed -> 解包 -> FP8（marlin 位重排，
            # fp8_value = fp4_value * 2^-6，1/64 因子由后面 b_scale 指数 +6 抵消）
            b_byte_pos = sub_k_pos // pack_factor                 # [SUB_K]
            b_shifter_sub = (sub_k_pos % pack_factor) * 4         # [SUB_K]
            b_packed_sub = tl.load(
                b_ptr + b_expert_byte_offs + b_n_byte_offs + b_byte_pos[:, None])
            b_nibble_sub = (b_packed_sub >> b_shifter_sub[:, None]) & 0xF
            fp8_bits_sub = ((b_nibble_sub & 0x8) << 4) | ((b_nibble_sub & 0x7) << 2)
            b_sub_unscaled = fp8_bits_sub.to(tl.uint8).to(tl.float8e4nv, bitcast=True)

            # B-scale [BSN]：本 sub-block 的 [BSN] 向量（E8M0 uint8）
            sub_in_bounds = (k * BLOCK_SIZE_K + sk * group_size) < K
            b_scale_u8 = tl.load(
                b_scale_ptr + b_scale_expert_n_offs + (k * NUM_SUB_K + sk),
                mask=sub_in_bounds,
                other=0)
            # E8M0 解码 + 2^6 补偿：实际乘子 = 2^((e+6) - 127) = 2^(e-127) * 2^6
            b_scale_fp32 = ((b_scale_u8.to(tl.uint32) + 6) << 23).to(
                tl.float32, bitcast=True)

            # 把 scale 融入 b 后 cast 回 fp8——与原 kernel 的 fp8 round-trip 完全一致。
            # 中间 [SUB_K, BSN] FP32 仅 ~4KB（vs 原版 [BSK, BSN] FP32 ~64KB）。
            b_sub_scaled = b_sub_unscaled.to(tl.float32) * b_scale_fp32[None, :]
            b_sub_fp8 = b_sub_scaled.to(tl.float8e4nv)

            # 关键：用 acc=sub_acc 累加，让 Triton 把 4 个 wgmma fold 成一个 chain，
            # max_num_imprecise_acc=32 与 fp8 默认值一致，明确告知编译器期望
            # "fp16 acc 累加 ≤32 次后 flush 到 fp32"，与原 kernel 行为对齐。
            sub_acc = tl.dot(a_sub, b_sub_fp8, sub_acc,
                             max_num_imprecise_acc=32)

        # K-block 结束才乘 a_scale 加到主 acc——与原 kernel `acc += dot(...) * scale`
        # 完全相同的累加顺序，浮点求和误差也保持一致。
        if not quant_a_per_tensor:
            accumulator += sub_acc * a_scale[:, None]
        else:
            accumulator += sub_acc * a_scale

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + N * offs_token[:, None] + offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def fused_moe_wfp4afp8(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        mul_routed_weight: bool,
        top_k: int,
        config: Dict[str, Any],
        compute_type: tl.dtype,
        quant_a_per_tensor: bool,
        block_shape: List[int],
):
    # A:        [m, K]                     fp8_e4m3
    # B:        [E, N, K/2]                uint8 (packed FP4 E2M1)
    # C:        [m, top_k, N]              fp16 / bf16
    # A_scale:  [1] 或 [m, K/128]          fp32
    # B_scale:  [E, N, K/group_size]       uint8 (E8M0)
    # block_shape: [_, group_size]，group_size 必须为 32（MXFP4 标准）
    K = A.shape[1]
    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.shape[0],
                 A.shape[0] * top_k * config['BLOCK_SIZE_M'])

    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.shape[1], META['BLOCK_SIZE_N']), )
    kernel = fused_moe_wfp4afp8_kernel[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        K,
        EM,
        topk_ids.numel(),
        quant_a_per_tensor=quant_a_per_tensor,
        block_k_diviable=K % config["BLOCK_SIZE_K"] == 0,
        group_size=block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        **config,
    )
    return C, kernel


def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--BLOCK_SIZE_M', type=int, required=True)
    parser.add_argument('--BLOCK_SIZE_N', type=int, required=True)
    parser.add_argument('--BLOCK_SIZE_K', type=int, required=True)
    parser.add_argument('--GROUP_SIZE_M', type=int, required=True)

    parser.add_argument('--MUL_ROUTED_WEIGHT', type=str_to_bool, required=True)
    parser.add_argument('--top_k', type=int, required=True)

    parser.add_argument('--compute_type',
                        type=str,
                        required=True,
                        choices=["FP16", "BF16"])

    # MXFP4 group_size 固定为 32（OCP MX 标准），保留 CLI 参数仅做参数固化。
    parser.add_argument('--group_size', type=int, required=True, choices=[32])

    parser.add_argument('--m', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--n', type=int, required=True)

    parser.add_argument('--num_experts', type=int, required=True)

    parser.add_argument("--quant_a_per_tensor", type=str_to_bool, required=True)

    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--deep_tune", action="store_true")

    args = parser.parse_args()
    if args.compute_type == "FP16":
        args.torch_dtype = torch.float16
        args.triton_compute_type = tl.float16
    elif args.compute_type == "BF16":
        args.torch_dtype = torch.bfloat16
        args.triton_compute_type = tl.bfloat16
    return args


if __name__ == "__main__":
    set_global_seed(0)
    args = args_config()

    m = args.m
    k = args.k
    n = args.n
    num_experts = args.num_experts
    topk = args.top_k
    mul_routed_weight = args.MUL_ROUTED_WEIGHT
    compute_type = args.triton_compute_type
    block_shape = [0, args.group_size]
    group_size = args.group_size
    input_dtype = args.torch_dtype
    numel = m * topk
    quant_a_per_tensor = args.quant_a_per_tensor

    config = {
        "BLOCK_SIZE_M": args.BLOCK_SIZE_M,
        "BLOCK_SIZE_N": args.BLOCK_SIZE_N,
        "BLOCK_SIZE_K": args.BLOCK_SIZE_K,
        "GROUP_SIZE_M": args.GROUP_SIZE_M
    }

    em = numel + num_experts * (config["BLOCK_SIZE_M"] - 1)
    max_num_m_blocks = (em + config["BLOCK_SIZE_M"] -
                        1) // config["BLOCK_SIZE_M"]

    A = torch.randn([m, k], device='cuda',
                    dtype=torch.float32).to(torch.float8_e4m3fn)
    # B：MXFP4 packed uint8，shape [E, N, K/2]
    B = torch.randint(0,
                      256, [num_experts, n, k // 2],
                      device='cuda',
                      dtype=torch.uint8)
    C = torch.empty([m, topk, n], device='cuda', dtype=input_dtype)
    A_scale = torch.randn([m, k // 128], device='cuda', dtype=torch.float32)
    if quant_a_per_tensor:
        A_scale = A_scale.flatten()[0:1]
    # B_scale：E8M0 uint8，shape [E, N, K/group_size]
    B_scale = torch.randint(0,
                            256, [num_experts, n, k // group_size],
                            device='cuda',
                            dtype=torch.uint8)

    topk_weights = torch.rand([m, topk], device='cuda', dtype=torch.float32)
    sorted_token_ids = torch.randint(size=[em],
                                     low=0,
                                     high=num_experts,
                                     device='cuda',
                                     dtype=torch.int32)
    expert_ids = torch.randint(size=[max_num_m_blocks],
                               low=0,
                               high=num_experts,
                               device='cuda',
                               dtype=torch.int32)
    # 注意：这里必须给 num_tokens_post_padded 一个真实值，
    # 否则 kernel 第一条 `if pid_m * BSM >= num_tokens_post_padded: return`
    # 会让所有 program 立即退出，performance_kernel 测出来的全是 ~0us 噪声，
    # 导致 tune 完全失效（所有 num_warps/num_stages 候选时间相同，
    # 最终永远回落到 default config）。
    num_tokens_post_padded = torch.zeros([1], device='cuda', dtype=torch.int32)
    topk_ids = torch.randint(size=[m, topk],
                             low=0,
                             high=num_experts,
                             device='cuda',
                             dtype=torch.int32)

    # 始终生成真实 routing layout；`--deep_tune` flag 保留向后兼容但不再
    # 是必要条件（旧调用方不传也能拿到正确 tune 时间）。
    moe_align_block_cpu(topk_ids.flatten(), expert_ids, sorted_token_ids,
                        num_tokens_post_padded, m, topk, num_experts,
                        config["BLOCK_SIZE_M"])

    candidate_configs = {
        "configs": [],
        "default": {
            "config": config,
            "kernel_time": 0.0,  # us
            "kernel": None,
        }
    }

    def fn():
        return fused_moe_wfp4afp8(A, B, C, A_scale, B_scale, topk_weights,
                                  topk_ids, sorted_token_ids, expert_ids,
                                  num_tokens_post_padded, mul_routed_weight,
                                  topk, config, compute_type,
                                  quant_a_per_tensor, block_shape)

    default_kernel, kernel_time, output_tensor = performance_kernel(fn)
    candidate_configs["default"]["kernel_time"] = kernel_time
    candidate_configs["default"]["kernel"] = default_kernel

    for num_warps in [2, 4, 8]:
        for num_stages in [2, 3, 4, 5, 6]:
            opt_config = {"num_warps": num_warps, "num_stages": num_stages}
            opt_config.update(config)

            def fn():
                return fused_moe_wfp4afp8(A, B, C, A_scale, B_scale,
                                          topk_weights, topk_ids,
                                          sorted_token_ids, expert_ids,
                                          num_tokens_post_padded,
                                          mul_routed_weight, topk, opt_config,
                                          compute_type, quant_a_per_tensor,
                                          block_shape)

            kernel, kernel_time, output_tensor = performance_kernel(fn)
            candidate_configs["configs"].append({
                "config": opt_config,
                "kernel_time": kernel_time,
                "kernel": kernel
            })

    opt_best_kernel_time = sys.float_info.max
    opt_best_kerenl_config = None
    opt_best_kernel = None
    for config in candidate_configs["configs"]:
        if opt_best_kernel_time > config["kernel_time"]:
            opt_best_kernel_time = config["kernel_time"]
            opt_best_kerenl_config = config
            opt_best_kernel = config["kernel"]

    kernel_name = "fused_moe_wfp4afp8_kernel" + \
                  f"_BLOCK_SIZE_M_{args.BLOCK_SIZE_M}" + \
                  f"_BLOCK_SIZE_N_{args.BLOCK_SIZE_N}" + \
                  f"_BLOCK_SIZE_K_{args.BLOCK_SIZE_K}" + \
                  f"_GROUP_SIZE_M_{args.GROUP_SIZE_M}" + \
                  f"_MUL_ROUTED_WEIGHT_{args.MUL_ROUTED_WEIGHT}" + \
                  f"_top_k_{args.top_k}" + \
                  f"_compute_type_{args.compute_type}" + \
                  f"_group_size_{args.group_size}" + \
                  f"_quant_a_per_tensor_{args.quant_a_per_tensor}"
    dump_kernel(default_kernel, args.output_dir, kernel_name,
                candidate_configs["default"]["config"])

    if opt_best_kernel_time > candidate_configs["default"]["kernel_time"]:
        opt_best_kernel_time = sys.float_info.max
        opt_best_kerenl_config = None

    if opt_best_kerenl_config is not None and args.tune:
        dump_kernel(opt_best_kernel, args.output_dir, kernel_name,
                    opt_best_kerenl_config)
        logging.info("Found best config after tuning")
        logging.info(opt_best_kerenl_config)
        logging.info(
            f"Tuned best config average latency: {opt_best_kernel_time} us")
        logging.info(
            f"Default config average latency: {candidate_configs['default']['kernel_time']} us"
        )
    else:
        logging.info("Using default config")
        logging.info(candidate_configs["default"]["config"])
        logging.info(
            f"Average latency: {candidate_configs['default']['kernel_time']} us"
        )
    exit(0)
