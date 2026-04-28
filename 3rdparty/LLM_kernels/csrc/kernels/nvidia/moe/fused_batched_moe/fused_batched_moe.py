# Copyright 2026 Tencent Inc. All rights reserved.
# Copyright 2026 vLLM Team
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
# [vLLM Project] https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_batched_moe.py
#
# ==============================================================================

import argparse
import os
import sys
import torch
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

os.environ["PATH"] = "/usr/local/nvidia/lib64:" + os.environ["PATH"]

import triton
import triton.language as tl

current_dir = os.path.dirname(os.path.abspath(__file__))
common_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, current_dir)
sys.path.insert(0, common_dir)
from common import dump_kernel, str_to_bool, set_global_seed, performance_kernel


@triton.jit(do_not_specialize=["K", "expert_id", "stride_ak", "stride_bk",
                                "stride_ase", "stride_asm", "stride_ask",
                                "stride_bse", "stride_bsk", "stride_bsn"])
def moe_mmk(
    a_ptrs,
    b_ptrs,
    K,
    expert_id,
    a_scale_ptr,
    b_scale_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_ak: tl.int64,
    stride_bk: tl.int64,
    stride_ase: tl.int64,
    stride_asm: tl.int64,
    stride_ask: tl.int64,
    stride_bse: tl.int64,
    stride_bsk: tl.int64,
    stride_bsn: tl.int64,
    # Offsets and masks
    offs_m,
    offs_n,
    offs_bn,
    mask_m,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    compute_type: tl.constexpr,
    use_w8a8: tl.constexpr,
    use_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
):
    offs_k = tl.arange(0, BLOCK_K)

    if use_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + expert_id * stride_bse + offs_n[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = b_scale_ptr + offs_bsn * stride_bsn

        # per act token
        elif per_act_token_quant:
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=mask_m, other=0.0)[:, None]

            b_scale_ptrs = b_scale_ptr + offs_bn[None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)

        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & (offs_k[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        # We accumulate along the K dimension.
        if use_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=mask_m, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                # acc used to enable fp8_fast_accum
                accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if use_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)

    return accumulator


@triton.jit(do_not_specialize=["expert_id", "M", "N", "K",
                                "stride_am", "stride_ak", "stride_bk", "stride_bn",
                                "stride_cm", "stride_cn", "stride_ase", "stride_asm",
                                "stride_ask", "stride_bse", "stride_bsk", "stride_bsn",
                                "group_n", "group_k"])
def expert_triton_kernel(
    a_ptr,  # [max_tokens, K]
    b_ptr,  # [K, N]
    c_ptr,  # [max_tokens, N]
    expert_id,
    compute_type: tl.constexpr,
    # Dimensions
    M,
    N,
    K,
    # Quantization data
    a_scale_ptr,
    b_scale_ptr,
    b_zp_ptr,
    # strides
    stride_am: tl.int64,
    stride_ak: tl.int64,
    stride_bk: tl.int64,
    stride_bn: tl.int64,
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    stride_ase: tl.int64,
    stride_asm: tl.int64,
    stride_ask: tl.int64,
    stride_bse: tl.int64,
    stride_bsk: tl.int64,
    stride_bsn: tl.int64,
    # offsets
    offs_bn,
    # Blockwise quantization data
    group_n,
    group_k,
    # Quantization schemes
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
    # Kernel config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) % N
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    # Make grids of a + b pointers
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = moe_mmk(
        a_ptrs,
        b_ptrs,
        K,
        expert_id,
        a_scale_ptr,
        b_scale_ptr,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_ak,
        stride_bk,
        stride_ase,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Offsets and masks
        offs_m,
        offs_n,
        offs_bn,
        mask_m,
        # Block size for block-wise quantization
        group_n,
        group_k,
        # Meta-parameters
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a16,
        per_act_token_quant,
    )

    # store in C
    offs_cn = tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = mask_m[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# Adapted from vLLM Project function batched_triton_kernel
@triton.jit(do_not_specialize=["expert_num_tokens", "max_num_tokens",
                                "K", "N",
                                "stride_ae", "stride_am", "stride_ak",
                                "stride_be", "stride_bk", "stride_bn",
                                "stride_ce", "stride_cm", "stride_cn",
                                "stride_ase", "stride_asm", "stride_ask",
                                "stride_bse", "stride_bsk", "stride_bsn"])
def fused_batched_moe_kernel(
    a_ptr,  # [E, max_num_tokens, K]
    b_ptr,  # [E, K, N]
    c_ptr,  # [E, max_num_tokens, N]
    expert_num_tokens,  # [E]
    compute_type: tl.constexpr,
    # Dimensions
    max_num_tokens,
    K,
    N,
    # Quantization data - 这些参数必须保留以保持与C++接口一致
    a_scale_ptr,  # 即使为None也必须传递
    b_scale_ptr,  # 即使为None也必须传递
    b_zp_ptr,     # 即使为None也必须传递
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_ae: tl.int64,
    stride_am: tl.int64,
    stride_ak: tl.int64,
    stride_be: tl.int64,
    stride_bk: tl.int64,
    stride_bn: tl.int64,
    stride_ce: tl.int64,
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    stride_ase: tl.int64,
    stride_asm: tl.int64,
    stride_ask: tl.int64,
    stride_bse: tl.int64,
    stride_bsk: tl.int64,
    stride_bsn: tl.int64,
    # Blockwise quantization data
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Quantization schemes
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
    # Kernel config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert_id = tl.program_id(axis=0)
    e_num_tokens = tl.load(expert_num_tokens + expert_id)
    if e_num_tokens == 0:
        # Early exit
        return

    # axis 1 is M_blocks * N_blocks
    pid_mn = tl.program_id(axis=1)
    # num_pid_m = tl.cdiv(max_num_tokens, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    cta_m_start = pid_m * BLOCK_M
    cta_n_start = pid_n * BLOCK_N
    if cta_m_start >= e_num_tokens:
        # Early exit
        return

    cta_m_size = min(BLOCK_M, e_num_tokens - cta_m_start)
    cta_n_size = min(BLOCK_N, N - cta_n_start)

    a_ptr = a_ptr + expert_id * stride_ae + cta_m_start * stride_am
    b_ptr = b_ptr + expert_id * stride_be + cta_n_start * stride_bn
    c_ptr = (
        c_ptr
        + expert_id * stride_ce
        + cta_m_start * stride_cm
        + cta_n_start * stride_cn
    )

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)) % N

    if use_fp8_w8a8:
        a_scale_ptr = a_scale_ptr + expert_id * stride_ase
        b_scale_ptr = b_scale_ptr + expert_id * stride_bse

        # block-wise
        if group_k > 0 and group_n > 0 or per_act_token_quant:
            a_scale_ptr = a_scale_ptr + cta_m_start * stride_asm

    expert_triton_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        expert_id,
        compute_type,
        cta_m_size,  # M
        cta_n_size,  # N
        K,  # K
        a_scale_ptr,
        b_scale_ptr,
        b_zp_ptr,
        # Strides
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_ase,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # offsets
        offs_bn,
        # Blockwise quantization data
        group_n,
        group_k,
        # Quantization schemes
        use_fp8_w8a8,
        use_int8_w8a16,
        per_act_token_quant,
        # Kernel config
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )


def invoke_fused_batched_moe_kernel(
    A: torch.Tensor,  # [E, max_tokens, K]
    B: torch.Tensor,  # [E, N, K]
    C: torch.Tensor,  # [E, max_tokens, N]
    expert_num_tokens: torch.Tensor,  # [E]
    compute_type: tl.dtype,
    # Quantization data
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor,
    # Quantization schemes
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    config: Dict[str, int],
    per_act_token_quant: bool,
    block_shape: List[int] | None = None,
):
    assert not use_int4_w4a16
    max_num_tokens = A.size(1)
    K = A.size(2)
    N = C.size(2)

    BLOCK_M = config["BLOCK_SIZE_M"]
    BLOCK_N = config["BLOCK_SIZE_N"]
    BLOCK_K = config["BLOCK_SIZE_K"]

    grid = (
        expert_num_tokens.size(0),
        triton.cdiv(max_num_tokens, BLOCK_M) * triton.cdiv(B.size(1), BLOCK_N),
    )

    # Normalize A_scale shape (只有当A_scale不是dummy tensor时才需要)
    if A_scale is not None and A_scale.numel() > 1:
        if A_scale.ndim == 1:
            assert A_scale.numel() == expert_num_tokens.shape[0]
            A_scale = A_scale.view(-1, 1, 1)
        elif A_scale.ndim == 2:
            A_scale = A_scale.unsqueeze(0).expand(expert_num_tokens.shape[0], -1, -1)

    if B_scale is not None and B_scale.numel() > 1 and B_scale.ndim == 1:
        assert B_scale.numel() == expert_num_tokens.shape[0]
        B_scale = B_scale.view(-1, 1, 1)

    # 只对非dummy tensor进行shape检查
    if A_scale.numel() > 1:
        assert A_scale.ndim == 3, f"A_scale shape: {A_scale.shape}"
    if B_scale.numel() > 1:
        assert B_scale.ndim == 1 or B_scale.ndim == 3, f"B_scale shape: {B_scale.shape}"

    if B_scale.numel() > 1:
        if B_scale.ndim == 1:
            stride_bse = 1
            stride_bsk = 0
            stride_bsn = 0
        else:
            stride_bse = B_scale.stride(0)
            stride_bsk = B_scale.stride(2)
            stride_bsn = B_scale.stride(1)

    else:
        stride_bse = 0
        stride_bsk = 0
        stride_bsn = 0

    if A_scale.numel() > 1:
        stride_ase = A_scale.stride(0)
        stride_asm = A_scale.stride(1)
        stride_ask = A_scale.stride(2)
    else:
        stride_ase = 0
        stride_asm = 0
        stride_ask = 0

    logging.debug("Invoke fused batched moe kernel")
    logging.debug(f"A.shape: {A.shape}, B.shape: {B.shape}, C.shape: {C.shape}")
    logging.debug(f"A.stride: {A.stride()}, B.stride: {B.stride()}, C.stride: {C.stride()}")
    logging.debug(f"expert_num_tokens.shape: {expert_num_tokens.shape}")
    logging.debug(f"max_num_tokens: {max_num_tokens}, K: {K}, N: {N}")

    kernel = fused_batched_moe_kernel[grid](
        A,
        B,
        C,
        expert_num_tokens,
        compute_type,
        # Dimensions
        max_num_tokens,
        K,
        N,
        # Quantization data - 现在总是非None
        A_scale,
        B_scale,
        B_zp,
        # Strides
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        stride_ase,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Blockwise quantization data
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        # Quantization schemes
        use_fp8_w8a8,
        use_int8_w8a16,
        per_act_token_quant,
        # Kernel config
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C, kernel


def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_n', type=int, default=0)
    parser.add_argument('--group_k', type=int, default=0)
    parser.add_argument('--BLOCK_SIZE_M', type=int, required=True)
    parser.add_argument('--BLOCK_SIZE_N', type=int, required=True)
    parser.add_argument('--BLOCK_SIZE_K', type=int, required=True)
    parser.add_argument('--GROUP_SIZE_M', type=int, required=True)
    parser.add_argument('--num_warps', type=int, default=0)
    parser.add_argument('--num_stages', type=int, default=0)
    parser.add_argument('--top_k', type=int, required=True)
    parser.add_argument('--compute_type',
                        type=str,
                        required=True,
                        choices=["FP16", "BF16"])
    parser.add_argument('--use_fp8_w8a8', type=str_to_bool, required=True)
    parser.add_argument('--use_int8_w8a16', type=str_to_bool, required=True)
    parser.add_argument('--m', type=int, default=128)
    parser.add_argument('--k', type=int, default=7168)
    parser.add_argument('--n', type=int, default=4096)
    parser.add_argument('--num_experts', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default="./")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    if args.compute_type == "FP16":
        args.torch_dtype = torch.float16
        args.triton_compute_type = tl.float16
    elif args.compute_type == "BF16":
        args.torch_dtype = torch.bfloat16
        args.triton_compute_type = tl.bfloat16
    return args


def test_basic_functionality(save_data=False, load_data=False, data_dir="./test_data"):
    """
    基础功能测试 - 对应C++测试中的BasicFunctionalityTest
    使用与C++测试相同的参数进行测试
    
    Args:
        save_data: 是否保存测试数据到文件，供C++测试使用
        load_data: 是否从文件加载测试数据
        data_dir: 数据保存/加载目录
    """
    logging.info("===== Testing InvokeFusedBatchedMoe Basic Functionality =====")
    
    # 测试参数设置 - 与C++测试保持一致
    num_experts = 32
    max_num_tokens = 4
    hidden_size = 2048  # k
    inter_size = 1408   # n
    topk = 8
    use_fp8_w8a8 = False
    use_int8_w8a16 = False
    use_int4_w4a16 = False
    group_size = 1
    
    logging.info(f"num_experts={num_experts}, max_num_tokens={max_num_tokens}, "
                 f"hidden_size={hidden_size}, inter_size={inter_size}, topk={topk}")
    
    # 设置随机种子以保证可重复性
    set_global_seed(42)
    
    if load_data:
        # 从文件加载数据
        logging.info(f"Loading test data from {data_dir}...")
        A = torch.load(os.path.join(data_dir, "input.pt"))
        B_weight1 = torch.load(os.path.join(data_dir, "weight1.pt"))
        B_weight2 = torch.load(os.path.join(data_dir, "weight2.pt"))
        expert_num_tokens = torch.load(os.path.join(data_dir, "expert_num_tokens.pt"))
        logging.info("Test data loaded successfully")
    else:
        # 生成随机数据
        # 创建输入张量 [num_experts, max_num_tokens, hidden_size]
        A = torch.rand([num_experts, max_num_tokens, hidden_size],
                       device='cuda', dtype=torch.float16)
        
        # 创建权重张量
        # weight1: [num_experts, inter_size * 2, hidden_size]
        # 注意：Python中B的形状是[E, N, K]，所以需要转置
        B_weight1 = torch.rand([num_experts, inter_size * 2, hidden_size],
                               device='cuda', dtype=torch.float16)
        
        # weight2: [num_experts, hidden_size, inter_size]
        B_weight2 = torch.rand([num_experts, hidden_size, inter_size],
                               device='cuda', dtype=torch.float16)
        
        # 为每个专家分配随机数量的tokens
        expert_num_tokens = torch.randint(0, max_num_tokens + 1, [num_experts],
                                          device='cuda', dtype=torch.int32)
        
        if save_data:
            # 保存数据到文件供C++测试使用
            os.makedirs(data_dir, exist_ok=True)
            torch.save(A, os.path.join(data_dir, "input.pt"))
            torch.save(B_weight1, os.path.join(data_dir, "weight1.pt"))
            torch.save(B_weight2, os.path.join(data_dir, "weight2.pt"))
            torch.save(expert_num_tokens, os.path.join(data_dir, "expert_num_tokens.pt"))
            logging.info(f"Test data saved to {data_dir}")
    
    # 创建输出张量
    C_intermediate = torch.zeros([num_experts, max_num_tokens, inter_size * 2],
                                 device='cuda', dtype=torch.float16)
    C_output = torch.zeros([num_experts, max_num_tokens, hidden_size],
                           device='cuda', dtype=torch.float16)
    
    logging.info(f"Expert token counts: {expert_num_tokens.cpu().tolist()}")
    
    # 配置kernel参数
    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1
    }
    
    compute_type = tl.float16
    block_shape = [0, 0]  # 不使用block-wise量化
    
    # 同步确保数据传输完成
    torch.cuda.synchronize()
    logging.info("Synchronizing before invoke_fused_batched_moe_kernel...")
    
    # 第一次调用：A @ B_weight1 -> C_intermediate
    logging.info("Starting first kernel call (input @ weight1)...")
    try:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        C_intermediate, kernel1 = invoke_fused_batched_moe_kernel(
            A, B_weight1, C_intermediate, expert_num_tokens, compute_type,
            None, None, None,  # 不使用量化
            use_fp8_w8a8, use_int8_w8a16, use_int4_w4a16,
            config, False, block_shape
        )
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event)
        logging.info(f"First kernel call completed in {elapsed_time:.3f} ms")
        
    except Exception as e:
        logging.error(f"Error in first kernel call: {e}")
        raise
    
    # 验证中间输出
    intermediate_output = C_intermediate.cpu()
    nonzero_count = torch.count_nonzero(intermediate_output).item()
    total_elements = intermediate_output.numel()
    avg_abs_value = torch.abs(intermediate_output[intermediate_output != 0]).mean().item() if nonzero_count > 0 else 0.0
    
    logging.info(f"Intermediate output statistics: nonzero_count={nonzero_count}/{total_elements}, "
                 f"avg_abs_value={avg_abs_value:.6f}")
    
    assert nonzero_count > 0, "Intermediate output should contain non-zero values"
    
    # 应用SiLU激活函数（简化版本，实际应该是gated SiLU）
    # 在实际实现中，这部分应该在kernel内部完成
    # 这里我们简单地对前半部分应用SiLU
    silu_input = C_intermediate[:, :, :inter_size]
    gate_input = C_intermediate[:, :, inter_size:]
    silu_output = silu_input * torch.sigmoid(silu_input) * gate_input
    
    # 第二次调用：silu_output @ B_weight2 -> C_output
    logging.info("Starting second kernel call (silu_output @ weight2)...")
    try:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        C_output, kernel2 = invoke_fused_batched_moe_kernel(
            silu_output, B_weight2, C_output, expert_num_tokens, compute_type,
            None, None, None,  # 不使用量化
            use_fp8_w8a8, use_int8_w8a16, use_int4_w4a16,
            config, False, block_shape
        )
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event)
        logging.info(f"Second kernel call completed in {elapsed_time:.3f} ms")
        
    except Exception as e:
        logging.error(f"Error in second kernel call: {e}")
        raise
    
    # 验证最终输出
    output = C_output.cpu()
    nonzero_count = torch.count_nonzero(output).item()
    total_elements = output.numel()
    avg_abs_value = torch.abs(output[output != 0]).mean().item() if nonzero_count > 0 else 0.0
    
    logging.info(f"Final output statistics: nonzero_count={nonzero_count}/{total_elements}, "
                 f"avg_abs_value={avg_abs_value:.6f}")
    
    assert nonzero_count > 0, "Final output should contain non-zero values"
    
    logging.info("InvokeFusedBatchedMoe basic functionality test PASSED")
    
    # 如果需要保存输出结果用于对拍
    if save_data:
        torch.save(C_intermediate, os.path.join(data_dir, "intermediate_output.pt"))
        torch.save(C_output, os.path.join(data_dir, "final_output.pt"))
        logging.info(f"Output data saved to {data_dir}")
    
    return True


if __name__ == "__main__":
    import sys
    
    # 检查是否运行测试模式
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        logging.info("Running in test mode...")
        
        # 解析额外参数
        save_data = "--save" in sys.argv
        load_data = "--load" in sys.argv
        data_dir = "./test_data"
        
        # 查找data_dir参数
        for i, arg in enumerate(sys.argv):
            if arg == "--data_dir" and i + 1 < len(sys.argv):
                data_dir = sys.argv[i + 1]
                break
        
        try:
            test_basic_functionality(save_data=save_data, load_data=load_data, data_dir=data_dir)
            logging.info("All tests passed!")
            sys.exit(0)
        except Exception as e:
            logging.error(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # 原有的性能测试代码
    set_global_seed(42)
    args = args_config()

    m = args.m
    k = args.k
    n = args.n
    num_experts = args.num_experts
    topk = args.top_k
    compute_type = args.triton_compute_type
    block_shape = [args.group_n, args.group_k]
    use_fp8_w8a8 = args.use_fp8_w8a8
    use_int8_w8a16 = args.use_int8_w8a16

    config = {
        "BLOCK_SIZE_M": args.BLOCK_SIZE_M,
        "BLOCK_SIZE_N": args.BLOCK_SIZE_N,
        "BLOCK_SIZE_K": args.BLOCK_SIZE_K,
        "GROUP_SIZE_M": args.GROUP_SIZE_M
    }
    if args.num_warps > 0:
        config["num_warps"] = args.num_warps
    if args.num_stages > 0:
        config["num_stages"] = args.num_stages

    max_num_tokens = m
    
    if args.use_fp8_w8a8:
        input_dtype = torch.float8_e4m3fn
    else:
        input_dtype = args.torch_dtype

    # Create test tensors
    A = torch.rand([num_experts, max_num_tokens, k], device='cuda', dtype=torch.float32).to(input_dtype)
    B = torch.rand([num_experts, n, k], device='cuda', dtype=torch.float32).to(input_dtype)
    C = torch.zeros([num_experts, max_num_tokens, n], device='cuda', dtype=args.torch_dtype)
    expert_num_tokens = torch.randint(1, max_num_tokens + 1, [num_experts], device='cuda', dtype=torch.int32)
    
    A_scale = torch.rand([num_experts, max_num_tokens, k // 128], device='cuda', dtype=torch.float32) if use_fp8_w8a8 else None
    B_scale = torch.rand([num_experts, n // 128, k // 128], device='cuda', dtype=torch.float32) if use_fp8_w8a8 or use_int8_w8a16 else None
    B_zp = None


    # 为了保持与C++接口的兼容性，当scale参数为None时，传递dummy tensor
    # 这样可以防止Triton编译器优化掉这些参数，确保PTX/CUBIN中保留所有参数位置
    # 必须在函数开始时就创建，确保所有后续操作都使用GPU tensor
    if A_scale is None:
        A_scale = torch.empty(1, device='cuda', dtype=torch.float32)
    if B_scale is None:
        B_scale = torch.empty(1, device='cuda', dtype=torch.float32)
    if B_zp is None:
        B_zp = torch.empty(1, device='cuda', dtype=torch.float32)

    def fn():
        return invoke_fused_batched_moe_kernel(
            A, B, C, expert_num_tokens, compute_type,
            A_scale, B_scale, B_zp,
            use_fp8_w8a8, use_int8_w8a16, False,
            config, False, block_shape
        )

    kernel, kernel_time, output_tensor = performance_kernel(fn)

    kernel_name = "fused_batched_moe_kernel" + \
                  f"_group_n_{block_shape[0]}" + \
                  f"_group_k_{block_shape[1]}" + \
                  f"_BLOCK_SIZE_M_{args.BLOCK_SIZE_M}" + \
                  f"_BLOCK_SIZE_N_{args.BLOCK_SIZE_N}" + \
                  f"_BLOCK_SIZE_K_{args.BLOCK_SIZE_K}" + \
                  f"_GROUP_SIZE_M_{args.GROUP_SIZE_M}" + \
                  f"_top_k_{topk}" + \
                  f"_compute_type_{args.compute_type}" + \
                  f"_use_fp8_w8a8_{args.use_fp8_w8a8}" + \
                  f"_use_int8_w8a16_{args.use_int8_w8a16}"
    
    dump_kernel(kernel, args.output_dir, kernel_name, config)
    logging.info(f"Kernel name: {kernel_name}")
    logging.info(f"Config: {config}")
    logging.info(f"Average latency: {kernel_time} us")