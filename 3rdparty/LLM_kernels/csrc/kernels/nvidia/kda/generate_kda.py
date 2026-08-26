#!/usr/bin/env python3
# Copyright 2026 Tencent Inc.  All rights reserved.
"""Generate Triton cubin/json/ptx artifacts for KDA kernels.

The AOT kernel contract uses:
  * BF16 inputs; the recurrent SSM state is FP32 by default and BF16 optionally
  * K = V = 128
  * local head counts selected by the creator script
  * chunk size = 64 and sub-chunk size = 16

The SSM state dtype is a Triton specialization: the state pointers appear in the
compiled signature, so every kernel that reads or writes the recurrent state is
generated once per requested dtype and carries an ``_SSM_{fp32,bf16}`` suffix.

T is deliberately kept dynamic by the original SGLang kernels. Kernel file
names contain every model/mode and structural tile specialization used by the
C++ KDA wrapper.

SGLang autotune decorators are replaced by an explicit build-time search over
num_warps/num_stages. Structural tile sizes are fixed below and shared with the
C++ wrapper through artifact names. Stride values needed by the C++ interface
remain runtime parameters instead of Python-JIT constants. The Triton kernel
computation bodies are otherwise unchanged.
"""

import argparse
import os
import sys
from typing import Callable, Optional

import torch
import triton
import triton.language as tl

# Triton >= 3.2 removed the GDC API. Stub the missing functions so that
# kernels guarded by ``USE_GDC=False`` still compile under newer Triton.
if not hasattr(tl.extra.cuda, "gdc_wait"):
    tl.extra.cuda.gdc_wait = lambda: None
if not hasattr(tl.extra.cuda, "gdc_launch_dependents"):
    tl.extra.cuda.gdc_launch_dependents = lambda: None

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_NVIDIA_KERNEL_DIR = os.path.dirname(_CURRENT_DIR)
sys.path.insert(0, _CURRENT_DIR)
sys.path.insert(0, _NVIDIA_KERNEL_DIR)

from chunk_delta_h import (  # noqa: E402
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from causal_conv1d import (  # noqa: E402
    PAD_SLOT_ID as CAUSAL_CONV_PAD_SLOT_ID,
    causal_conv1d_fwd_kernel,
    causal_conv1d_update_kernel,
)
from chunk_intra import (  # noqa: E402
    chunk_kda_fwd_kernel_inter_solve_fused,
)
from chunk_intra_token_parallel import (  # noqa: E402
    chunk_kda_fwd_kernel_intra_token_parallel,
)
from fused_recurrent import fused_recurrent_kda_packed_decode_kernel  # noqa: E402
from fused_sigmoid_gating_recurrent import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update_kernel,
)
from fused_norm_gate import layer_norm_gated_fwd_kernel  # noqa: E402
from kda_kernel import (  # noqa: E402
    chunk_gla_fwd_kernel_o,
    kda_gate_chunk_cumsum_vector_kernel,
    recompute_w_u_fwd_kernel,
)
from l2norm import l2norm_fwd_kernel  # noqa: E402
from moe.common import dump_kernel, performance_kernel  # noqa: E402


HEAD_DIM = 128
CHUNK_SIZE = 64
SUB_CHUNK_SIZE = 16
RCP_LN2 = 1.4426950216293335
L2_EPS = 1e-6
SOFTPLUS_BETA = 1.0
SOFTPLUS_THRESHOLD = 20.0

# Gate activation variants. ``None`` compiles the default softplus gate
# ``-exp(A_log) * softplus(x)``; a float compiles the bounded sigmoid gate
# ``lower_bound * sigmoid(exp(A_log) * x)``. The bound itself stays a runtime
# argument of the bounded-gate artifacts, so this value only selects the AOT
# variant; the actual gate_lower_bound comes from model config at launch time.
GATE_LOWER_BOUNDS = (None, -5.0)

# Recurrent-state storage dtypes. FP32 is the reference precision of the delta
# rule and the KsanaLLM default; BF16 halves the state pool at the cost of
# precision. The short token appears in the artifact name and in the C++
# wrapper's SSM specialization key.
SSM_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
SSM_DTYPE_TOKENS = {
    "float32": "fp32",
    "bfloat16": "bf16",
}

# Structural tile sizes are part of the artifact key. Keep these values in
# sync with csrc/kernels/nvidia/kda_kernel_wrapper.cpp. The build-time search
# below only chooses launch parameters (num_warps and num_stages).
L2_BLOCK_T = 16
GATE_BLOCK_S = 32
INTRA_BLOCK_H = 1
INTER_BLOCK_K = 64
INTER_BLOCK_V = 64
RECOMPUTE_BLOCK_K = 64
RECOMPUTE_BLOCK_V = 64
STATE_BLOCK_V = 32
OUTPUT_BLOCK_K = 64
OUTPUT_BLOCK_V = 64
RECURRENT_BLOCK_K = 128
RECURRENT_BLOCK_V = 32
CAUSAL_CONV_WIDTH = 4
CAUSAL_CONV_STATE_LEN = CAUSAL_CONV_WIDTH - 1
CAUSAL_CONV_NP2_STATELEN = 4  # next_power_of_2(KERNEL_WIDTH - 1)
CAUSAL_CONV_BLOCK_M = 8
CAUSAL_CONV_BLOCK_N = 256
GATED_RMSNORM_BLOCK_T = 32


def _config(
    meta: Optional[dict[str, int]] = None,
    *,
    num_warps: int,
    num_stages: int = 3,
) -> dict:
    return {
        "meta": dict(meta or {}),
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def _runtime_arg_names(compiled) -> list[str]:
    return [
        str(arg_name)
        for arg_name, arg_type in compiled.src.signature.items()
        if arg_type != "constexpr"
    ]


def _tune_and_dump(
    output_dir: str,
    name: str,
    candidates: list[dict],
    launch: Callable[[dict], tuple[torch.Tensor, object]],
    expected_arg_names: list[str],
    reset: Optional[Callable[[], None]] = None,
) -> None:
    """Explicitly benchmark configs at build time and dump only the winner.

    KsanaLLM never executes Triton Python at inference time.  The original
    SGLang Triton autotune decorators are therefore replaced by this AOT
    search. Structural tile sizes are fixed in the artifact name; the standard
    json records only the winning launch parameters. No runtime config
    selection remains.
    """
    best = None
    for candidate in candidates:
        try:
            if reset is not None:
                reset()
            compiled, kernel_time_us, _ = performance_kernel(
                lambda: launch(candidate)
            )
            actual_arg_names = _runtime_arg_names(compiled)
            if actual_arg_names != expected_arg_names:
                raise RuntimeError(
                    f"AOT ABI mismatch for {name}: expected "
                    f"{expected_arg_names}, got {actual_arg_names}"
                )
            print(
                f"Candidate {name}: meta={candidate['meta']}, "
                f"num_warps={candidate['num_warps']}, "
                f"num_stages={candidate['num_stages']}, "
                f"latency={kernel_time_us:.1f} us"
            )
            if best is None or kernel_time_us < best[0]:
                best = (kernel_time_us, compiled, candidate)
        except Exception as error:
            print(f"Skip invalid KDA config for {name}: {candidate}: {error}")

    if best is None:
        raise RuntimeError(f"No valid AOT config found for {name}")

    kernel_time_us, compiled, candidate = best
    dump_kernel(
        compiled,
        output_dir,
        name,
        {
            "config": {
                "num_warps": candidate["num_warps"],
                "num_stages": candidate["num_stages"],
            }
        }
    )

    print(
        f"Generated {name}.{{cubin,json,ptx}} in {output_dir} "
        f"with launch config {candidate} (kernel_time={kernel_time_us:.1f} us)"
    )


def _make_prefill_tensors(num_heads: int, num_tokens: int = 128) -> dict:
    device = "cuda"
    dtype = torch.bfloat16
    shape = (1, num_tokens, num_heads, HEAD_DIM)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_g = torch.randn(shape, device=device, dtype=dtype)
    g = torch.empty(shape, device=device, dtype=torch.float32)
    beta = torch.sigmoid(
        torch.randn(1, num_tokens, num_heads, device=device, dtype=torch.float32)
    )
    cu_seqlens = torch.tensor([0, num_tokens], device=device, dtype=torch.int64)
    chunk_indices = torch.tensor([[0, 0], [0, 1]], device=device, dtype=torch.int32)
    chunk_offsets = torch.tensor([0, 2], device=device, dtype=torch.int32)
    state_indices = torch.zeros(1, device=device, dtype=torch.int32)
    return {
        "q": q,
        "k": k,
        "v": v,
        "raw_g": raw_g,
        "g": g,
        "beta": beta,
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
        "chunk_offsets": chunk_offsets,
        "state_indices": state_indices,
    }


def _generate_l2norm(output_dir: str) -> None:
    num_tokens = 128
    x = torch.randn(
        num_tokens, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    y = torch.empty_like(x)

    def launch(config):
        block_t = config["meta"]["BT"]
        kernel = l2norm_fwd_kernel[(triton.cdiv(num_tokens, block_t),)](
            x,
            y,
            L2_EPS,
            T=num_tokens,
            D=HEAD_DIM,
            BT=block_t,
            BD=HEAD_DIM,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
        return y, kernel

    _tune_and_dump(
        output_dir,
        f"l2norm_fwd_kernel_D_{HEAD_DIM}_BT_{L2_BLOCK_T}",
        [_config({"BT": L2_BLOCK_T}, num_warps=8, num_stages=3)],
        launch,
        ["x", "y", "eps", "T"],
    )


def _generate_gated_rmsnorm(output_dir: str) -> None:
    """Generate output RMSNorm followed by a sigmoid gate.

    The KDA output and ``g_proj`` are flattened to [T * H, 128] before this
    operation. D=128 and the activation are therefore AOT constants while the
    number of rows remains a runtime parameter.
    """
    num_rows = 128
    dtype = torch.bfloat16
    x = torch.randn(num_rows, HEAD_DIM, device="cuda", dtype=dtype)
    gate = torch.randn_like(x)
    output = torch.empty_like(x)
    weight = torch.randn(HEAD_DIM, device="cuda", dtype=dtype)
    rstd = torch.empty(num_rows, device="cuda", dtype=torch.float32)

    # Keep the complete source-level pointer ABI even though this specialization
    # has no bias or residual. The C++ wrapper supplies null values for these
    # dead arguments, which makes the ABI explicit and stable.
    def launch(config):
        kernel = layer_norm_gated_fwd_kernel[
            (triton.cdiv(num_rows, GATED_RMSNORM_BLOCK_T),)
        ](
            x=x,
            g=gate,
            y=output,
            w=weight,
            b=weight,
            residual=x,
            residual_out=output,
            mean=rstd,
            rstd=rstd,
            eps=1e-5,
            T=num_rows,
            D=HEAD_DIM,
            BT=GATED_RMSNORM_BLOCK_T,
            BD=HEAD_DIM,
            ACTIVATION="sigmoid",
            IS_RMS_NORM=True,
            STORE_RESIDUAL_OUT=False,
            HAS_RESIDUAL=False,
            HAS_WEIGHT=True,
            HAS_BIAS=False,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
        return output, kernel

    _tune_and_dump(
        output_dir,
        (
            "layer_norm_gated_fwd_kernel"
            f"_D_{HEAD_DIM}_BT_{GATED_RMSNORM_BLOCK_T}_ACT_sigmoid"
        ),
        [_config(num_warps=4, num_stages=2)],
        launch,
        [
            "x",
            "g",
            "y",
            "w",
            "b",
            "residual",
            "residual_out",
            "mean",
            "rstd",
            "eps",
            "T",
        ],
    )


def _generate_gate_cumsum(
    output_dir: str, num_heads: int, tensors: dict, lower_bound: Optional[float]
) -> None:
    raw_g = tensors["raw_g"]
    g = tensors["g"]
    num_tokens = g.shape[1]
    out = g
    a_log = torch.randn(num_heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(
        num_heads * HEAD_DIM, device="cuda", dtype=torch.float32
    )
    cu_seqlens = tensors["cu_seqlens"]
    chunk_indices = tensors["chunk_indices"]
    num_chunks = chunk_indices.shape[0]

    def grid(meta):
        return (
            triton.cdiv(HEAD_DIM, meta["BS"]),
            num_chunks,
            num_heads,
        )

    def launch(config):
        kernel = kda_gate_chunk_cumsum_vector_kernel[grid](
            s=raw_g,
            A_log=a_log,
            dt_bias=dt_bias,
            o=out,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
            T=num_tokens,
            H=num_heads,
            S=HEAD_DIM,
            BT=CHUNK_SIZE,
            BS=config["meta"]["BS"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
        return out, kernel

    candidates = [
        _config({"BS": GATE_BLOCK_S}, num_warps=num_warps)
        for num_warps in (2, 4, 8)
    ]
    # A None lower_bound is a Python constant that Triton drops from the
    # signature, so only the safe-gate artifact carries the runtime bound.
    expected_arg_names = [
        "s",
        "A_log",
        "dt_bias",
        "o",
        "scale",
        "cu_seqlens",
        "chunk_indices",
    ]
    if lower_bound is not None:
        expected_arg_names.append("lower_bound")
    expected_arg_names.append("T")
    _tune_and_dump(
        output_dir,
        (
            "kda_gate_chunk_cumsum_vector_kernel"
            f"_H_{num_heads}_BS_{GATE_BLOCK_S}"
            f"_LB_{int(lower_bound is not None)}"
        ),
        candidates,
        launch,
        expected_arg_names,
    )


def _generate_intra_token_parallel(
    output_dir: str, num_heads: int, tensors: dict
) -> None:
    q, k, g, beta = (
        tensors["q"],
        tensors["k"],
        tensors["g"],
        tensors["beta"],
    )
    num_tokens = q.shape[1]
    aqk = torch.empty(
        1,
        num_tokens,
        num_heads,
        CHUNK_SIZE,
        device="cuda",
        dtype=q.dtype,
    )
    akk = torch.empty(
        1,
        num_tokens,
        num_heads,
        SUB_CHUNK_SIZE,
        device="cuda",
        dtype=torch.float32,
    )

    def grid(meta):
        return (
            num_tokens,
            triton.cdiv(num_heads, meta["BH"]),
        )

    def launch(config):
        kernel = chunk_kda_fwd_kernel_intra_token_parallel[grid](
            q=q,
            k=k,
            g=g,
            beta=beta,
            Aqk=aqk,
            Akk=akk,
            scale=HEAD_DIM**-0.5,
            cu_seqlens=tensors["cu_seqlens"],
            N=1,
            T=num_tokens,
            H=num_heads,
            K=HEAD_DIM,
            BT=CHUNK_SIZE,
            BC=SUB_CHUNK_SIZE,
            BK=HEAD_DIM,
            BH=config["meta"]["BH"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
        return aqk, kernel

    candidates = [
        _config({"BH": INTRA_BLOCK_H}, num_warps=num_warps)
        for num_warps in (1, 2, 4, 8)
    ]
    _tune_and_dump(
        output_dir,
        (
            "chunk_kda_fwd_kernel_intra_token_parallel"
            f"_H_{num_heads}_BH_{INTRA_BLOCK_H}"
        ),
        candidates,
        launch,
        ["q", "k", "g", "beta", "Aqk", "Akk", "scale", "cu_seqlens", "N", "T"],
    )


def _generate_inter_solve(
    output_dir: str,
    num_heads: int,
    tensors: dict,
    fuse_recompute: bool,
    fuse_diagonal: bool,
) -> None:
    q, k, v, g, beta = (
        tensors["q"],
        tensors["k"],
        tensors["v"],
        tensors["g"],
        tensors["beta"],
    )
    num_tokens = q.shape[1]
    aqk = torch.zeros(
        1,
        num_tokens,
        num_heads,
        CHUNK_SIZE,
        device="cuda",
        dtype=q.dtype,
    )
    akk_diag = torch.empty(
        1,
        num_tokens,
        num_heads,
        SUB_CHUNK_SIZE,
        device="cuda",
        dtype=torch.float32,
    )
    akk = torch.zeros_like(aqk)
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k)
    grid = (tensors["chunk_indices"].shape[0], num_heads)

    def launch(config):
        kernel = chunk_kda_fwd_kernel_inter_solve_fused[grid](
            q=q,
            k=k,
            g=g,
            beta=beta,
            Aqk=aqk,
            Akkd=akk_diag,
            Akk=akk,
            scale=HEAD_DIM**-0.5,
            v_in=v,
            w_out=w,
            u_out=u,
            kg_out=kg,
            cu_seqlens=tensors["cu_seqlens"],
            chunk_indices=tensors["chunk_indices"],
            T=num_tokens,
            H=num_heads,
            K=HEAD_DIM,
            V=HEAD_DIM if fuse_recompute else 0,
            BT=CHUNK_SIZE,
            BC=SUB_CHUNK_SIZE,
            BK=config["meta"]["BK"],
            BV=config["meta"]["BV"],
            USE_SAFE_GATE=False,
            FUSE_RECOMPUTE=fuse_recompute,
            FUSE_DIAGONAL=fuse_diagonal,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
        return aqk, kernel

    candidates = [
        _config(
            {"BK": INTER_BLOCK_K, "BV": INTER_BLOCK_V},
            num_warps=num_warps,
            num_stages=3,
        )
        for num_warps in (1, 2, 4)
    ]
    _tune_and_dump(
        output_dir,
        (
            "chunk_kda_fwd_kernel_inter_solve_fused"
            f"_H_{num_heads}"
            f"_FR_{int(fuse_recompute)}"
            f"_FD_{int(fuse_diagonal)}"
            f"_BK_{INTER_BLOCK_K}"
            f"_BV_{INTER_BLOCK_V}"
        ),
        candidates,
        launch,
        [
            "q",
            "k",
            "g",
            "beta",
            "Aqk",
            "Akkd",
            "Akk",
            "scale",
            "v_in",
            "w_out",
            "u_out",
            "kg_out",
            "cu_seqlens",
            "chunk_indices",
            "T",
        ],
    )


def _generate_recompute(output_dir: str, num_heads: int, tensors: dict) -> None:
    k, v, g, beta = (
        tensors["k"],
        tensors["v"],
        tensors["g"],
        tensors["beta"],
    )
    num_tokens = k.shape[1]
    akk = torch.randn(
        1,
        num_tokens,
        num_heads,
        CHUNK_SIZE,
        device="cuda",
        dtype=k.dtype,
    )
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k)
    grid = (tensors["chunk_indices"].shape[0], num_heads)

    def launch(config):
        kernel = recompute_w_u_fwd_kernel[grid](
            k=k,
            kg=kg,
            v=v,
            beta=beta,
            w=w,
            u=u,
            A=akk,
            gk=g,
            cu_seqlens=tensors["cu_seqlens"],
            chunk_indices=tensors["chunk_indices"],
            T=num_tokens,
            H=num_heads,
            K=HEAD_DIM,
            V=HEAD_DIM,
            BT=CHUNK_SIZE,
            BK=config["meta"]["BK"],
            BV=config["meta"]["BV"],
            STORE_KG=True,
            IS_VARLEN=True,
            DOT_PRECISION="tf32",
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
        return w, kernel

    candidates = [
        _config(
            {"BK": RECOMPUTE_BLOCK_K, "BV": RECOMPUTE_BLOCK_V},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for num_warps in (2, 4, 8)
        for num_stages in (2, 3, 4)
    ]
    _tune_and_dump(
        output_dir,
        (
            "recompute_w_u_fwd_kernel"
            f"_H_{num_heads}"
            f"_BK_{RECOMPUTE_BLOCK_K}"
            f"_BV_{RECOMPUTE_BLOCK_V}"
        ),
        candidates,
        launch,
        ["k", "kg", "v", "beta", "w", "u", "A", "gk", "cu_seqlens", "chunk_indices", "T"],
    )


def _generate_chunk_state(
    output_dir: str,
    num_heads: int,
    tensors: dict,
    nt_bucket: int,
    ssm_dtype_name: str,
) -> None:
    k = tensors["k"]
    u = tensors["v"]
    g = tensors["g"]
    num_tokens = k.shape[1]
    w = torch.randn_like(k)
    h = torch.empty(
        1,
        tensors["chunk_indices"].shape[0],
        num_heads,
        HEAD_DIM,
        HEAD_DIM,
        device="cuda",
        dtype=k.dtype,
    )
    v_new = torch.empty_like(u)
    # Only the persistent state pool follows the SSM dtype; the per-chunk ``h``
    # scratch above stays in the activation dtype.
    state = torch.randn(
        1,
        num_heads,
        HEAD_DIM,
        HEAD_DIM,
        device="cuda",
        dtype=SSM_DTYPES[ssm_dtype_name],
    )
    initial_state_backup = state.clone()

    def grid(meta):
        return (
            triton.cdiv(HEAD_DIM, meta["BV"]),
            num_heads,
        )

    def launch(config):
        kernel = chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
            k=k,
            v=u,
            w=w,
            v_new=v_new,
            g=None,
            gk=g,
            h=h,
            initial_state=state,
            initial_state_indices=tensors["state_indices"],
            stride_init_state=state.stride(0),
            cu_seqlens=tensors["cu_seqlens"],
            chunk_offsets=tensors["chunk_offsets"],
            T=num_tokens,
            H=num_heads,
            Hg=num_heads,
            K=HEAD_DIM,
            V=HEAD_DIM,
            BT=CHUNK_SIZE,
            BV=config["meta"]["BV"],
            USE_G=False,
            USE_GK=True,
            USE_INITIAL_STATE=True,
            INPLACE_UPDATE=True,
            SAVE_NEW_VALUE=True,
            IS_VARLEN=True,
            NT_BUCKET=nt_bucket,
            USE_EXP2=True,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
        return h, kernel

    candidates = [
        _config(
            {"BV": STATE_BLOCK_V},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for num_warps in (2, 4)
        for num_stages in (2, 3)
    ]
    _tune_and_dump(
        output_dir,
        (
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
            f"_H_{num_heads}_NTB_{nt_bucket}"
            f"_BV_{STATE_BLOCK_V}"
            f"_SSM_{SSM_DTYPE_TOKENS[ssm_dtype_name]}"
        ),
        candidates,
        launch,
        [
            "k",
            "v",
            "w",
            "v_new",
            "gk",
            "h",
            "initial_state",
            "initial_state_indices",
            "stride_init_state",
            "cu_seqlens",
            "chunk_offsets",
            "T",
        ],
        reset=lambda: state.copy_(initial_state_backup),
    )


def _generate_chunk_output(
    output_dir: str, num_heads: int, tensors: dict
) -> None:
    q, v, g = tensors["q"], tensors["v"], tensors["g"]
    num_tokens = q.shape[1]
    num_chunks = tensors["chunk_indices"].shape[0]
    h = torch.randn(
        1,
        num_chunks,
        num_heads,
        HEAD_DIM,
        HEAD_DIM,
        device="cuda",
        dtype=q.dtype,
    )
    aqk = torch.randn(
        1,
        num_tokens,
        num_heads,
        CHUNK_SIZE,
        device="cuda",
        dtype=q.dtype,
    )
    out = torch.empty_like(v)

    def grid(meta):
        return (
            triton.cdiv(HEAD_DIM, meta["BV"]),
            num_chunks,
            num_heads,
        )

    def launch(config):
        kernel = chunk_gla_fwd_kernel_o[grid](
            q=q,
            v=v,
            g=g,
            h=h,
            o=out,
            A=aqk,
            cu_seqlens=tensors["cu_seqlens"],
            chunk_indices=tensors["chunk_indices"],
            scale=HEAD_DIM**-0.5,
            T=num_tokens,
            H=num_heads,
            K=HEAD_DIM,
            V=HEAD_DIM,
            BT=CHUNK_SIZE,
            BK=config["meta"]["BK"],
            BV=config["meta"]["BV"],
            IS_VARLEN=True,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
        return out, kernel

    candidates = [
        _config(
            {"BK": OUTPUT_BLOCK_K, "BV": OUTPUT_BLOCK_V},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for num_warps in (2, 4, 8)
        for num_stages in (2, 3, 4)
    ]
    _tune_and_dump(
        output_dir,
        (
            "chunk_gla_fwd_kernel_o"
            f"_H_{num_heads}"
            f"_BK_{OUTPUT_BLOCK_K}"
            f"_BV_{OUTPUT_BLOCK_V}"
        ),
        candidates,
        launch,
        ["q", "v", "g", "h", "o", "A", "cu_seqlens", "chunk_indices", "scale", "T"],
    )


def _generate_packed_decode(
    output_dir: str,
    num_heads: int,
    lower_bound: Optional[float],
    ssm_dtype_name: str,
) -> None:
    batch = 4
    dtype = torch.bfloat16
    device = "cuda"
    qkv_dim = 3 * num_heads * HEAD_DIM
    mixed_qkv = torch.randn(batch, qkv_dim, device=device, dtype=dtype)
    a = torch.randn(batch, num_heads * HEAD_DIM, device=device, dtype=dtype)
    b = torch.randn(batch, num_heads, device=device, dtype=dtype)
    a_log = torch.randn(num_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(
        num_heads * HEAD_DIM, device=device, dtype=torch.float32
    )
    out = torch.empty(batch, 1, num_heads, HEAD_DIM, device=device, dtype=dtype)
    state = torch.randn(
        batch,
        num_heads,
        HEAD_DIM,
        HEAD_DIM,
        device=device,
        dtype=SSM_DTYPES[ssm_dtype_name],
    )
    indices = torch.arange(batch, device=device, dtype=torch.int32)
    def launch(config):
        grid = (
            triton.cdiv(HEAD_DIM, config["meta"]["BV"]),
            batch * num_heads,
        )
        kernel = fused_recurrent_kda_packed_decode_kernel[grid](
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=a_log,
            dt_bias=dt_bias,
            o=out,
            h0=state,
            ht=state,
            ssm_state_indices=indices,
            scale=HEAD_DIM**-0.5,
            lower_bound=0.0 if lower_bound is None else lower_bound,
            stride_mixed_qkv_tok=mixed_qkv.stride(0),
            stride_a_tok=a.stride(0),
            stride_b_tok=b.stride(0),
            stride_init_state_token=state.stride(0),
            stride_final_state_token=state.stride(0),
            stride_indices_seq=indices.stride(0),
            H=num_heads,
            HV=num_heads,
            K=HEAD_DIM,
            V=HEAD_DIM,
            BK=config["meta"]["BK"],
            BV=config["meta"]["BV"],
            SOFTPLUS_THRESHOLD=SOFTPLUS_THRESHOLD,
            USE_QK_L2NORM_IN_KERNEL=True,
            USE_LOWER_BOUND=lower_bound is not None,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
        return out, kernel

    _tune_and_dump(
        output_dir,
        (
            "fused_recurrent_kda_packed_decode_kernel"
            f"_H_{num_heads}"
            f"_BK_{RECURRENT_BLOCK_K}"
            f"_BV_{RECURRENT_BLOCK_V}"
            f"_LB_{int(lower_bound is not None)}"
            f"_SSM_{SSM_DTYPE_TOKENS[ssm_dtype_name]}"
        ),
        [
            _config(
                {"BK": RECURRENT_BLOCK_K, "BV": RECURRENT_BLOCK_V},
                num_warps=1,
                num_stages=3,
            )
        ],
        launch,
        [
            "mixed_qkv",
            "a",
            "b",
            "A_log",
            "dt_bias",
            "o",
            "h0",
            "ht",
            "ssm_state_indices",
            "scale",
            "lower_bound",
            "stride_mixed_qkv_tok",
            "stride_a_tok",
            "stride_b_tok",
            "stride_init_state_token",
            "stride_final_state_token",
            "stride_indices_seq",
        ],
    )


def _generate_sigmoid_recurrent(
    output_dir: str,
    num_heads: int,
    mode: str,
    np2_t: int,
    lower_bound: Optional[float],
    ssm_dtype_name: str,
) -> None:
    is_verify = mode != "decode"
    has_tree = mode == "verify_tree"
    batch = 1 if is_verify else 2
    tokens_per_sequence = min(np2_t, 3) if is_verify else 1
    total_tokens = batch * tokens_per_sequence
    dtype = torch.bfloat16
    device = "cuda"

    q = torch.randn(1, total_tokens, num_heads, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    a = torch.randn(
        total_tokens, num_heads * HEAD_DIM, device=device, dtype=dtype
    )
    b = torch.randn(1, total_tokens, num_heads, device=device, dtype=dtype)
    out = torch.empty(1, 1, total_tokens, num_heads, HEAD_DIM, device=device, dtype=dtype)
    a_log = torch.randn(num_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(
        num_heads * HEAD_DIM, device=device, dtype=torch.float32
    )
    ssm_dtype = SSM_DTYPES[ssm_dtype_name]
    state = torch.randn(
        batch,
        num_heads,
        HEAD_DIM,
        HEAD_DIM,
        device=device,
        dtype=ssm_dtype,
    )
    state_indices = torch.arange(batch, device=device, dtype=torch.int32)
    cu_seqlens = torch.arange(
        0,
        total_tokens + 1,
        tokens_per_sequence,
        device=device,
        dtype=torch.int64,
    )

    if is_verify:
        # The per-step verify checkpoints hold recurrent states, so they follow
        # the SSM dtype instead of the activation dtype.
        intermediate = torch.empty(
            batch,
            tokens_per_sequence,
            num_heads,
            HEAD_DIM,
            HEAD_DIM,
            device=device,
            dtype=ssm_dtype,
        )
        intermediate_indices = torch.arange(
            batch, device=device, dtype=torch.int32
        )
        if has_tree:
            parents = torch.zeros(
                batch, tokens_per_sequence, device=device, dtype=torch.int32
            )
            parent_stride_seq, parent_stride_token = parents.stride()
        else:
            parents = None
            parent_stride_seq = parent_stride_token = 0
    else:
        intermediate = None
        intermediate_indices = None
        parents = None
        parent_stride_seq = parent_stride_token = 0

    def launch(config):
        grid = (
            triton.cdiv(HEAD_DIM, config["meta"]["BK"]),
            triton.cdiv(HEAD_DIM, config["meta"]["BV"]),
            batch * num_heads,
        )
        kernel = fused_sigmoid_gating_delta_rule_update_kernel[grid](
            A_log=a_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=SOFTPLUS_BETA,
            softplus_threshold=SOFTPLUS_THRESHOLD,
            lower_bound=0.0 if lower_bound is None else lower_bound,
            q=q,
            k=k,
            v=v,
            b=b,
            o=out,
            h0_source=state,
            h0_indices=state_indices,
            cu_seqlens=cu_seqlens,
            intermediate_states_buffer=intermediate,
            intermediate_state_indices=intermediate_indices,
            cache_steps=tokens_per_sequence if is_verify else 0,
            retrieve_parent_token_ptr=parents,
            stride_retrieve_parent_token_seq=parent_stride_seq,
            stride_retrieve_parent_token_token=parent_stride_token,
            scale=HEAD_DIM**-0.5,
            T=total_tokens,
            stride_a=a.stride(0),
            stride_q=q.stride(1),
            stride_k=k.stride(1),
            stride_v=v.stride(1),
            stride_b=b.stride(-2),
            NP2_T=np2_t,
            B=1,
            H=num_heads,
            HV=num_heads,
            K=HEAD_DIM,
            V=HEAD_DIM,
            BK=config["meta"]["BK"],
            BV=config["meta"]["BV"],
            USE_INITIAL_STATE=True,
            USE_QK_L2NORM_IN_KERNEL=True,
            IS_VARLEN=True,
            IS_KDA=True,
            USE_LOWER_BOUND=lower_bound is not None,
            DISABLE_STATE_UPDATE=is_verify,
            CACHE_INTERMEDIATE_STATES=is_verify,
            HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=has_tree,
            replayssm_rawv=None,
            replayssm_rawk=None,
            replayssm_g=None,
            replayssm_beta=None,
            stride_rawv_slot=0,
            stride_rawk_slot=0,
            stride_g_slot=0,
            stride_beta_slot=0,
            MAX_CACHE_LEN=0,
            CACHE_RING=False,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
        return out, kernel

    expected_arg_names = [
        "A_log",
        "a",
        "dt_bias",
        "softplus_beta",
        "softplus_threshold",
        "lower_bound",
        "q",
        "k",
        "v",
        "b",
        "o",
        "h0_source",
        "h0_indices",
        "cu_seqlens",
    ]
    if is_verify:
        expected_arg_names.extend(
            ["intermediate_states_buffer", "intermediate_state_indices"]
        )
    expected_arg_names.append("cache_steps")
    if has_tree:
        expected_arg_names.append("retrieve_parent_token_ptr")
    expected_arg_names.extend(
        [
            "stride_retrieve_parent_token_seq",
            "stride_retrieve_parent_token_token",
            "scale",
            "T",
            "stride_a",
            "stride_q",
            "stride_k",
            "stride_v",
            "stride_b",
        ]
    )

    _tune_and_dump(
        output_dir,
        (
            "fused_sigmoid_gating_delta_rule_update_kernel"
            f"_H_{num_heads}_MODE_{mode}_NP2_{np2_t}"
            f"_BK_{RECURRENT_BLOCK_K}_BV_{RECURRENT_BLOCK_V}"
            f"_LB_{int(lower_bound is not None)}"
            f"_SSM_{SSM_DTYPE_TOKENS[ssm_dtype_name]}"
        ),
        [
            _config(
                {"BK": RECURRENT_BLOCK_K, "BV": RECURRENT_BLOCK_V},
                num_warps=1,
                num_stages=3,
            )
        ],
        launch,
        expected_arg_names,
    )


def _powers_of_two(limit: int) -> list[int]:
    values = []
    value = 1
    while value <= limit:
        values.append(value)
        value *= 2
    return values


def _generate_causal_conv_fwd(output_dir: str, num_heads: int) -> None:
    """Generate packed-varlen prefill conv for one TP-local head count."""
    num_sequences = 2
    num_tokens = 128
    max_seq_len = 80
    num_cache_lines = 4
    channels = num_heads * HEAD_DIM
    mixed_channels = 3 * channels
    dtype = torch.bfloat16
    device = "cuda"

    # Match KDA's non-contiguous Q/K/V split after ``mixed_qkv.T``.
    mixed_qkv = torch.randn(
        num_tokens, mixed_channels, device=device, dtype=dtype
    )
    x = mixed_qkv[:, :channels].transpose(0, 1)
    weight = torch.randn(
        channels, CAUSAL_CONV_WIDTH, device=device, dtype=torch.float32
    )
    # Kimi-K3 has no learned conv bias. HAS_BIAS=False dead-strips bias_ptr
    # from the cubin ABI; the C++ launcher must omit it.
    state_storage = torch.randn(
        num_cache_lines,
        CAUSAL_CONV_STATE_LEN,
        mixed_channels,
        device=device,
        dtype=dtype,
    )
    state = state_storage.transpose(1, 2)[:, :channels, :]
    cache_indices = torch.tensor([0, 2], device=device, dtype=torch.int32)
    has_initial_state = torch.tensor(
        [False, True], device=device, dtype=torch.bool
    )
    query_start_loc = torch.tensor(
        [0, max_seq_len, num_tokens], device=device, dtype=torch.int64
    )
    out = torch.empty(num_tokens, channels, device=device, dtype=dtype)

    def launch(config):
        grid = (
            num_sequences,
            triton.cdiv(max_seq_len, CAUSAL_CONV_BLOCK_M),
            triton.cdiv(channels, CAUSAL_CONV_BLOCK_N),
        )
        kernel = causal_conv1d_fwd_kernel[grid](
            x_ptr=x,
            w_ptr=weight,
            bias_ptr=None,
            initial_states_ptr=state,
            cache_indices_ptr=cache_indices,
            has_initial_states_ptr=has_initial_state,
            query_start_loc_ptr=query_start_loc,
            o_ptr=out,
            dim=channels,
            seqlen=0,
            num_cache_lines=num_cache_lines,
            stride_x_seq=0,
            stride_x_dim=x.stride(0),
            stride_x_token=x.stride(1),
            stride_w_dim=weight.stride(0),
            stride_w_width=weight.stride(1),
            stride_istate_seq=state.stride(0),
            stride_istate_dim=state.stride(1),
            stride_istate_token=state.stride(2),
            stride_o_seq=0,
            stride_o_dim=out.stride(1),
            stride_o_token=out.stride(0),
            pad_slot_id=CAUSAL_CONV_PAD_SLOT_ID,
            HAS_BIAS=False,
            KERNEL_WIDTH=CAUSAL_CONV_WIDTH,
            SILU_ACTIVATION=True,
            HAS_INITIAL_STATES=True,
            HAS_CACHE=True,
            IS_CONTINUOUS_BATCHING=True,
            USE_PAD_SLOT=True,
            NP2_STATELEN=CAUSAL_CONV_NP2_STATELEN,
            BLOCK_M=CAUSAL_CONV_BLOCK_M,
            BLOCK_N=CAUSAL_CONV_BLOCK_N,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
        return out, kernel

    _tune_and_dump(
        output_dir,
        (
            "causal_conv1d_fwd_kernel"
            f"_D_{channels}_W_{CAUSAL_CONV_WIDTH}"
        ),
        [_config(num_warps=4, num_stages=2)],
        launch,
        [
            "x_ptr",
            "w_ptr",
            "initial_states_ptr",
            "cache_indices_ptr",
            "has_initial_states_ptr",
            "query_start_loc_ptr",
            "o_ptr",
            "seqlen",
            "num_cache_lines",
            "stride_x_seq",
            "stride_x_dim",
            "stride_x_token",
            "stride_w_dim",
            "stride_w_width",
            "stride_istate_seq",
            "stride_istate_dim",
            "stride_istate_token",
            "stride_o_seq",
            "stride_o_dim",
            "stride_o_token",
        ],
    )


def _generate_causal_conv_update(
    output_dir: str,
    num_heads: int,
    mode_name: str,
    np2_seqlen: int,
) -> None:
    """Generate decode/verify conv; verify length remains a runtime scalar."""
    is_verify = mode_name != "decode"
    has_tree = mode_name == "verify_tree"
    batch = 2
    seqlen = min(np2_seqlen, 3) if is_verify else 1
    num_cache_lines = 4
    channels = 3 * num_heads * HEAD_DIM
    dtype = torch.bfloat16
    device = "cuda"

    mixed_qkv = torch.randn(batch, seqlen, channels, device=device, dtype=dtype)
    x = mixed_qkv.transpose(1, 2)
    out_storage = torch.empty_like(mixed_qkv)
    out = out_storage.transpose(1, 2)
    weight = torch.randn(
        channels, CAUSAL_CONV_WIDTH, device=device, dtype=torch.float32
    )
    # Kimi-K3 has no learned conv bias. HAS_BIAS=False dead-strips bias_ptr
    # from the cubin ABI; the C++ launcher must omit it.
    state_storage = torch.randn(
        num_cache_lines,
        CAUSAL_CONV_STATE_LEN,
        channels,
        device=device,
        dtype=dtype,
    )
    state = state_storage.transpose(1, 2)
    state_indices = torch.tensor([0, 2], device=device, dtype=torch.int32)

    # cache_seqlens stays in the source signature. The compiled ABI drops
    # num_accept_tokens because IS_SPEC_DECODING=False dead-strips the load.
    cache_seqlens = torch.zeros(batch, device=device, dtype=torch.int32)

    if is_verify:
        intermediate_storage = torch.empty(
            batch,
            seqlen,
            CAUSAL_CONV_STATE_LEN,
            channels,
            device=device,
            dtype=dtype,
        )
        intermediate = intermediate_storage.transpose(2, 3)
        intermediate_indices = torch.arange(
            batch, device=device, dtype=torch.int32
        )
    else:
        intermediate = x
        intermediate_indices = state_indices

    if has_tree:
        next_tokens = torch.full(
            (batch, seqlen), -1, device=device, dtype=torch.int32
        )
        next_siblings = torch.full_like(next_tokens, -1)
        if seqlen > 1:
            next_tokens[:, 0] = 1
        if seqlen > 2:
            next_siblings[:, 1] = 2
        parent_tokens = torch.empty_like(next_tokens)
    else:
        next_tokens = state_indices
        next_siblings = state_indices
        parent_tokens = state_indices

    def launch(config):
        grid = (batch, triton.cdiv(channels, CAUSAL_CONV_BLOCK_N))
        kernel = causal_conv1d_update_kernel[grid](
            x_ptr=x,
            w_ptr=weight,
            bias_ptr=None,
            conv_state_ptr=state,
            cache_seqlens_ptr=cache_seqlens,
            conv_state_indices_ptr=state_indices,
            num_accept_tokens_ptr=None,
            intermediate_conv_window_ptr=intermediate,
            intermediate_state_indices_ptr=intermediate_indices,
            retrieve_next_token_ptr=next_tokens,
            retrieve_next_sibling_ptr=next_siblings,
            retrieve_parent_token_ptr=parent_tokens,
            o_ptr=out,
            batch=batch,
            dim=channels,
            seqlen=seqlen,
            state_len=CAUSAL_CONV_STATE_LEN,
            num_cache_lines=num_cache_lines,
            stride_x_seq=x.stride(0),
            stride_x_dim=x.stride(1),
            stride_x_token=x.stride(2),
            stride_w_dim=weight.stride(0),
            stride_w_width=weight.stride(1),
            stride_conv_state_seq=state.stride(0),
            stride_conv_state_dim=state.stride(1),
            stride_conv_state_tok=state.stride(2),
            stride_state_indices=state_indices.stride(0),
            stride_inter_seq=intermediate.stride(0) if is_verify else 0,
            stride_inter_step=intermediate.stride(1) if is_verify else 0,
            stride_inter_dim=intermediate.stride(2) if is_verify else 0,
            stride_inter_win=intermediate.stride(3) if is_verify else 0,
            stride_intermediate_state_indices=(
                intermediate_indices.stride(0) if is_verify else 0
            ),
            stride_retrieve_next_token_seq=(
                next_tokens.stride(0) if has_tree else 0
            ),
            stride_retrieve_next_token_token=(
                next_tokens.stride(1) if has_tree else 0
            ),
            stride_retrieve_next_sibling_seq=(
                next_siblings.stride(0) if has_tree else 0
            ),
            stride_retrieve_next_sibling_token=(
                next_siblings.stride(1) if has_tree else 0
            ),
            stride_retrieve_parent_token_seq=(
                parent_tokens.stride(0) if has_tree else 0
            ),
            stride_retrieve_parent_token_token=(
                parent_tokens.stride(1) if has_tree else 0
            ),
            stride_o_seq=out.stride(0),
            stride_o_dim=out.stride(1),
            stride_o_token=out.stride(2),
            pad_slot_id=CAUSAL_CONV_PAD_SLOT_ID,
            HAS_BIAS=False,
            KERNEL_WIDTH=CAUSAL_CONV_WIDTH,
            SILU_ACTIVATION=True,
            IS_CONTINUOUS_BATCHING=True,
            IS_SPEC_DECODING=False,
            NP2_STATELEN=CAUSAL_CONV_NP2_STATELEN,
            NP2_SEQLEN=np2_seqlen,
            USE_PAD_SLOT=True,
            BLOCK_N=CAUSAL_CONV_BLOCK_N,
            SAVE_INTERMEDIATE=is_verify,
            HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=has_tree,
            USE_GDC=False,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
        return out, kernel

    expected_arg_names = [
        "x_ptr",
        "w_ptr",
        "conv_state_ptr",
        "cache_seqlens_ptr",
        "conv_state_indices_ptr",
        "intermediate_conv_window_ptr",
        "intermediate_state_indices_ptr",
        "retrieve_next_token_ptr",
        "retrieve_next_sibling_ptr",
        "retrieve_parent_token_ptr",
        "o_ptr",
        "batch",
        "seqlen",
        "num_cache_lines",
        "stride_x_seq",
        "stride_x_dim",
        "stride_x_token",
        "stride_w_dim",
        "stride_w_width",
        "stride_conv_state_seq",
        "stride_conv_state_dim",
        "stride_conv_state_tok",
        "stride_state_indices",
        "stride_inter_seq",
        "stride_inter_step",
        "stride_inter_dim",
        "stride_inter_win",
        "stride_intermediate_state_indices",
        "stride_retrieve_next_token_seq",
        "stride_retrieve_next_token_token",
        "stride_retrieve_next_sibling_seq",
        "stride_retrieve_next_sibling_token",
        "stride_retrieve_parent_token_seq",
        "stride_retrieve_parent_token_token",
        "stride_o_seq",
        "stride_o_dim",
        "stride_o_token",
    ]

    _tune_and_dump(
        output_dir,
        (
            "causal_conv1d_update_kernel"
            f"_D_{channels}_W_{CAUSAL_CONV_WIDTH}"
            f"_MODE_{mode_name}_NP2_{np2_seqlen}"
        ),
        [_config(num_warps=4, num_stages=2)],
        launch,
        expected_arg_names,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate KDA Triton kernel artifacts"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        nargs="+",
        required=True,
        help="Local KDA head counts to generate",
    )
    parser.add_argument(
        "--max-verify-tokens",
        type=int,
        default=128,
        help="Maximum target-verify NP2_T specialization",
    )
    parser.add_argument(
        "--ssm-dtype",
        type=str,
        nargs="+",
        choices=sorted(SSM_DTYPES),
        default=["float32", "bfloat16"],
        help=(
            "Recurrent-state dtypes to specialize the state kernels for. "
            "Generating both lets the state pool switch dtype without a "
            "rebuild; pass a single value to shorten the build."
        ),
    )
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    ssm_dtype_names = sorted(set(args.ssm_dtype))

    os.makedirs(args.output_dir, exist_ok=True)
    _generate_l2norm(args.output_dir)
    _generate_gated_rmsnorm(args.output_dir)

    verify_sizes = _powers_of_two(args.max_verify_tokens)
    for num_heads in sorted(set(args.num_heads)):
        _generate_causal_conv_fwd(args.output_dir, num_heads)
        _generate_causal_conv_update(
            args.output_dir, num_heads, mode_name="decode", np2_seqlen=1
        )
        _generate_causal_conv_update(
            args.output_dir,
            num_heads,
            mode_name="verify_chain",
            np2_seqlen=1,
        )
        for np2_seqlen in verify_sizes:
            _generate_causal_conv_update(
                args.output_dir,
                num_heads,
                mode_name="verify_tree",
                np2_seqlen=np2_seqlen,
            )
        tensors = _make_prefill_tensors(num_heads)
        for lower_bound in GATE_LOWER_BOUNDS:
            _generate_gate_cumsum(
                args.output_dir, num_heads, tensors, lower_bound
            )
        _generate_intra_token_parallel(args.output_dir, num_heads, tensors)
        _generate_inter_solve(
            args.output_dir,
            num_heads,
            tensors,
            fuse_recompute=True,
            fuse_diagonal=True,
        )
        _generate_inter_solve(
            args.output_dir,
            num_heads,
            tensors,
            fuse_recompute=False,
            fuse_diagonal=False,
        )
        _generate_recompute(args.output_dir, num_heads, tensors)
        for ssm_dtype_name in ssm_dtype_names:
            for nt_bucket in (0, 1, 2):
                _generate_chunk_state(
                    args.output_dir, num_heads, tensors, nt_bucket, ssm_dtype_name
                )
        _generate_chunk_output(args.output_dir, num_heads, tensors)
        for ssm_dtype_name in ssm_dtype_names:
            for lower_bound in GATE_LOWER_BOUNDS:
                _generate_packed_decode(
                    args.output_dir, num_heads, lower_bound, ssm_dtype_name
                )
                _generate_sigmoid_recurrent(
                    args.output_dir,
                    num_heads,
                    mode="decode",
                    np2_t=1,
                    lower_bound=lower_bound,
                    ssm_dtype_name=ssm_dtype_name,
                )
                _generate_sigmoid_recurrent(
                    args.output_dir,
                    num_heads,
                    mode="verify_chain",
                    np2_t=1,
                    lower_bound=lower_bound,
                    ssm_dtype_name=ssm_dtype_name,
                )
                for np2_t in verify_sizes:
                    _generate_sigmoid_recurrent(
                        args.output_dir,
                        num_heads,
                        mode="verify_tree",
                        np2_t=np2_t,
                        lower_bound=lower_bound,
                        ssm_dtype_name=ssm_dtype_name,
                    )


if __name__ == "__main__":
    main()
