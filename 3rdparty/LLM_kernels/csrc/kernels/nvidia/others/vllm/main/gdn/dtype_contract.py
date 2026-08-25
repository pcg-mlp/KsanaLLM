# Copyright 2026 Tencent Inc. All rights reserved.
#
# GDN kernel dtype contract: params default to model dtype; those in FP32_PARAMS
# are fp32. BUFFER_FP32 must stay in sync with C++ PlanWorkspace() buffer sizes.

import torch

# (kernel_name, param_name) → always fp32 regardless of model dtype.
FP32_PARAMS: set[tuple[str, str]] = {
    ("fused_post_conv",           "g"),
    ("fused_post_conv",           "beta"),
    ("chunk_local_cumsum_scalar", "s"),
    ("chunk_scaled_dot_kkt",      "A"),
    ("chunk_scaled_dot_kkt",      "g"),
    ("chunk_scaled_dot_kkt",      "beta"),
    ("solve_tril_64",             "A"),
    ("wy_fast_recompute",         "beta"),
    ("wy_fast_recompute",         "g"),
    ("chunk_h",                   "h0"),
    ("chunk_h",                   "g"),
    ("chunk_o",                   "g"),
    ("fused_recurrent_fwd",       "g"),
    ("fused_recurrent_fwd",       "beta"),
    ("fused_conv_post_c1",        "g"),
    ("fused_conv_post_c1",        "beta"),
    ("fused_conv_post_c1",        "A_log"),
    ("fused_conv_post_c1",        "dt_bias"),
    ("packed_decode",             "A_log"),
    ("packed_decode",             "dt_bias"),
    ("rmsnorm_gated",             "rstd"),
}

# Workspace buffers that are fp32.
BUFFER_FP32: set[str] = {"g", "beta", "A", "h0", "cumsum", "rstd", "ht"}


def resolve(kernel: str, param: str, model_dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if (kernel, param) in FP32_PARAMS else model_dtype
