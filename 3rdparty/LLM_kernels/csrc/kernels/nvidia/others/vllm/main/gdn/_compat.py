# Copyright 2026 Tencent Inc. All rights reserved.
#
# Copied-code copyright (cdiv / next_power_of_2 / platform helpers / constants
# below, vllm v0.21.0, Apache-2.0):
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Compatibility shim for vendored vllm v0.21.0 fla / mamba kernels.
# Replaces the following vllm-internal modules used by gdn/ops/*.py (the tiny
# helpers marked below are copied verbatim; the rest is interface-matching
# original code):
#   - vllm.triton_utils
#       https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/triton_utils.py
#   - vllm.utils.math_utils                 -> {cdiv, next_power_of_2}
#       https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/utils/math_utils.py
#   - vllm.utils.platform_utils             -> {num_compute_units}
#       https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/utils/platform_utils.py
#   - vllm.platforms.current_platform       -> CurrentPlatform (CUDA-only stub)
#   - vllm.v1.attention.backends.utils      -> {NULL_BLOCK_ID=0, PAD_SLOT_ID=-1}
#       https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/v1/attention/backends/utils.py
#
# This file lets the vendored ops/ run with only `torch` + `triton` installed,
# without depending on the vllm package.

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice  # noqa: F401  (re-exported)

import torch
from packaging.version import Version

# Pin Triton 3.2: its value-1 arg specialization strips signature slots the
# C++ side counts on; later versions changed that, silently shifting the ABI.
_triton_version = Version(triton.__version__)
assert _triton_version.release[:2] == (3, 2), \
    f"GDN cubins are generated against the Triton 3.2 ABI, got {triton.__version__}"


# ---------------------------------------------------------------------------
# Re-exports matching vllm.triton_utils
# ---------------------------------------------------------------------------
LOG2E = 1.4426950408889634
LOGE2 = 0.6931471805599453


# ---------------------------------------------------------------------------
# math_utils — copied from
# https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/utils/math_utils.py
# ---------------------------------------------------------------------------
def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def next_power_of_2(n: int) -> int:
    """The next power of 2 (inclusive)."""
    return 1 if n < 1 else 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# platform_utils — copied from
# https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/utils/platform_utils.py
# ---------------------------------------------------------------------------
def num_compute_units(device_id: int = 0) -> int:
    """Number of streaming multiprocessors / compute units of the device."""
    if not torch.cuda.is_available():
        return 1
    return torch.cuda.get_device_properties(device_id).multi_processor_count


# ---------------------------------------------------------------------------
# current_platform (CUDA-only stub matching vllm interface used by fla/mamba)
# ---------------------------------------------------------------------------
class _CurrentPlatform:
    """Minimal subset of vllm.platforms.current_platform used by the
    vendored kernels.  KsanaLLM is CUDA-only, so we hard-code that.
    """

    @staticmethod
    def is_cuda_alike() -> bool:
        return True

    @staticmethod
    def is_cuda() -> bool:
        return True

    @staticmethod
    def is_rocm() -> bool:
        return False

    @staticmethod
    def num_compute_units(device_id: int = 0) -> int:
        return num_compute_units(device_id)


current_platform = _CurrentPlatform()


# ---------------------------------------------------------------------------
# Constants copied from vllm.v1.attention.backends.utils
# https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/v1/attention/backends/utils.py
# ---------------------------------------------------------------------------
NULL_BLOCK_ID: int = 0
PAD_SLOT_ID: int = -1


__all__ = [
    "triton",
    "tl",
    "tldevice",
    "LOG2E",
    "LOGE2",
    "cdiv",
    "next_power_of_2",
    "num_compute_units",
    "current_platform",
    "NULL_BLOCK_ID",
    "PAD_SLOT_ID",
]
