/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#pragma once

// Redirect to the canonical sgl_kernel utils header used across the whole repo.
// The original utils_compat.cuh was a standalone copy of sgl_kernel/utils.cuh;
// including both caused duplicate-symbol errors. Using the shared header instead.
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/utils.cuh"
