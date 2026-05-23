/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

// Plain C++ POD layout for PrefillPlan.
// This header is host-only and can be included from both .cu and .cpp files.
// The full CUDA device-side kernels live in plan.cuh which includes this file.

#pragma once

#include <cstdint>

namespace llm_kernels {
namespace nvidia {


// Unify length and index-related types to int32_t
using IndicesT = int32_t;
// -----------------------------------------------------------------------------
// PrefillPlan
// -----------------------------------------------------------------------------
// Layout matches SGLang's `device::compress::PrefillPlan` exactly (16 bytes,
// 16-byte aligned), so a host/device-produced tensor of shape
// `[num_tokens, kPrefillPlanDim]` with dtype=uint8 can be reinterpret-cast to
// `PrefillPlan*`.
struct alignas(16) PrefillPlan {
  uint32_t ragged_id;
  uint32_t batch_id;
  uint32_t position;
  uint32_t window_len;  // must be in `[0, compress_ratio * (1 + is_overlap))`
};

using PrefillPlanTensorDtype = uint8_t;
inline constexpr int64_t kPrefillPlanDim = 16;

static_assert(alignof(PrefillPlan) == sizeof(PrefillPlan));
static_assert(sizeof(PrefillPlan) == kPrefillPlanDim * sizeof(PrefillPlanTensorDtype));

}  // namespace nvidia
}  // namespace llm_kernels
