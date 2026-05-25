/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <type_traits>

#include "device/device_utils.h"
#include "utils/logger.h"

// Upload input's persistent host vector to a shared device tensor view, updating offset.
// Allocates a sub-view from `name##_tensor` and advances its shape[0].
// Requires `context_` (std::shared_ptr<Context>) and `name##_tensor` (shared Tensor) in scope.
#define HOST_TO_SHARED_DEVICE(name, input)                                                                    \
  do {                                                                                                        \
    KLLM_LOG_DEBUG << #name "_host: " << (input).name##_host;                                                 \
    (input).name##_tensor = name##_tensor.GetView({(input).name##_host.size()}, name##_tensor.shape[0]);      \
    name##_tensor.shape[0] += (input).name##_tensor.GetElementNumber();                                       \
    MemcpyAsync((input).name##_tensor.GetPtr<void>(), (input).name##_host.data(),                             \
                (input).name##_host.size() * sizeof(std::decay_t<decltype((input).name##_host)>::value_type), \
                MEMCPY_HOST_TO_DEVICE, context_->device->GetH2DStream());                                     \
  } while (0)

// Upload a host vector directly to an owned device tensor (no shared offset tracking).
// Sets tensor shape to list size, then issues async H2D memcpy.
// Requires `context_` (std::shared_ptr<Context>) in scope.
#define HOST_TO_DEVICE(list, tensor)                                                                     \
  do {                                                                                                   \
    KLLM_LOG_DEBUG << #list ": " << (list);                                                              \
    (tensor).shape = {(list).size()};                                                                    \
    MemcpyAsync((tensor).GetPtr<void>(), (list).data(),                                                  \
                (list).size() * sizeof(std::decay_t<decltype(list)>::value_type), MEMCPY_HOST_TO_DEVICE, \
                context_->device->GetH2DStream());                                                       \
  } while (0)
