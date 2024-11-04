/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "csrc/utils/nvidia/string_utils.h"
#include "csrc/utils/nvidia/kllm_exception.h"

#include <string>

namespace llm_kernels {
namespace utils {
[[noreturn]] inline void throwRuntimeError(char const* const file, int const line, std::string const& info = "")
{
    throw KllmException(file, line, fmtstr("[ERROR] Assertion failed: %s", info.c_str()));
}

}  // namespace utils
}  // namespace llm_kernels


class DebugConfig
{
public:
    static bool isCheckDebugEnabled();
};

#if defined(_WIN32)
#define KLLM_KERNEL_LIKELY(x) (__assume((x) == 1), (x))
#define KLLM_KERNEL_UNLIKELY(x) (__assume((x) == 0), (x))
#else
#define KLLM_KERNEL_LIKELY(x) __builtin_expect((x), 1)
#define KLLM_KERNEL_UNLIKELY(x) __builtin_expect((x), 0)
#endif

#define KLLM_KERNEL_CHECK(val)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        KLLM_KERNEL_LIKELY(static_cast<bool>(val)) ? ((void) 0)                                                               \
                                            : llm_kernels::utils::throwRuntimeError(__FILE__, __LINE__, #val);         \
    } while (0)

#define KLLM_KERNEL_CHECK_WITH_INFO(val, info, ...)                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        KLLM_KERNEL_LIKELY(static_cast<bool>(val))                                                                            \
        ? ((void) 0)                                                                                                   \
        : llm_kernels::utils::throwRuntimeError(                                                                       \
            __FILE__, __LINE__, llm_kernels::utils::fmtstr(info, ##__VA_ARGS__));                                      \
    } while (0)

#define KLLM_KERNEL_CHECK_DEBUG(val)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if (KLLM_KERNEL_UNLIKELY(DebugConfig::isCheckDebugEnabled()))                                                         \
        {                                                                                                              \
            KLLM_KERNEL_LIKELY(static_cast<bool>(val)) ? ((void) 0)                                                           \
                                                : llm_kernels::utils::throwRuntimeError(__FILE__, __LINE__, #val);     \
        }                                                                                                              \
    } while (0)

#define KLLM_KERNEL_CHECK_DEBUG_WITH_INFO(val, info, ...)                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        if (KLLM_KERNEL_UNLIKELY(DebugConfig::isCheckDebugEnabled()))                                                         \
        {                                                                                                              \
            KLLM_KERNEL_LIKELY(static_cast<bool>(val))                                                                        \
            ? ((void) 0)                                                                                               \
            : llm_kernels::utils::throwRuntimeError(                                                                   \
                __FILE__, __LINE__, llm_kernels::utils::fmtstr(info, ##__VA_ARGS__));                                  \
        }                                                                                                              \
    } while (0)

#define KLLM_KERNEL_THROW(...)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        throw NEW_KLLM_KERNEL_EXCEPTION(__VA_ARGS__);                                                                         \
    } while (0)

#define KLLM_KERNEL_WRAP(ex)                                                                                                  \
    NEW_KLLM_KERNEL_EXCEPTION("%s: %s", llm_kernels::utils::KllmException::demangle(typeid(ex).name()).c_str(), ex.what())
