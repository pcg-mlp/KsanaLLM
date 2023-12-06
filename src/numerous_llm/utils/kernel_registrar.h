/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#define CONCAT_IMPL(x, y) x##y
#define CONCAT(x, y) CONCAT_IMPL(x, y)

#define REGISTER_NVIDIA_KERNEL(name, func) static auto CONCAT(kernel_nvidia_, name) = func
#define EXECUTE_NVIDIA_KERNEL(name, ...) CONCAT(kernel_nvidia_, name)(__VA_ARGS__)

#define REGISTER_ASCEND_KERNEL(name, func) static auto CONCAT(kernel_ascend_, name)(&func)
#define EXECUTE_ASCEND_KERNEL(name, ...) CONCAT(kernel_ascend_, name)(__VA_ARGS__)
