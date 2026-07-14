/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================
 *
 * 这个头文件作为DeepEP库头文件的统一入口点，避免DeepEP的通用头文件名
 * (如config.hpp, event.hpp等)与KsanaLLM项目中的其他头文件产生命名冲突。
 * 路径通过CMake的DEEPEP_INCLUDE_DIR配置
 */

#pragma once

#include "api.cuh"
#include "config.hpp"
#include "event.hpp"
#include "kernels/configs.cuh"
#include "kernels/exception.cuh"