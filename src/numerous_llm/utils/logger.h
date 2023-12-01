/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <iostream>

namespace numerous_llm {

class Logger {};

#define NLLM_LOG_TRACE std::cout
#define NLLM_LOG_DEBUG std::cout
#define NLLM_LOG_INFO std::cout
#define NLLM_LOG_WARNING std::cout
#define NLLM_LOG_ERROR std::cout

} // namespace numerous_llm.
