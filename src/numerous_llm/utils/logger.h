/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

namespace numerous_llm {

class Logger {};

#define NLLM_LOG(level, ...)                                                   \
  tensorrt_llm::common::Logger::getLogger()->log(level, __VA_ARGS__)
#define NLLM_LOG_TRACE(...)                                                    \
  NLLM_LOG(tensorrt_llm::common::Logger::TRACE, __VA_ARGS__)
#define NLLM_LOG_DEBUG(...)                                                    \
  NLLM_LOG(tensorrt_llm::common::Logger::DEBUG, __VA_ARGS__)
#define NLLM_LOG_INFO(...)                                                     \
  NLLM_LOG(tensorrt_llm::common::Logger::INFO, __VA_ARGS__)
#define NLLM_LOG_WARNING(...)                                                  \
  NLLM_LOG(tensorrt_llm::common::Logger::WARNING, __VA_ARGS__)
#define NLLM_LOG_ERROR(...)                                                    \
  NLLM_LOG(tensorrt_llm::common::Logger::ERROR, __VA_ARGS__)

} // namespace numerous_llm.
