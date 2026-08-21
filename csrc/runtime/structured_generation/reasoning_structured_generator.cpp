/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/structured_generation/reasoning_structured_generator.h"

#include <algorithm>
#include <stdexcept>

namespace ksana_llm {

ReasoningStructuredGenerator::ReasoningStructuredGenerator(
    std::shared_ptr<StructuredGeneratorInterface> inner_generator,
    int think_end_token_id,
    bool start_in_reasoning)
    : inner_generator_(inner_generator),
      think_end_token_id_(think_end_token_id),
      start_in_reasoning_(start_in_reasoning),
      is_in_reasoning_(start_in_reasoning) {
  if (!inner_generator_) {
    throw std::invalid_argument("Inner generator cannot be null");
  }
}

bool ReasoningStructuredGenerator::AcceptToken(int token_id) {
  if (is_in_reasoning_) {
    if (token_id == think_end_token_id_) {
      is_in_reasoning_ = false;
      reasoning_end_seen_ = true;
      return true;
    }
    ++reasoning_token_count_;
    return true;
  }

  const bool accepted = inner_generator_->AcceptToken(token_id);
  if (accepted) {
    ++generation_token_count_;
  }
  return accepted;
}

bool ReasoningStructuredGenerator::FillNextTokenBitmask(void* next_token_bitmask) {
  // 如果仍在推理阶段，不应用任何约束（返回false表示不需要约束）
  if (is_in_reasoning_) {
    return false;
  }

  // 推理阶段结束后，使用内部生成器的约束
  return inner_generator_->FillNextTokenBitmask(next_token_bitmask);
}

void ReasoningStructuredGenerator::Rollback(int rollback_token_num) {
  if (rollback_token_num < 0) {
    throw std::invalid_argument("Rollback token num must be greater than or equal to 0");
  }

  size_t rollback_num = static_cast<size_t>(rollback_token_num);
  const size_t accepted_token_count = reasoning_token_count_ + generation_token_count_ +
                                      static_cast<size_t>(reasoning_end_seen_);
  if (rollback_num > accepted_token_count) {
    throw std::invalid_argument("Rollback token num exceeds accepted token count");
  }

  const size_t generation_rollback = std::min(rollback_num, generation_token_count_);
  if (generation_rollback > 0) {
    inner_generator_->Rollback(static_cast<int>(generation_rollback));
    generation_token_count_ -= generation_rollback;
    rollback_num -= generation_rollback;
  }

  if (rollback_num > 0 && reasoning_end_seen_) {
    reasoning_end_seen_ = false;
    is_in_reasoning_ = true;
    --rollback_num;
  }

  reasoning_token_count_ -= rollback_num;
}

void ReasoningStructuredGenerator::Reset() {
  inner_generator_->Reset();
  is_in_reasoning_ = start_in_reasoning_;
  reasoning_end_seen_ = false;
  reasoning_token_count_ = 0;
  generation_token_count_ = 0;
}

bool ReasoningStructuredGenerator::FindJumpForwardTokens(std::vector<int>& jump_tokens) {
  // 如果仍在推理阶段，不支持跳跃前进
  if (is_in_reasoning_) {
    jump_tokens.clear();
    return false;
  }

  // 推理阶段结束后，使用内部生成器的跳跃前进功能
  return inner_generator_->FindJumpForwardTokens(jump_tokens);
}

bool ReasoningStructuredGenerator::IsTerminated() const {
  // 如果仍在推理阶段，未终止
  if (is_in_reasoning_) {
    return false;
  }

  // 推理阶段结束后，检查内部生成器的终止状态
  return inner_generator_->IsTerminated();
}

bool ReasoningStructuredGenerator::IsCompleted() const {
  if (is_in_reasoning_) {
    return false;
  }
  return inner_generator_->IsCompleted();
}

bool ReasoningStructuredGenerator::IsValid() const {
  return inner_generator_ && inner_generator_->IsValid();
}

StructuredConstraintType ReasoningStructuredGenerator::GetConstraintType() const {
  return inner_generator_->GetConstraintType();
}

}  // namespace ksana_llm
