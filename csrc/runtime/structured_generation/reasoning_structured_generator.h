/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "runtime/structured_generation/structured_generator_interface.h"

namespace ksana_llm {

/*!
 * \brief Wrapper for structured generators that handles reasoning content.
 *
 * Delays the inner constraint generator until think_end_token_id is accepted.
 * Reasoning tokens and think_end are not forwarded to inner; only the
 * constrained suffix is. Rollback therefore peels newest-first across the
 * three accepted segments documented on the counters below.
 */
class ReasoningStructuredGenerator : public StructuredGeneratorInterface {
 public:
  /*!
   * \brief Constructor
   * \param inner_generator The underlying structured generator to wrap
   * \param think_end_token_id The token ID that marks the end of reasoning
   * \param start_in_reasoning Whether to start in reasoning mode
   *                          Set to true for models like DeepSeek-R1/QwQ that start with thinking
   *                          Set to false if reasoning needs to be explicitly triggered
   */
  ReasoningStructuredGenerator(std::shared_ptr<StructuredGeneratorInterface> inner_generator,
                               int think_end_token_id,
                               bool start_in_reasoning);

  virtual ~ReasoningStructuredGenerator() = default;

  // StructuredGeneratorInterface implementation
  bool AcceptToken(int token_id) override;
  bool FillNextTokenBitmask(void* next_token_bitmask) override;
  void Rollback(int rollback_token_num) override;
  void Reset() override;
  bool FindJumpForwardTokens(std::vector<int>& jump_tokens) override;
  bool IsTerminated() const override;
  bool IsCompleted() const override;
  bool IsValid() const override;
  StructuredConstraintType GetConstraintType() const override;

  /*!
   * \brief Check if currently in reasoning phase
   * \return true if in reasoning phase, false otherwise
   */
  bool IsInReasoningPhase() const { return is_in_reasoning_; }

 private:
  std::shared_ptr<StructuredGeneratorInterface> inner_generator_;
  int think_end_token_id_;
  const bool start_in_reasoning_;
  bool is_in_reasoning_;  // bitmask / jump / terminated; restored when think_end is peeled
  // Accepted segments, oldest → newest. Rollback peels the reverse order so
  // inner->Rollback() only covers generation_token_count_.
  size_t reasoning_token_count_ = 0;   // unconstrained body, not sent to inner
  bool reasoning_end_seen_ = false;    // single think_end token, not sent to inner
  size_t generation_token_count_ = 0;  // constrained tokens, sent to inner
};

}  // namespace ksana_llm
