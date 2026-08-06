/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "runtime/speculative/common/draft_generator_interface.h"

namespace ksana_llm {

class PtpGenerator : public DraftGeneratorInterface {
 public:
  PtpGenerator(size_t ptp_step_num, uint32_t ptp_token_id);

  void GenerateDraft(std::shared_ptr<ExecutorInferRequest> req) override;

  // Override to implement PTP-specific draft token filtering logic
  size_t FilterDraftTokens(std::shared_ptr<ExecutorInferRequest> req) override;

  // PTP: extra_token_capacity = 2 * ptp_step_num (verify + placeholder tokens)
  SpecSchedulingInfo GetSchedulingInfo() const override;

 private:
  size_t ptp_step_num_ = 0;
  uint32_t ptp_token_id_ = 0;
};

}  // namespace ksana_llm