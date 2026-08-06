/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/speculative/dspark/dspark_decoder_adapter.h"

#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <vector>

#include "runtime/executor_infer_request.h"
#include "runtime/speculative/common/draft_generator_interface.h"
#include "runtime/speculative/dspark/dspark_decoder.h"

namespace ksana_llm {

class DSparkDecoderAdapterTest : public testing::Test {
 protected:
  void SetUp() override {
    // DSparkDecoder with minimal (null) dependencies — adapter shell tests do not
    // exercise the real forward path (which requires GPU components).
    decoder_ = std::make_shared<DSparkDecoder>(/*dspark_block_size=*/4, /*dspark_noise_token_id=*/99999,
                                               &infer_req_mutex_, /*serialize_dspark=*/false);
    adapter_ = std::make_unique<DSparkDecoderAdapter>(decoder_, /*model_executor=*/nullptr,
                                                      /*model_sampler=*/nullptr, /*model_builder=*/nullptr,
                                                      /*model_instance=*/nullptr);
  }

  std::shared_ptr<DSparkDecoder> decoder_;
  std::unique_ptr<DSparkDecoderAdapter> adapter_;
  std::mutex infer_req_mutex_;
};

TEST_F(DSparkDecoderAdapterTest, GetNameReturnsDSpark) { EXPECT_STREQ(adapter_->GetName(), "DSpark"); }

// GetSchedulingInfo: DSparkDecoder with block_size=4 returns all-4.
TEST_F(DSparkDecoderAdapterTest, GetSchedulingInfoDelegatesToDecoder) {
  auto info = adapter_->GetSchedulingInfo();
  EXPECT_EQ(info.extra_kv_tokens, 4u);
  EXPECT_EQ(info.extra_token_capacity, 4u);
  EXPECT_EQ(info.workload_multiplier, 4u);
}

// GetSchedulingInfo returns empty struct when dspark_decoder_ is null.
TEST_F(DSparkDecoderAdapterTest, GetSchedulingInfoEmptyWhenNoDecoder) {
  DSparkDecoderAdapter empty_adapter(/*dspark_decoder=*/nullptr, nullptr, nullptr, nullptr, nullptr);
  auto info = empty_adapter.GetSchedulingInfo();
  EXPECT_EQ(info.extra_kv_tokens, 0u);
  EXPECT_EQ(info.extra_token_capacity, 0u);
  EXPECT_EQ(info.workload_multiplier, 0u);
}

// BeforeForward is a no-op.
TEST_F(DSparkDecoderAdapterTest, BeforeForwardIsNoOp) {
  SpecContext ctx;
  adapter_->BeforeForward(ctx);
  SUCCEED();
}

// AfterForward is a no-op.
TEST_F(DSparkDecoderAdapterTest, AfterForwardIsNoOp) {
  SpecContext ctx;
  adapter_->AfterForward(ctx);
  SUCCEED();
}

// ExecuteForward returns early when ctx.reqs/hidden_unit/schedule_output are null.
TEST_F(DSparkDecoderAdapterTest, ExecuteForwardEarlyReturnWhenNullCtxMembers) {
  SpecContext ctx;
  ctx.reqs = nullptr;
  adapter_->ExecuteForward(ctx);
  SUCCEED();

  auto req = std::make_shared<ExecutorInferRequest>();
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req};
  ctx.reqs = &reqs;
  ctx.hidden_unit = nullptr;
  ctx.schedule_output = nullptr;
  adapter_->ExecuteForward(ctx);
  SUCCEED();
}

// Verify delegates to DefaultFilterDraftTokens.
TEST_F(DSparkDecoderAdapterTest, VerifyDelegatesToDefaultFilterDraftTokens) {
  auto req = std::make_shared<ExecutorInferRequest>();
  req->req_id = 1;
  req->spec_state.draft_tokens.mtp = {10, 20};
  req->sampling_result_tokens = {10, 20, 30};
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req};
  SpecContext ctx = SpecContext::ForVerify(&reqs);

  adapter_->Verify(ctx);

  EXPECT_EQ(req->spec_state.accepted_tokens.size(), 2u);
  EXPECT_EQ(req->generated_tokens[0], 30);
}

// Verify is a no-op when ctx.reqs is null.
TEST_F(DSparkDecoderAdapterTest, VerifyNoopWhenReqsNull) {
  SpecContext ctx;
  ctx.reqs = nullptr;
  adapter_->Verify(ctx);
  SUCCEED();
}

// IsCombinableWith: nullptr is always compatible.
TEST_F(DSparkDecoderAdapterTest, IsCombinableWithNullptrIsTrue) { EXPECT_TRUE(adapter_->IsCombinableWith(nullptr)); }

// DSpark rejects only MTP.
class NamedFakeDecoderForDSpark : public SpeculativeDecoderInterface {
 public:
  explicit NamedFakeDecoderForDSpark(const char* name) : name_(name) {}
  const char* GetName() const override { return name_; }
  SpecSchedulingInfo GetSchedulingInfo() const override { return {}; }
  void BeforeForward(SpecContext&) override {}
  void ExecuteForward(SpecContext&) override {}
  void AfterForward(SpecContext&) override {}
  void Verify(SpecContext&) override {}
  bool IsCombinableWith(const SpeculativeDecoderInterface*) const override { return true; }

 private:
  const char* name_;
};

TEST_F(DSparkDecoderAdapterTest, IsCombinableWithRejectsOnlyMtp) {
  NamedFakeDecoderForDSpark mtp("MTP");
  NamedFakeDecoderForDSpark ptp("PTP");
  EXPECT_FALSE(adapter_->IsCombinableWith(&mtp));
  EXPECT_TRUE(adapter_->IsCombinableWith(&ptp));
}

}  // namespace ksana_llm
