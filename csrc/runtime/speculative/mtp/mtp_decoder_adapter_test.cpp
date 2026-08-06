/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/speculative/mtp/mtp_decoder_adapter.h"

#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <vector>

#include "runtime/executor_infer_request.h"
#include "runtime/speculative/common/draft_generator_interface.h"
#include "runtime/speculative/mtp/mtp_decoder.h"

namespace ksana_llm {

class MtpDecoderAdapterTest : public testing::Test {
 protected:
  void SetUp() override {
    // MtpDecoder with trivial (null) dependencies — the adapter shell tests only
    // exercise paths that don't call into the real MtpDecoder::ExecuteForward.
    decoder_ =
        std::make_shared<MtpDecoder>(/*draft_model=*/nullptr, /*mtp_step_num=*/2,
                                     /*serialize_main_mtp=*/false, /*model_delegate=*/nullptr, &infer_req_mutex_);
    adapter_ = std::make_unique<MtpDecoderAdapter>(decoder_, /*model_executor=*/nullptr,
                                                   /*model_sampler=*/nullptr, /*model_builder=*/nullptr);
  }

  std::shared_ptr<MtpDecoder> decoder_;
  std::unique_ptr<MtpDecoderAdapter> adapter_;
  std::mutex infer_req_mutex_;
};

TEST_F(MtpDecoderAdapterTest, GetNameReturnsMTP) { EXPECT_STREQ(adapter_->GetName(), "MTP"); }

// GetSchedulingInfo delegates to MtpDecoder: extra_kv_tokens=step_num, extra_token_capacity=step_num-1,
// workload_multiplier=step_num.
TEST_F(MtpDecoderAdapterTest, GetSchedulingInfoDelegatesToDecoder) {
  auto info = adapter_->GetSchedulingInfo();
  EXPECT_EQ(info.extra_kv_tokens, 2u);
  EXPECT_EQ(info.extra_token_capacity, 1u);  // step_num - 1
  EXPECT_EQ(info.workload_multiplier, 2u);
}

// GetSchedulingInfo returns empty struct when mtp_decoder_ is null.
TEST_F(MtpDecoderAdapterTest, GetSchedulingInfoEmptyWhenNoDecoder) {
  MtpDecoderAdapter empty_adapter(/*mtp_decoder=*/nullptr, nullptr, nullptr, nullptr);
  auto info = empty_adapter.GetSchedulingInfo();
  EXPECT_EQ(info.extra_kv_tokens, 0u);
  EXPECT_EQ(info.extra_token_capacity, 0u);
  EXPECT_EQ(info.workload_multiplier, 0u);
}

// IsCombinableWith always returns true for any other decoder (including nullptr).
TEST_F(MtpDecoderAdapterTest, IsCombinableWithAlwaysTrue) {
  EXPECT_TRUE(adapter_->IsCombinableWith(nullptr));
  // Even with a fake decoder, MTP accepts everything.
}

// BeforeForward is currently a no-op (TODO in mtp_decoder_adapter.cpp:53).
TEST_F(MtpDecoderAdapterTest, BeforeForwardIsNoOp) {
  auto req = std::make_shared<ExecutorInferRequest>();
  req->req_id = 1;
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req};
  SpecContext ctx = SpecContext::ForDraftGen(&reqs, nullptr);
  adapter_->BeforeForward(ctx);
  SUCCEED();
}

// AfterForward is a no-op.
TEST_F(MtpDecoderAdapterTest, AfterForwardIsNoOp) {
  SpecContext ctx;
  adapter_->AfterForward(ctx);
  SUCCEED();
}

// ExecuteForward returns early when ctx.reqs/hidden_unit/schedule_output are null.
TEST_F(MtpDecoderAdapterTest, ExecuteForwardEarlyReturnWhenNullCtxMembers) {
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

// Verify delegates to DefaultFilterDraftTokens for every request.
TEST_F(MtpDecoderAdapterTest, VerifyDelegatesToDefaultFilterDraftTokens) {
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
TEST_F(MtpDecoderAdapterTest, VerifyNoopWhenReqsNull) {
  SpecContext ctx;
  ctx.reqs = nullptr;
  adapter_->Verify(ctx);
  SUCCEED();
}

// PreMainForward returns early when mtp_decoder_ is null.
TEST_F(MtpDecoderAdapterTest, PreMainForwardEarlyReturnWhenNoDecoder) {
  MtpDecoderAdapter empty_adapter(/*mtp_decoder=*/nullptr, nullptr, nullptr, nullptr);
  SpecContext ctx;
  empty_adapter.PreMainForward(ctx);
  SUCCEED();
}

// PreMainForward with a valid decoder: serialize_main_mtp is false,
// so the Wait/Reset branch is skipped. forwarding_context is null,
// so PrepareForMainForward is skipped as well.
TEST_F(MtpDecoderAdapterTest, PreMainForwardWithValidDecoderNoSerialization) {
  SpecContext ctx;
  ctx.forwarding_context = nullptr;
  adapter_->PreMainForward(ctx);
  SUCCEED();
}

}  // namespace ksana_llm
