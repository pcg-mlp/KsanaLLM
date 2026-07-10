/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "models/deepseek_v4/deepseek_v4_test_base.h"

namespace ksana_llm {

class DeepSeekV4Test : public DeepSeekV4TestBase {
 protected:
  void SetUp() override {
    model_path_ = "/model/DeepSeek-V4-Flash-FP8-Layer3-28096-v3";
    Initialize();
  }

  void RunForward() {
    CheckModelPathOrSkip();
    if (HasFatalFailure() || IsSkipped()) return;

    std::shared_ptr<BaseWeight> weight;
    std::unique_ptr<WeightInstance> weight_inst;
    auto model = CreateModel(&weight, &weight_inst);

    std::vector<FlexibleCachedCopyTask> flex_tasks;
    EmbeddingSlice embedding_slice;
    auto forward_reqs = BuildContextDecodeReqs(kInputIds, &flex_tasks, &embedding_slice);

    std::vector<std::vector<int>> tokens_per_req;
    RunMainForwardAndSample(model, weight, forward_reqs, kInputIds, &tokens_per_req);
    std::cout << fmt::format("V4 forward: tokens0={}, tokens1={}", tokens_per_req[0][0], tokens_per_req[1][0])
              << std::endl;

    Finalize(model, weight);
  }
};

TEST_F(DeepSeekV4Test, ForwardFP8Test) {
#ifdef ENABLE_CUDA
  RunForward();
#endif
}

}  // namespace ksana_llm
