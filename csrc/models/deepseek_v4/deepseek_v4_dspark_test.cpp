/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "models/deepseek_v4/deepseek_v4_test_base.h"

namespace ksana_llm {

class DeepSeekV4DSparkTest : public DeepSeekV4TestBase {
 protected:
  void SetUp() override {
    model_path_ = "/model/DeepSeek-V4-Flash-FP8-DSpark-Layer3-29207";
    Initialize();
  }

  void ConfigureDspark(DsparkConfig* dspark) override {
    dspark->dspark_block_size = 5;
    dspark->dspark_noise_token_id = 128799;
    dspark->dspark_markov_rank = 256;
    dspark->dspark_target_layer_ids = {0, 1, 2};
  }

  void RunForward() {
    CheckModelPathOrSkip();
    if (HasFatalFailure() || IsSkipped()) return;
    ASSERT_FALSE(model_config.dspark_config.dspark_target_layer_ids.empty())
        << "DSpark model must declare dspark_target_layer_ids";

    std::shared_ptr<BaseWeight> weight;
    std::unique_ptr<WeightInstance> weight_inst;
    auto model = CreateModel(&weight, &weight_inst);

    std::vector<FlexibleCachedCopyTask> flex_tasks;
    EmbeddingSlice embedding_slice;
    auto forward_reqs = BuildContextDecodeReqs(kInputIds, &flex_tasks, &embedding_slice);

    // Phase 1: kMain forward + sampling (smoke).
    std::vector<std::vector<int>> main_tokens;
    auto sampler = RunMainForwardAndSample(model, weight, forward_reqs, kInputIds, &main_tokens);
    std::cout << fmt::format("V4 DSpark main: tokens0={}, tokens1={}", main_tokens[0][0], main_tokens[1][0])
              << std::endl;

    // Phase 2: kDSpark forward + per-step ForwardHead + sampling.
    std::vector<std::vector<int>> draft_tokens;
    RunDSparkPhase(model, weight, forward_reqs, sampler, &draft_tokens);
    const size_t block_size = model_config.dspark_config.dspark_block_size;
    for (size_t r = 0; r < forward_reqs.size(); ++r) {
      EXPECT_EQ(draft_tokens[r].size(), block_size);
      std::cout << fmt::format("V4 DSpark req{} draft tokens: [{}]", r, fmt::join(draft_tokens[r], ", ")) << std::endl;
    }

    Finalize(model, weight);
  }
};

TEST_F(DeepSeekV4DSparkTest, ForwardFP8Test) {
#ifdef ENABLE_CUDA
  RunForward();
#endif
}

}  // namespace ksana_llm
