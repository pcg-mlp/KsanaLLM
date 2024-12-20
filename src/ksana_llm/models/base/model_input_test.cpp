/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>
#include <filesystem>

#include "ksana_llm/models/base/model_input.h"
#include "ksana_llm/models/common/common_model.h"

namespace ksana_llm {

class ModelInputTest : public testing::Test {
 protected:
  void SetUp() override {
    int rank = 0;
    auto context = std::make_shared<Context>(1, 1);

    // Parse the yaml config file.
    const auto& env = Singleton<Environment>::GetInstance();
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();
    env->ParseConfig(config_path);

    // Initialize the model config
    ModelConfig model_config;
    env->GetModelConfig("", model_config);

    // Initialize the block manager.
    env->InitializeBlockManagerConfig();
    BlockManagerConfig& block_manager_config = env->block_manager_config_;
    block_manager_config.block_host_memory_factor = 0.0;
    block_manager_ = std::make_unique<BlockManager>(block_manager_config, context);
    SetBlockManager(block_manager_.get());

    // Allocate blocks.
    block_manager_->PreAllocateBlocks();

    // Initialize the model input object.
    model_input_ = std::make_unique<ModelInput>(model_config, rank, context);
  }

  void TearDown() override {}

 private:
  std::unique_ptr<BlockManager> block_manager_;

  std::unique_ptr<ModelInput> model_input_;
};

TEST_F(ModelInputTest, CheckUseCacheTest) {
  // Construct forward requests as test input.
  SamplingConfig sampling_config1, sampling_config2;
  sampling_config1.max_new_tokens = 1;
  sampling_config2.max_new_tokens = 2;
  ForwardRequest forward_req1, forward_req2;
  forward_req1.sampling_config = &sampling_config1;
  forward_req2.sampling_config = &sampling_config2;
  std::vector<ForwardRequest> forward_reqs = {forward_req1, forward_req2};

  const auto& env = Singleton<Environment>::GetInstance();
  CacheManagerConfig& cache_manager_config = env->cache_manager_config_;

  // Test case 1: All the caching is disabled and all the requests only require the next token.
  EXPECT_FALSE(env->IsPrefixCachingEnabled());
  EXPECT_FALSE(env->IsFlexibleCachingEnabled());
  model_input_->multi_token_request_num = 1;
  model_input_->CheckUseCache(forward_reqs);
  EXPECT_FALSE(model_input_->use_cache);

  // Test case 2: All the caching is disabled but some requests require more than one token.
  model_input_->multi_token_request_num = 2;
  model_input_->CheckUseCache(forward_reqs);
  EXPECT_TRUE(model_input_->use_cache);

  // Test case 3: Prefix caching is enabled.
  cache_manager_config.enable_prefix_caching = true;
  EXPECT_TRUE(env->IsPrefixCachingEnabled());
  EXPECT_FALSE(env->IsFlexibleCachingEnabled());
  model_input_->multi_token_request_num = 1;
  model_input_->CheckUseCache(forward_reqs);
  EXPECT_TRUE(model_input_->use_cache);

  // Test case 4: Flexible caching is enabled.
  cache_manager_config.enable_prefix_caching = false;
  cache_manager_config.min_flexible_cache_num = 256;
  EXPECT_FALSE(env->IsPrefixCachingEnabled());
  EXPECT_TRUE(env->IsFlexibleCachingEnabled());
  model_input_->multi_token_request_num = 1;
  model_input_->CheckUseCache(forward_reqs);
  EXPECT_TRUE(model_input_->use_cache);
}

}  // namespace ksana_llm
