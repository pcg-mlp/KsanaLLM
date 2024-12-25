/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/torch.h>

#include <cstring>
#include <filesystem>
#include <random>

#include "ksana_llm/models/base/model_input.h"
#include "ksana_llm/models/common/common_model.h"
#include "tests/test.h"

namespace py = pybind11;

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

    // Initialize the random seed with 0.
    std::srand(0);
  }

  void TearDown() override {}

 private:
  std::unique_ptr<BlockManager> block_manager_;

  std::unique_ptr<ModelInput> model_input_;
};

TEST_F(ModelInputTest, PrepareInputRefitTest) {
  std::vector<float*> input_refit_emb_ptr;
  std::vector<std::pair<int64_t, int64_t>> input_refit_pos_pair;

  auto VerifyPrepareInputRefit = [&]() {
    const size_t input_refit_size = input_refit_emb_ptr.size();
    EXPECT_EQ(model_input_->cpu_input_refit_tensor.emb_fp32_ptr_tensor.shape.size(), 1);
    EXPECT_EQ(model_input_->cpu_input_refit_tensor.emb_fp32_ptr_tensor.shape[0], input_refit_size);
    EXPECT_EQ(model_input_->cpu_input_refit_tensor.pos_pair_tensor.shape.size(), 2);
    EXPECT_EQ(model_input_->cpu_input_refit_tensor.pos_pair_tensor.shape[0], input_refit_size);
    EXPECT_EQ(model_input_->cpu_input_refit_tensor.pos_pair_tensor.shape[1], 2);
    void** cpu_input_refit_emb_fp32_ptr =
        reinterpret_cast<void**>(model_input_->cpu_input_refit_tensor.emb_fp32_ptr_tensor.GetPtr<void>());
    int64_t* cpu_input_refit_pos_pair =
        reinterpret_cast<int64_t*>(model_input_->cpu_input_refit_tensor.pos_pair_tensor.GetPtr<void>());
    for (size_t i = 0; i < input_refit_size; i++) {
      EXPECT_EQ(cpu_input_refit_emb_fp32_ptr[i], input_refit_emb_ptr[i]);
      EXPECT_EQ(cpu_input_refit_pos_pair[i * 2], input_refit_pos_pair[i].first);
      EXPECT_EQ(cpu_input_refit_pos_pair[i * 2 + 1], input_refit_pos_pair[i].second);
    }
  };

  // Ensure that torch is imported, so that `THPVariableClass` is not nullptr.
  py::module torch = py::module::import("torch");

  // Test for each selected batch size.
  for (const int batch_size : {1, 3, 4}) {
    input_refit_emb_ptr.clear();
    input_refit_pos_pair.clear();

    std::vector<ForwardRequest> forward_reqs;

    // Reserve memory to avoid memory address being moved.
    std::vector<std::vector<int>> output_tokens;
    std::vector<EmbeddingSlice> embedding_slices;
    forward_reqs.reserve(batch_size);
    output_tokens.reserve(batch_size);
    embedding_slices.reserve(batch_size);

    model_input_->multi_token_request_num = batch_size;
    size_t pos_offset = 0;

    // Construct input refit embeddings.
    for (int i = 0; i < batch_size; i++) {
      ForwardRequest forward_req;
      const size_t output_tokens_size = std::rand() % 4096 + 10;
      output_tokens.emplace_back(output_tokens_size);
      forward_req.output_tokens = &output_tokens.back();
      EmbeddingSlice embedding_slice;
      const int input_refit_size = std::rand() % 3 + 1;
      for (int j = 0; j < input_refit_size; j++) {
        const size_t embedding_size = std::rand() % output_tokens_size + 1;
        const size_t embedding_start_pos = std::rand() % embedding_size;
        embedding_slice.embeddings.emplace_back(embedding_size);
        embedding_slice.pos.push_back(embedding_start_pos);
        input_refit_emb_ptr.emplace_back(embedding_slice.embeddings.back().data());
        input_refit_pos_pair.emplace_back(pos_offset + embedding_start_pos, embedding_size);
      }
      embedding_slices.push_back(std::move(embedding_slice));
      forward_req.input_refit_embedding = &embedding_slices.back();
      forward_reqs.push_back(std::move(forward_req));
      pos_offset += output_tokens_size;
    }

    // Parse and load the input refit embeddings.
    model_input_->PrepareInputRefit(forward_reqs);

    // Check the result of PrepareInputRefit.
    VerifyPrepareInputRefit();

    // Construct input refit embedding tensors.
    input_refit_emb_ptr.clear();
    for (int i = 0; i < batch_size; i++) {
      ForwardRequest& forward_req = forward_reqs[i];
      auto& embedding_slice = forward_req.input_refit_embedding;
      embedding_slice->embedding_tensors.reserve(embedding_slice->embeddings.size());
      for (const auto& embedding : embedding_slice->embeddings) {
        torch::Tensor embedding_tensor = torch::randn(static_cast<int64_t>(embedding.size()), torch::kFloat32);
        input_refit_emb_ptr.push_back(reinterpret_cast<float*>(embedding_tensor.data_ptr()));
        {
          py::gil_scoped_acquire acquire;
          embedding_slice->embedding_tensors.push_back(
              py::reinterpret_steal<py::object>(THPVariable_Wrap(embedding_tensor)));
        }
      }
      embedding_slice->embeddings.clear();
    }

    // Parse and load the input refit embeddings.
    model_input_->PrepareInputRefit(forward_reqs);

    // Check the result of PrepareInputRefit.
    VerifyPrepareInputRefit();

    // Construct bad input.
    forward_reqs[0].input_refit_embedding->embedding_tensors.clear();
    EXPECT_THROW(
        try { model_input_->PrepareInputRefit(forward_reqs); } catch (const std::runtime_error& e) {
          EXPECT_NE(strstr(e.what(),
                           "`input_refit_pos.size()` should be equal to `input_refit_embeddings.size()` or "
                           "`input_refit_embedding_tensors.size()`."),
                    nullptr);
          throw;
        },
        std::runtime_error);
  }
}

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
