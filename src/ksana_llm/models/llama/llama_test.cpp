/* Copyright 2023 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/

#include "ksana_llm/models/llama/llama.h"
#include <filesystem>
#include "flash_api.h"
#include "ksana_llm/models/llama/create_test_model.h"
#include "ksana_llm/samplers/sampler.h"
#include "test.h"

using namespace ksana_llm;
// 定义一个 LlamaTest 类,继承自 testing::Test
class LlamaTest : public testing::Test {
 protected:
  void SetUp() override {
    context_ = std::make_shared<Context>(1, 1);

    // 解析 config.ini,初始化 ModelConfig 以及 BlockManager
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    Singleton<Environment>::GetInstance()->ParseConfig(config_path);
    Singleton<Environment>::GetInstance()->GetModelConfig("llama", model_config);

    BlockManagerConfig block_manager_config;
    Singleton<Environment>::GetInstance()->GetBlockManagerConfig(block_manager_config);
    NLLM_LOG_WARNING << fmt::format("block_size {}", block_manager_config.device_allocator_config.block_size);

    block_manager = new BlockManager(block_manager_config, context_);
    block_manager->PreAllocateBlocks();
    SetBlockManager(block_manager);
  }

  void TearDown() override { delete block_manager; }

 protected:
  ModelConfig model_config;
  BlockManager *block_manager = nullptr;

  std::shared_ptr<Context> context_{nullptr};
};

TEST_F(LlamaTest, ForwardTest) {
  int device_id = 0;
  CUDA_CHECK(cudaSetDevice(device_id));
  std::filesystem::path ft_path(model_config.path);
  if (!std::filesystem::exists(ft_path)) {
    NLLM_LOG_ERROR << fmt::format("The given model path {} does not exist.", model_config.path);
    EXPECT_TRUE(std::filesystem::exists(ft_path));
  }

  cudaEvent_t start;
  cudaEvent_t stop;
  float milliseconds = 0;
  int rounds = 10;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::shared_ptr<BaseWeight> llama_weight = std::make_shared<LlamaWeight<half>>(model_config, 0, context_);
  std::shared_ptr<Llama<half>> llama = std::make_shared<Llama<half>>(model_config, 0, context_);

  // ContextDecode
  ForwardRequest forward;
  std::vector<int> input_ids = {233, 1681};
  forward.output_tokens = &input_ids;
  forward.logits_buf.resize(1);
  forward.logits_buf[0] = llama->GetLogitsPtr();
  forward.logits_offset = 0;
  std::vector<int> block_ids;
  GetBlockManager()->AllocateBlocks(1, block_ids);
  forward.kv_cache_ptrs.resize(1);
  GetBlockManager()->GetBlockPtrs(block_ids, forward.kv_cache_ptrs[0]);
  CUDA_CHECK(cudaMemset(forward.kv_cache_ptrs[0][0], 0, GetBlockManager()->GetBlockSize()));
  NLLM_LOG_WARNING << fmt::format("kv_cache_ptrs {} end {}", forward.kv_cache_ptrs[0][0],
                                  forward.kv_cache_ptrs[0][0] + (GetBlockManager()->GetBlockSize()));
  std::vector<ForwardRequest> forward_reqs = {forward};
  EXPECT_TRUE(llama->ContextDecode(llama_weight, forward_reqs).OK());

  std::vector<ForwardRequest> multi_forward_reqs = {forward, forward};
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < rounds; ++i) {
    llama->ContextDecode(llama_weight, multi_forward_reqs);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  EXPECT_TRUE((milliseconds / 10) < 35);

  // Sampling
  SamplingRequest sample_req;
  sample_req.logits_offset = forward_reqs[0].logits_offset;
  sample_req.output_tokens = forward_reqs[0].output_tokens;
  sample_req.logits_buf = forward_reqs[0].logits_buf;
  sample_req.model_config = &model_config;
  SamplingConfig sample_config;
  sample_config.beam_width = 1;
  sample_config.topk = 1;
  sample_config.topp = 0;
  sample_config.temperature = 0;
  sample_req.sampling_config = &sample_config;

  std::vector<SamplingRequest> sample_reqs = {sample_req};
  std::shared_ptr<Sampler> sampler = std::make_shared<Sampler>(device_id);
  sampler->Sampling(sample_reqs, context_->GetComputeStreams()[device_id]);
  EXPECT_EQ(29871, (*forward_reqs[0].output_tokens)[2]);

  // Decode
  EXPECT_TRUE(llama->Decode(llama_weight, forward_reqs).OK());
  sampler->Sampling(sample_reqs, context_->GetComputeStreams()[device_id]);
  EXPECT_EQ(29896, (*forward_reqs[0].output_tokens)[3]);

  EXPECT_TRUE(llama->Decode(llama_weight, forward_reqs).OK());
  sampler->Sampling(sample_reqs, context_->GetComputeStreams()[device_id]);
  EXPECT_EQ(29929, (*forward_reqs[0].output_tokens)[4]);

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < rounds; ++i) {
    llama->Decode(llama_weight, multi_forward_reqs);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

  EXPECT_TRUE((milliseconds / 10) < 30);

  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
}

TEST(TorchTensorTest, TorchTensorTest) {
  int device_id = 0;
  CUDA_CHECK(cudaSetDevice(device_id));
  // 设定张量的大小
  const int64_t size = 10;

  // 在GPU上分配内存
  float *a_ptr, *b_ptr, *c_ptr;
  CUDA_CHECK(cudaMalloc(&a_ptr, size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b_ptr, size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c_ptr, size * sizeof(float)));

  // 创建并初始化输入数据
  std::vector<float> a_host(size, 1.0), b_host(size, 2.0);

  // 将数据复制到GPU
  CUDA_CHECK(cudaMemcpy(a_ptr, a_host.data(), size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b_ptr, b_host.data(), size * sizeof(float), cudaMemcpyHostToDevice));

  // 创建torch::Tensor，它们共享GPU内存
  auto options = torch::TensorOptions().device(torch::kCUDA, device_id).dtype(torch::kFloat32);
  torch::Tensor a = torch::from_blob(a_ptr, {size}, options);
  torch::Tensor b = torch::from_blob(b_ptr, {size}, options);
  torch::Tensor c = torch::from_blob(c_ptr, {size}, options);

  // 计算a + b = c
  c.copy_(a.add_(b));
  std::ostringstream oss;
  // 传输到cpu打印
  oss << c.to(torch::kCPU);
  EXPECT_EQ('3', oss.str()[1]);

  // 将结果复制回CPU以进行验证
  std::vector<float> c_host(size);
  CUDA_CHECK(cudaMemcpy(c_host.data(), c_ptr, size * sizeof(float), cudaMemcpyDeviceToHost));

  // 验证结果
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(c_host[i], 3.0);
  }

  // 清理GPU内存
  cudaFree(a_ptr);
  cudaFree(b_ptr);
  cudaFree(c_ptr);
}