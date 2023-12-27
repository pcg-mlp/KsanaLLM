/* Copyright 2023 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/

#include "numerous_llm/models/llama/llama.h"
#include "numerous_llm/models/llama/create_test_model.h"
#include "test.h"
#include "flash_api.h"

using namespace numerous_llm;
// 定义一个 LlamaTest 类,继承自 testing::Test
class LlamaTest : public testing::Test {
 protected:
  void SetUp() override {
    model_config.path = "/model/llama-ft/7B/1-gpu/";
    model_config.weight_data_type = TYPE_FP16;
    model_config.head_num = 32;
    model_config.size_per_head = 128;
    model_config.inter_size = 11008;
    model_config.num_layer = 32;
    model_config.vocab_size = 32000;
    model_config.tensor_para_size = 1;
    model_config.layernorm_eps = 1e-6;

    BlockManagerConfig block_manager_config;
    block_manager_config.cpu_allocator_config.blocks_num = 2;
    block_manager_config.cpu_allocator_config.block_size = 1024;
    block_manager_config.cpu_allocator_config.device = MEMORY_CPU_PINNED;
    block_manager_config.device_allocator_config.blocks_num = 2;
    block_manager_config.device_allocator_config.block_size = 1024;
    block_manager_config.device_allocator_config.device = MEMORY_GPU;

    context_ = std::make_shared<Context>(1, 1);

    // 使用配置创建一个 BlockManager 对象
    block_manager = new BlockManager(block_manager_config, context_);
    SetBlockManager(block_manager);
  }

  void TearDown() override { delete block_manager; }

 protected:
  ModelConfig model_config;
  BlockManager* block_manager = nullptr;

  std::shared_ptr<Context> context_{nullptr};
};

TEST_F(LlamaTest, ContextDecodeTest) {
  // 当环境中不包含该路径时, 下载该模型
  std::filesystem::path ft_path(model_config.path);
  if (!std::filesystem::exists(ft_path)) {
    NLLM_LOG_WARNING << fmt::format("The given model path {} does not exist. Generating a test model",
                                    model_config.path);
    std::filesystem::create_directories(model_config.path);
    create_model(model_config);
  }

  std::shared_ptr<BaseWeight> llama_weight = std::make_shared<LlamaWeight<half>>(model_config, 0, context_);
  std::shared_ptr<Llama<half>> llama = std::make_shared<Llama<half>>(model_config, 0, context_);

  ForwardRequest forward;
  std::vector<int> input_ids = {233, 1681};
  forward.output_tokens = &input_ids;
  forward.logits_buf.resize(1);
  std::vector<ForwardRequest> forward_reqs = {forward};
  llama->ContextDecode(llama_weight, forward_reqs);
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
