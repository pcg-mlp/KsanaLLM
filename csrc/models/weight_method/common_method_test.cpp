/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/weight_method/common_method.h"

#include <memory>
#include <string>
#include <unordered_map>

#include "helpers/model_test_helper.h"
#include "test.h"

#include "configure/context.h"
#include "configure/environment.h"
#include "models/base/common_model_weight_loader.h"
#include "utils/singleton.h"
#include "utils/tensor.h"

using namespace ksana_llm;

class CommonMethodTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();

    const auto& env = Singleton<Environment>::GetInstance();

    context_ = std::make_shared<Context>();
    context_->global.reset(new GlobalContext(1, 1, 1, env));
    context_->global->InitDistributedConfig();

    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context_->device = std::make_shared<DeviceContext>(1);
    context_->device->ext->InitNcclParam(nccl_uid);

    env_ = env;
    model_config_ = std::make_shared<BaseModelConfig>();
    common_weight_loader_ = std::make_shared<CommonModelWeightLoader>(model_config_, env_, context_);

    // 创建 CommonMethod 实例
    common_method_ = std::make_shared<CommonMethod>(common_weight_loader_, tp_);
  }

  void TearDown() override { device_model_weights_.clear(); }

 protected:
  std::shared_ptr<Context> context_{nullptr};
  std::shared_ptr<Environment> env_{nullptr};
  std::shared_ptr<BaseModelConfig> model_config_{nullptr};
  std::shared_ptr<CommonModelWeightLoader> common_weight_loader_{nullptr};
  std::shared_ptr<CommonMethod> common_method_{nullptr};
  std::unordered_map<std::string, Tensor> device_model_weights_;

  int dev_rank_{0};
  int tp_{2};
};

// 测试 load_attn_q_k_v_proj 和 process_attn_qkv_proj 的综合测试
TEST_F(CommonMethodTest, TestLoadAndProcessAttnQKVProj) {
#ifdef ENABLE_CUDA
  // 定义测试参数 - 使用不同的 N 和 K 来避免取巧
  const size_t N = 1024;  // 第一维
  const size_t K = 768;   // 第二维

  // 创建 q_proj, k_proj, v_proj 的权重
  std::string q_proj_name = "model.layers.0.self_attn.q_proj.weight";
  std::string k_proj_name = "model.layers.0.self_attn.k_proj.weight";
  std::string v_proj_name = "model.layers.0.self_attn.v_proj.weight";

  // 创建测试张量，形状为 (N, K)
  Tensor q_weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {N, K});
  Tensor k_weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {N, K});
  Tensor v_weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {N, K});

  // 测试 load_attn_q_k_v_proj 方法
  // 加载 q_proj
  common_method_->load_attn_q_k_v_proj(device_model_weights_, q_proj_name, q_weight, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(q_proj_name) != device_model_weights_.end());

  // 加载 k_proj
  common_method_->load_attn_q_k_v_proj(device_model_weights_, k_proj_name, k_weight, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(k_proj_name) != device_model_weights_.end());

  // 加载 v_proj
  common_method_->load_attn_q_k_v_proj(device_model_weights_, v_proj_name, v_weight, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(v_proj_name) != device_model_weights_.end());

  // 验证加载后的张量形状（应该被 tensor parallel split）
  const Tensor& loaded_q = device_model_weights_[q_proj_name];
  const Tensor& loaded_k = device_model_weights_[k_proj_name];
  const Tensor& loaded_v = device_model_weights_[v_proj_name];

  // 由于使用了 RowPara 模式，第一维（N）应该被分割
  size_t expected_N_split = N / tp_;
  EXPECT_EQ(loaded_q.shape[0], expected_N_split);
  EXPECT_EQ(loaded_k.shape[0], expected_N_split);
  EXPECT_EQ(loaded_v.shape[0], expected_N_split);
  // 第二维（K）保持不变
  EXPECT_EQ(loaded_q.shape[1], K);
  EXPECT_EQ(loaded_k.shape[1], K);
  EXPECT_EQ(loaded_v.shape[1], K);

  // 测试 process_attn_qkv_proj 方法
  // 注意：weight_prefix_name 应该是 "model.layers.0.self_attn.q_proj."，不包含 "weight"
  std::string weight_prefix_name = "model.layers.0.self_attn.q_proj.";
  common_method_->process_attn_qkv_proj(device_model_weights_, weight_prefix_name, dev_rank_);

  // 验证合并后的 qkv 权重存在
  std::string qkv_proj_name = "model.layers.0.self_attn.query_key_value.weight";
  EXPECT_TRUE(device_model_weights_.find(qkv_proj_name) != device_model_weights_.end());

  // 验证合并后的张量形状
  const Tensor& merged_qkv = device_model_weights_[qkv_proj_name];
  EXPECT_EQ(merged_qkv.shape.size(), 2);

  // 合并后：3个 (N/tp, K) 合并成 (3*N/tp, K)
  // 然后 Permute2D 转置：(3*N/tp, K) -> (K, 3*N/tp)
  EXPECT_EQ(merged_qkv.shape[0], K);
  EXPECT_EQ(merged_qkv.shape[1], 3 * expected_N_split);
#endif
}

// 测试 load_attn_o_proj 和 process_attn_o_proj 的综合测试
TEST_F(CommonMethodTest, TestLoadAndProcessAttnOProj) {
#ifdef ENABLE_CUDA
  // 定义测试参数 - 使用不同的 N 和 K 来避免取巧
  const size_t N = 1024;  // 第一维
  const size_t K = 768;   // 第二维

  // 创建 o_proj 的权重
  std::string o_proj_name = "model.layers.0.self_attn.o_proj.weight";

  // 创建测试张量，形状为 (N, K)
  Tensor o_weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {N, K});

  // 测试 load_attn_o_proj 方法
  // 加载 o_proj
  common_method_->load_attn_o_proj(device_model_weights_, o_proj_name, o_weight, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(o_proj_name) != device_model_weights_.end());

  // 验证加载后的张量形状（应该被 tensor parallel split）
  const Tensor& loaded_o = device_model_weights_[o_proj_name];

  // 由于使用了 ColPara 模式，第二维（K）应该被分割
  size_t expected_K_split = K / tp_;
  EXPECT_EQ(loaded_o.shape[0], N);
  EXPECT_EQ(loaded_o.shape[1], expected_K_split);

  // 测试 process_attn_o_proj 方法
  std::string weight_prefix_name = "model.layers.0.self_attn.o_proj.";
  common_method_->process_attn_o_proj(device_model_weights_, weight_prefix_name, dev_rank_);

  // 验证处理后的张量形状
  const Tensor& processed_o = device_model_weights_[o_proj_name];
  EXPECT_EQ(processed_o.shape.size(), 2);

  // process 只做 Permute2D 转置：(N, K/tp) -> (K/tp, N)
  EXPECT_EQ(processed_o.shape[0], expected_K_split);
  EXPECT_EQ(processed_o.shape[1], N);
#endif
}

// 测试 load_mlp_gate_up_proj 和 process_mlp_gate_up_proj 的综合测试
TEST_F(CommonMethodTest, TestLoadAndProcessMlpGateUpProj) {
#ifdef ENABLE_CUDA
  // 定义测试参数 - 使用不同的 N 和 K 来避免取巧
  const size_t N = 2048;  // 第一维
  const size_t K = 1536;  // 第二维

  // 创建 gate_proj 和 up_proj 的权重
  std::string gate_proj_name = "model.layers.0.mlp.gate_proj.weight";
  std::string up_proj_name = "model.layers.0.mlp.up_proj.weight";

  // 创建测试张量，形状为 (N, K)
  Tensor gate_weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {N, K});
  Tensor up_weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {N, K});

  // 测试 load_mlp_gate_up_proj 方法
  // 加载 gate_proj
  common_method_->load_mlp_gate_up_proj(device_model_weights_, gate_proj_name, gate_weight, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(gate_proj_name) != device_model_weights_.end());

  // 加载 up_proj
  common_method_->load_mlp_gate_up_proj(device_model_weights_, up_proj_name, up_weight, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(up_proj_name) != device_model_weights_.end());

  // 验证加载后的张量形状（应该被 tensor parallel split）
  const Tensor& loaded_gate = device_model_weights_[gate_proj_name];
  const Tensor& loaded_up = device_model_weights_[up_proj_name];

  // 由于使用了 RowPara 模式，第一维（N）应该被分割
  size_t expected_N_split = N / tp_;
  EXPECT_EQ(loaded_gate.shape[0], expected_N_split);
  EXPECT_EQ(loaded_up.shape[0], expected_N_split);
  // 第二维（K）保持不变
  EXPECT_EQ(loaded_gate.shape[1], K);
  EXPECT_EQ(loaded_up.shape[1], K);

  // 测试 process_mlp_gate_up_proj 方法
  std::string weight_prefix_name = "model.layers.0.mlp.gate_proj.";
  common_method_->process_mlp_gate_up_proj(device_model_weights_, weight_prefix_name, dev_rank_);

  // 验证合并后的 gate_up 权重存在
  std::string gate_up_proj_name = "model.layers.0.mlp.gate_up_proj.weight";
  EXPECT_TRUE(device_model_weights_.find(gate_up_proj_name) != device_model_weights_.end());

  // 验证合并后的张量形状
  const Tensor& merged_gate_up = device_model_weights_[gate_up_proj_name];
  EXPECT_EQ(merged_gate_up.shape.size(), 2);

  // 合并后：2个 (N/tp, K) 合并成 (2*N/tp, K)
  // 然后 Permute2D 转置：(2*N/tp, K) -> (K, 2*N/tp)
  EXPECT_EQ(merged_gate_up.shape[0], K);
  EXPECT_EQ(merged_gate_up.shape[1], 2 * expected_N_split);
#endif
}

// 测试 load_mlp_down_proj 和 process_mlp_down_proj 的综合测试
TEST_F(CommonMethodTest, TestLoadAndProcessMlpDownProj) {
#ifdef ENABLE_CUDA
  // 定义测试参数 - 使用不同的 N 和 K 来避免取巧
  const size_t N = 2048;  // 第一维
  const size_t K = 1536;  // 第二维

  // 创建 down_proj 的权重
  std::string down_proj_name = "model.layers.0.mlp.down_proj.weight";

  // 创建测试张量，形状为 (N, K)
  Tensor down_weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {N, K});

  // 测试 load_mlp_down_proj 方法
  // 加载 down_proj
  common_method_->load_mlp_down_proj(device_model_weights_, down_proj_name, down_weight, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(down_proj_name) != device_model_weights_.end());

  // 验证加载后的张量形状（应该被 tensor parallel split）
  const Tensor& loaded_down = device_model_weights_[down_proj_name];

  // 由于使用了 ColPara 模式，第二维（K）应该被分割
  size_t expected_K_split = K / tp_;
  EXPECT_EQ(loaded_down.shape[0], N);
  EXPECT_EQ(loaded_down.shape[1], expected_K_split);

  // 测试 process_mlp_down_proj 方法
  std::string weight_prefix_name = "model.layers.0.mlp.down_proj.";
  common_method_->process_mlp_down_proj(device_model_weights_, weight_prefix_name, dev_rank_);

  // 验证处理后的张量形状
  const Tensor& processed_down = device_model_weights_[down_proj_name];
  EXPECT_EQ(processed_down.shape.size(), 2);

  // process 只做 Permute2D 转置：(N, K/tp) -> (K/tp, N)
  EXPECT_EQ(processed_down.shape[0], expected_K_split);
  EXPECT_EQ(processed_down.shape[1], N);
#endif
}

// 测试 load_norm 的测试
TEST_F(CommonMethodTest, TestLoadNorm) {
#ifdef ENABLE_CUDA
  // 定义测试参数 - norm 通常是一维的 normalization 权重
  const size_t hidden_size = 4096;  // hidden dimension

  // 创建 norm 的权重（例如 model.norm.weight）
  std::string norm_name = "model.norm.weight";

  // 创建测试张量，形状为 (hidden_size,)
  Tensor norm_weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {hidden_size});

  // 测试 load_norm 方法
  // 加载 norm
  common_method_->load_norm(device_model_weights_, norm_name, norm_weight, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(norm_name) != device_model_weights_.end());

  // 验证加载后的张量形状（不应该被 tensor parallel split，因为使用的是 MoveToDevice）
  const Tensor& loaded_norm = device_model_weights_[norm_name];

  // norm 不进行 tensor parallel split，形状应该保持不变
  EXPECT_EQ(loaded_norm.shape.size(), 1);
  EXPECT_EQ(loaded_norm.shape[0], hidden_size);

  // 验证张量已经被移动到设备上
  EXPECT_EQ(loaded_norm.location, MemoryLocation::LOCATION_DEVICE);
#endif
}

// 测试 load_embed_tokens 的测试
TEST_F(CommonMethodTest, TestLoadEmbedTokens) {
#ifdef ENABLE_CUDA
  // 定义测试参数 - embed_tokens 通常是二维的 embedding 权重
  const size_t vocab_size = 32000;  // 词汇表大小
  const size_t hidden_size = 4096;  // hidden dimension

  // 创建 embed_tokens 的权重
  std::string embed_tokens_name = "model.embed_tokens.weight";

  // 创建测试张量，形状为 (vocab_size, hidden_size)
  Tensor embed_weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {vocab_size, hidden_size});

  // 测试 load_embed_tokens 方法
  // 加载 embed_tokens
  common_method_->load_embed_tokens(device_model_weights_, embed_tokens_name, embed_weight, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(embed_tokens_name) != device_model_weights_.end());

  // 验证加载后的张量形状（应该被 tensor parallel split）
  const Tensor& loaded_embed = device_model_weights_[embed_tokens_name];

  // 由于使用了 ColPara 模式，第二维（hidden_size）应该被分割
  size_t expected_hidden_split = hidden_size / tp_;
  EXPECT_EQ(loaded_embed.shape[0], vocab_size);
  EXPECT_EQ(loaded_embed.shape[1], expected_hidden_split);

  // 验证张量已经被移动到设备上
  EXPECT_EQ(loaded_embed.location, MemoryLocation::LOCATION_DEVICE);
#endif
}

// 测试 load_attn_norm 的测试
TEST_F(CommonMethodTest, TestLoadAttnNorm) {
#ifdef ENABLE_CUDA
  // 定义测试参数 - attn_norm 通常是一维的 normalization 权重
  const size_t hidden_size = 4096;  // hidden dimension

  // 创建 attn_norm 的权重
  std::string attn_norm_name = "model.layers.0.input_layernorm.weight";

  // 创建测试张量，形状为 (hidden_size,)
  Tensor norm_weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {hidden_size});

  // 测试 load_attn_norm 方法
  // 加载 attn_norm
  common_method_->load_attn_norm(device_model_weights_, attn_norm_name, norm_weight, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(attn_norm_name) != device_model_weights_.end());

  // 验证加载后的张量形状（不应该被 tensor parallel split，因为使用的是 MoveToDevice）
  const Tensor& loaded_norm = device_model_weights_[attn_norm_name];

  // attn_norm 不进行 tensor parallel split，形状应该保持不变
  EXPECT_EQ(loaded_norm.shape.size(), 1);
  EXPECT_EQ(loaded_norm.shape[0], hidden_size);

  // 验证张量已经被移动到设备上
  EXPECT_EQ(loaded_norm.location, MemoryLocation::LOCATION_DEVICE);
#endif
}

// 测试 load_lm_head 和 process_lm_head 的综合测试
TEST_F(CommonMethodTest, TestLoadAndProcessLmHead) {
#ifdef ENABLE_CUDA
  // 定义测试参数 - 使用不同的 N 和 K 来避免取巧
  const size_t N = 32000;  // vocab_size (第一维)
  const size_t K = 4096;   // hidden_size (第二维)

  // 创建 lm_head 的权重
  std::string lm_head_name = "lm_head.weight";

  // 创建测试张量，形状为 (N, K)
  Tensor lm_head_weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {N, K});

  // 测试 load_lm_head 方法
  // 加载 lm_head
  common_method_->load_lm_head(device_model_weights_, lm_head_name, lm_head_weight, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(lm_head_name) != device_model_weights_.end());

  // 验证加载后的张量形状（应该被 tensor parallel split）
  const Tensor& loaded_lm_head = device_model_weights_[lm_head_name];

  // 由于使用了 RowPara 模式，第一维（N）应该被分割
  size_t expected_N_split = N / tp_;
  EXPECT_EQ(loaded_lm_head.shape[0], expected_N_split);
  // 第二维（K）保持不变
  EXPECT_EQ(loaded_lm_head.shape[1], K);

  // 验证张量已经被移动到设备上
  EXPECT_EQ(loaded_lm_head.location, MemoryLocation::LOCATION_DEVICE);

  // 测试 process_lm_head 方法
  std::string weight_prefix_name = "lm_head.";
  common_method_->process_lm_head(device_model_weights_, weight_prefix_name, dev_rank_);

  // 验证处理后的张量形状
  const Tensor& processed_lm_head = device_model_weights_[lm_head_name];
  EXPECT_EQ(processed_lm_head.shape.size(), 2);

  // process 只做 Permute2D 转置：(N/tp, K) -> (K, N/tp)
  EXPECT_EQ(processed_lm_head.shape[0], K);
  EXPECT_EQ(processed_lm_head.shape[1], expected_N_split);
#endif
}

// 测试 process_attn_qkv_bias 的测试
TEST_F(CommonMethodTest, TestProcessAttnQKVBias) {
#ifdef ENABLE_CUDA
  // 定义测试参数 - 使用不同的 q、k、v bias 大小
  // q_bias 大小为 3584，k_bias 和 v_bias 大小为 512
  const size_t q_bias_size = 3584;
  const size_t k_bias_size = 512;
  const size_t v_bias_size = 512;

  // 创建 q_proj.bias, k_proj.bias, v_proj.bias 的权重名称
  std::string q_bias_name = "model.layers.0.self_attn.q_proj.bias";
  std::string k_bias_name = "model.layers.0.self_attn.k_proj.bias";
  std::string v_bias_name = "model.layers.0.self_attn.v_proj.bias";

  // 创建测试张量，形状为一维，类型为 BF16
  Tensor q_bias(MemoryLocation::LOCATION_HOST, DataType::TYPE_BF16, {q_bias_size});
  Tensor k_bias(MemoryLocation::LOCATION_HOST, DataType::TYPE_BF16, {k_bias_size});
  Tensor v_bias(MemoryLocation::LOCATION_HOST, DataType::TYPE_BF16, {v_bias_size});

  // 使用 load_attn_q_k_v_proj 方法加载 bias（该方法也处理 bias）
  common_method_->load_attn_q_k_v_proj(device_model_weights_, q_bias_name, q_bias, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(q_bias_name) != device_model_weights_.end());

  common_method_->load_attn_q_k_v_proj(device_model_weights_, k_bias_name, k_bias, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(k_bias_name) != device_model_weights_.end());

  common_method_->load_attn_q_k_v_proj(device_model_weights_, v_bias_name, v_bias, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(v_bias_name) != device_model_weights_.end());

  // 验证加载后的张量形状（应该被 tensor parallel split）
  const Tensor& loaded_q_bias = device_model_weights_[q_bias_name];
  const Tensor& loaded_k_bias = device_model_weights_[k_bias_name];
  const Tensor& loaded_v_bias = device_model_weights_[v_bias_name];

  // 由于使用了 RowPara 模式，bias 应该被分割
  size_t expected_q_bias_split = q_bias_size / tp_;  // 3584 / 2 = 1792
  size_t expected_k_bias_split = k_bias_size / tp_;  // 512 / 2 = 256
  size_t expected_v_bias_split = v_bias_size / tp_;  // 512 / 2 = 256

  EXPECT_EQ(loaded_q_bias.shape.size(), 1);
  EXPECT_EQ(loaded_q_bias.shape[0], expected_q_bias_split);
  EXPECT_EQ(loaded_k_bias.shape.size(), 1);
  EXPECT_EQ(loaded_k_bias.shape[0], expected_k_bias_split);
  EXPECT_EQ(loaded_v_bias.shape.size(), 1);
  EXPECT_EQ(loaded_v_bias.shape[0], expected_v_bias_split);

  // 验证数据类型为 BF16
  EXPECT_EQ(loaded_q_bias.dtype, DataType::TYPE_BF16);
  EXPECT_EQ(loaded_k_bias.dtype, DataType::TYPE_BF16);
  EXPECT_EQ(loaded_v_bias.dtype, DataType::TYPE_BF16);

  // 测试 process_attn_qkv_bias 方法
  // 注意：weight_prefix_name 应该是 "model.layers.0.self_attn.q_proj."，不包含 "bias"
  std::string weight_prefix_name = "model.layers.0.self_attn.q_proj.";
  common_method_->process_attn_qkv_bias(device_model_weights_, weight_prefix_name, dev_rank_);

  // 验证合并后的 qkv bias 存在
  std::string qkv_bias_name = "model.layers.0.self_attn.query_key_value.bias";
  EXPECT_TRUE(device_model_weights_.find(qkv_bias_name) != device_model_weights_.end());

  // 验证合并后的张量形状
  const Tensor& merged_qkv_bias = device_model_weights_[qkv_bias_name];

  // 根据 process_attn_qkv_bias 的实现：
  // kv_head_dims = k_bias.shape[0] = 256
  // q_head_dims = q_bias.shape[0] = 1792
  // shape = {1, q_head_dims / kv_head_dims + 2, kv_head_dims}
  //       = {1, 1792 / 256 + 2, 256}
  //       = {1, 7 + 2, 256}
  //       = {1, 9, 256}
  EXPECT_EQ(merged_qkv_bias.shape.size(), 3);
  EXPECT_EQ(merged_qkv_bias.shape[0], 1);
  EXPECT_EQ(merged_qkv_bias.shape[1], expected_q_bias_split / expected_k_bias_split + 2);  // 1792/256 + 2 = 9
  EXPECT_EQ(merged_qkv_bias.shape[2], expected_k_bias_split);                              // 256

  // 验证数据类型保持为 BF16
  EXPECT_EQ(merged_qkv_bias.dtype, DataType::TYPE_BF16);
#endif
}
