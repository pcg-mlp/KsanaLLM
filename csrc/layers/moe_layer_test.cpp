/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#include "layers/layer_test_base.h"

#include "layers/cutlass_moe_layer.h"
#include "layers/grouped_topk_layer.h"
#include "layers/marlin_moe_layer.h"
#include "layers/moe_layer.h"
#include "models/common_moe/moe_config.h"
#include "utils/search_status.h"

#ifdef ENABLE_CUDA
#  ifdef ENABLE_FP8
#    include "layers/fp8_moe_layer.h"
#  endif
#endif

namespace ksana_llm {

TEST_F(LayerTest, Fp8MoeLayerTest) {
#ifdef ENABLE_CUDA
#  ifdef ENABLE_FP8
  if (!context_->global->IsGemmFp8Supported()) {
    return;
  }
  constexpr int kDeviceRank = 0;

  // params
  MoeScaleNormMode moe_scale_norm_mode = ksana_llm::MoeScaleNormMode::NO_NORM;
  size_t max_token_num = 4096;
  size_t expert_num = 4;
  size_t expert_hidden_size = 1024;
  size_t expert_inter_size = 2688;
  size_t expert_topk = 1;
  size_t tp_size = 1;
  bool use_vllm_moe = false;
  uint32_t num_expert_group = 1;
  uint32_t expert_groups_topk = 1;
  std::string scoring_func = "softmax";
  std::string topk_method = "greedy";
  bool norm_topk_prob = false;
  float routed_scaling_factor = 1.0f;
  bool use_e_score_correction_bias = false;
  bool enable_full_shared_expert = false;
  DataType fp8_weight_dtype = DataType::TYPE_INVALID;
  DataType int_weight_dtype = DataType::TYPE_INVALID;
  int group_size = 0;
  bool apply_weight = false;

  std::vector<std::any> params;
  params.push_back(moe_scale_norm_mode);
  params.push_back(max_token_num);
  params.push_back(expert_num);
  params.push_back(expert_hidden_size);
  params.push_back(expert_inter_size);
  params.push_back(expert_topk);
  params.push_back(tp_size);
  params.push_back(use_vllm_moe);
  params.push_back(num_expert_group);
  params.push_back(expert_groups_topk);
  params.push_back(scoring_func);
  params.push_back(topk_method);
  params.push_back(norm_topk_prob);
  params.push_back(routed_scaling_factor);
  params.push_back(use_e_score_correction_bias);
  params.push_back(enable_full_shared_expert);
  params.push_back(fp8_weight_dtype);
  params.push_back(int_weight_dtype);
  params.push_back(group_size);
  params.push_back(apply_weight);

  int num_tokens = 9;
  auto options = torch::TensorOptions().device(torch::kCUDA, kDeviceRank);

  // initialize input and weight tensor
  std::vector<Tensor> inputs(4);
  inputs[0] =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {static_cast<uint64_t>(num_tokens), expert_hidden_size});
  inputs[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {expert_num, expert_num});
  inputs[2] =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP8_E4M3, {expert_num, expert_inter_size * 2, expert_hidden_size});
  inputs[3] =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP8_E4M3, {expert_num, expert_hidden_size, expert_inter_size});
  for (int i = 0; i < inputs.size(); ++i) {
    torch::Tensor tensor = torch::from_blob(inputs[i].GetPtr<void>(),
                                            {std::vector<int64_t>(inputs[i].shape.begin(), inputs[i].shape.end())},
                                            options.dtype(GetTorchTypeFromDataType(inputs[i].dtype)));
    if (i < 2) {
      tensor.fill_(0.8213);
    } else {
      tensor.fill_(89);
    }
  }

  // initialize scales tensor
  std::vector<Tensor> scales(4);
  scales[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {1});
  scales[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {1});
  scales[2] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {1});
  scales[3] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {1});
  for (int i = 0; i < scales.size(); ++i) {
    torch::Tensor tensor = torch::from_blob(scales[i].GetPtr<void>(),
                                            {std::vector<int64_t>(scales[i].shape.begin(), scales[i].shape.end())},
                                            options.dtype(GetTorchTypeFromDataType(scales[i].dtype)));
    if (i < 2) {
      tensor.fill_(0.01);
    } else {
      tensor.fill_(1.8601190e-06);
    }
  }

  // binding scales
  inputs[2].input_scales = &scales[0];
  inputs[3].input_scales = &scales[1];
  inputs[2].weight_scales = &scales[2];
  inputs[3].weight_scales = &scales[3];

  // initialize output tensor
  std::vector<Tensor> outputs(1);
  outputs[0] =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {static_cast<uint64_t>(num_tokens), expert_hidden_size});

  // run moe_layer
  std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
  Fp8MoeLayer moe_layer = Fp8MoeLayer();
  moe_layer.Init(params, runtime_config, context_, kDeviceRank);
  moe_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(moe_layer.GetWorkspaceSize()));
  moe_layer.Preprocess(model_config, runtime_config);
  EXPECT_TRUE(moe_layer.Forward(inputs, outputs).OK());

  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  // check output
  // output is a repetition of 0.1768
  torch::Tensor tensor = torch::from_blob(outputs[0].GetPtr<void>(),
                                          {std::vector<int64_t>(outputs[0].shape.begin(), outputs[0].shape.end())},
                                          options.dtype(GetTorchTypeFromDataType(outputs[0].dtype)))
                             .cpu();
  EXPECT_TRUE(torch::all(torch::eq(tensor[0], 0.1768)).item<bool>());

#  endif
#endif
}

TEST_F(LayerTest, MarlinMoeLayerTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  int num_bits = 4;
  // params
  MoeScaleNormMode moe_scale_norm_mode = MoeScaleNormMode::NO_NORM;
  size_t max_token_num = 4096;
  size_t expert_num = 4;
  size_t expert_hidden_size = 1024;
  size_t expert_inter_size = 2688;
  size_t expert_topk = 1;
  size_t tp_size = 1;
  bool use_vllm_moe = false;
  uint32_t num_expert_group = 1;
  uint32_t expert_groups_topk = 1;
  std::string scoring_func = "softmax";
  std::string topk_method = "greedy";
  bool norm_topk_prob = false;
  float routed_scaling_factor = 1.0f;
  bool use_e_score_correction_bias = false;
  bool enable_full_shared_expert = false;
  DataType fp8_weight_dtype = DataType::TYPE_INVALID;
  DataType int_weight_dtype = DataType::TYPE_I4_GROUP;
  int group_size = 128;
  bool apply_weight = false;

  std::vector<std::any> params;
  params.push_back(moe_scale_norm_mode);
  params.push_back(max_token_num);
  params.push_back(expert_num);
  params.push_back(expert_hidden_size);
  params.push_back(expert_inter_size);
  params.push_back(expert_topk);
  params.push_back(tp_size);
  params.push_back(use_vllm_moe);
  params.push_back(num_expert_group);
  params.push_back(expert_groups_topk);
  params.push_back(scoring_func);
  params.push_back(topk_method);
  params.push_back(norm_topk_prob);
  params.push_back(routed_scaling_factor);
  params.push_back(use_e_score_correction_bias);
  params.push_back(enable_full_shared_expert);
  params.push_back(fp8_weight_dtype);
  params.push_back(int_weight_dtype);
  params.push_back(group_size);
  params.push_back(apply_weight);

  int num_tokens = 9;
  auto options = torch::TensorOptions().device(torch::kCUDA, kDeviceRank);

  // initialize input and weight tensor
  std::vector<Tensor> inputs(4);
  inputs[0] =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {static_cast<uint64_t>(num_tokens), expert_hidden_size});
  inputs[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {static_cast<uint64_t>(num_tokens), expert_num});
  inputs[2] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
                     {expert_num, expert_hidden_size / 16, 2 * expert_inter_size * (num_bits / 2)});
  inputs[3] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
                     {expert_num, expert_inter_size / 16, expert_hidden_size * (num_bits / 2)});
  for (size_t i = 0; i < inputs.size(); ++i) {
    torch::Tensor tensor = torch::from_blob(inputs[i].GetPtr<void>(),
                                            {std::vector<int64_t>(inputs[i].shape.begin(), inputs[i].shape.end())},
                                            options.dtype(GetTorchTypeFromDataType(inputs[i].dtype)));
    if (i < 2) {
      tensor.fill_(0.8213);
    } else {
      tensor.fill_(1754889370);
    }
  }

  // initialize scales tensor
  std::vector<Tensor> scales(2);
  scales[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
                     {expert_num, expert_hidden_size / group_size, expert_inter_size * 2});
  scales[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
                     {expert_num, expert_inter_size / group_size, expert_hidden_size});
  for (size_t i = 0; i < scales.size(); ++i) {
    torch::Tensor tensor = torch::from_blob(scales[i].GetPtr<void>(),
                                            {std::vector<int64_t>(scales[i].shape.begin(), scales[i].shape.end())},
                                            options.dtype(GetTorchTypeFromDataType(scales[i].dtype)));
    tensor.fill_(0.002735);
  }

  // binding scales
  inputs[2].scales = &scales[0];
  inputs[3].scales = &scales[1];

  // initialize output tensor
  std::vector<Tensor> outputs(1);
  outputs[0] =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {static_cast<uint64_t>(num_tokens), expert_hidden_size});

  // run moe_layer
  std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
  MarlinMoeLayer moe_layer = MarlinMoeLayer();
  moe_layer.Init(params, runtime_config, context_, kDeviceRank);
  moe_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(moe_layer.GetWorkspaceSize()));
  moe_layer.Preprocess(model_config, runtime_config);
  EXPECT_TRUE(moe_layer.Forward(inputs, outputs).OK());

  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  // check output
  // output is a repetition of: [10.984, 10.984, 10.984, 10.984, 10.984, 10.984, 10.984, 10.984,
  // 14.1, 14.1, 14.1, 14.1, 14.1, 14.1, 14.1, 14.1]
  torch::Tensor tensor = torch::from_blob(outputs[0].GetPtr<void>(),
                                          {std::vector<int64_t>(outputs[0].shape.begin(), outputs[0].shape.end())},
                                          options.dtype(GetTorchTypeFromDataType(outputs[0].dtype)))
                             .cpu()
                             .view({static_cast<int64_t>(outputs[0].shape[0] * outputs[0].shape[1] / 16), 2, 8})
                             .permute({1, 0, 2});
  EXPECT_TRUE(torch::all(torch::eq(tensor[0], 10.984)).item<bool>());
  EXPECT_TRUE(torch::all(torch::eq(tensor[1], 14.1)).item<bool>());
#endif
}

TEST_F(LayerTest, GroupedTopkLayerTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  using dtype = half_float::half;

  // 测试参数
  const int num_tokens = 4;
  const int num_experts = 8;
  const int topk = 2;
  const bool renormalize = true;
  const int num_expert_group = 4;
  const int topk_group = 2;
  const std::string scoring_func = "softmax";
  const float routed_scaling_factor = 1.0f;
  const bool use_e_score_correction_bias = false;

  // 创建 GroupedTopkLayer
  GroupedTopkLayer grouped_topk_layer;

  // 测试初始化
  GroupedTopkLayerParameters params{.topk = topk,
                                    .renormalize = renormalize,
                                    .num_expert_group = num_expert_group,
                                    .topk_group = topk_group,
                                    .scoring_func = scoring_func,
                                    .routed_scaling_factor = routed_scaling_factor,
                                    .use_e_score_correction_bias = use_e_score_correction_bias,
                                    .score_threshold = 0.f};

  EXPECT_TRUE(grouped_topk_layer.Init({params}, runtime_config, context_, kDeviceRank).OK());

  // 准备输入张量 - gating_output
  Tensor gating_output;
  gating_output = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
                         {static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)});

  // 初始化 gating_output 数据 - 使用不同的值来测试 topk 选择
  std::vector<dtype> gating_host(num_tokens * num_experts);
  for (int token = 0; token < num_tokens; ++token) {
    for (int expert = 0; expert < num_experts; ++expert) {
      // 为每个 token 设置不同的专家权重，确保 topk 选择有意义
      gating_host[token * num_experts + expert] = static_cast<dtype>((expert + token * 0.1f + 1.0f) / num_experts);
    }
  }
  MemcpyAsync(gating_output.GetPtr<void>(), gating_host.data(), gating_output.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());

  // 准备输出张量
  Tensor topk_weights, topk_ids;
  topk_weights =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {static_cast<size_t>(num_tokens), static_cast<size_t>(topk)});
  topk_ids =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {static_cast<size_t>(num_tokens), static_cast<size_t>(topk)});

  std::vector<Tensor> input_tensors = {gating_output};
  std::vector<Tensor> output_tensors = {topk_weights, topk_ids};

  StreamSynchronize(context_->device->GetMemoryManageStream());

  // 测试前向传播
  EXPECT_TRUE(grouped_topk_layer.Forward(input_tensors, output_tensors).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  // 验证输出
  std::vector<float> weights_host(num_tokens * topk);
  std::vector<int32_t> ids_host(num_tokens * topk);

  Memcpy(weights_host.data(), topk_weights.GetPtr<void>(), topk_weights.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);
  Memcpy(ids_host.data(), topk_ids.GetPtr<void>(), topk_ids.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);

  // 验证 topk_ids 在有效范围内
  for (int i = 0; i < num_tokens * topk; ++i) {
    EXPECT_GE(ids_host[i], 0);
    EXPECT_LT(ids_host[i], num_experts);
  }

  // 验证权重为正数（softmax 输出）
  for (int i = 0; i < num_tokens * topk; ++i) {
    EXPECT_GT(weights_host[i], 0.0f);
    EXPECT_LE(weights_host[i], 1.0f);
  }

  // 验证每个 token 的权重和接近 1.0（如果 renormalize=true）
  if (renormalize) {
    for (int token = 0; token < num_tokens; ++token) {
      float weight_sum = 0.0f;
      for (int k = 0; k < topk; ++k) {
        weight_sum += weights_host[token * topk + k];
      }
      EXPECT_NEAR(weight_sum, 1.0f, 0.01f);  // 允许小的数值误差
    }
  }

  // 验证 topk 选择的正确性 - 检查选中的专家确实是权重最大的
  for (int token = 0; token < num_tokens; ++token) {
    std::vector<std::pair<float, int>> expert_weights;
    for (int expert = 0; expert < num_experts; ++expert) {
      expert_weights.push_back({static_cast<float>(gating_host[token * num_experts + expert]), expert});
    }
    std::sort(expert_weights.rbegin(), expert_weights.rend());  // 降序排列

    // 检查选中的专家是否在前 topk 中
    std::set<int> expected_experts;
    for (int k = 0; k < topk; ++k) {
      expected_experts.insert(expert_weights[k].second);
    }

    std::set<int> actual_experts;
    for (int k = 0; k < topk; ++k) {
      actual_experts.insert(ids_host[token * topk + k]);
    }

    EXPECT_EQ(expected_experts, actual_experts);
  }

  // 测试带 e_bias 的情况
  GroupedTopkLayer grouped_topk_layer_with_bias;
  GroupedTopkLayerParameters params_with_bias{.topk = topk,
                                              .renormalize = renormalize,
                                              .num_expert_group = num_expert_group,
                                              .topk_group = topk_group,
                                              .scoring_func = scoring_func,
                                              .routed_scaling_factor = routed_scaling_factor,
                                              .use_e_score_correction_bias = true,
                                              .score_threshold = 0.f};

  EXPECT_TRUE(grouped_topk_layer_with_bias.Init({params_with_bias}, runtime_config, context_, kDeviceRank).OK());

  // 准备 e_bias 张量
  Tensor e_bias(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {static_cast<size_t>(num_experts)});
  std::vector<float> e_bias_host(num_experts);
  for (int i = 0; i < num_experts; ++i) {
    e_bias_host[i] = 0.1f * i;  // 简单的偏置值
  }
  MemcpyAsync(e_bias.GetPtr<void>(), e_bias_host.data(), e_bias.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());

  std::vector<Tensor> input_tensors_with_bias = {gating_output, e_bias};
  std::vector<Tensor> output_tensors_with_bias = {topk_weights, topk_ids};

  StreamSynchronize(context_->device->GetMemoryManageStream());

  // 测试带偏置的前向传播
  EXPECT_TRUE(grouped_topk_layer_with_bias.Forward(input_tensors_with_bias, output_tensors_with_bias).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  // 测试profile模式下输出固定的专家
  GroupedTopkLayer grouped_topk_layer_profile_mode;
  GroupedTopkLayerParameters params_profile{.topk = topk,
                                            .renormalize = renormalize,
                                            .num_expert_group = num_expert_group,
                                            .topk_group = topk_group,
                                            .scoring_func = scoring_func,
                                            .routed_scaling_factor = routed_scaling_factor,
                                            .use_e_score_correction_bias = true,
                                            .score_threshold = 0.f};
  runtime_config.is_profile_mode = true;
  EXPECT_TRUE(grouped_topk_layer_profile_mode.Init({params_profile}, runtime_config, context_, kDeviceRank).OK());
  EXPECT_TRUE(grouped_topk_layer_profile_mode.Forward(input_tensors, output_tensors).OK());
  Memcpy(ids_host.data(), topk_ids.GetPtr<void>(), topk_ids.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);

  // 验证 topk_ids 的值是固定的根据seed选出来的专家
  EXPECT_TRUE(ids_host[0] == 3 && ids_host[1] == 1 && ids_host[2] == 6 && ids_host[3] == 2);
  // set is_profile_mode back to false
  runtime_config.is_profile_mode = false;
#endif
}

TEST_F(LayerTest, MoeLayerTest) {
#ifdef ENABLE_CUDA
  if (context_->device->ext->GetComputeCapacity() != 90) {
    return;
  }
  constexpr int kDeviceRank = 0;
  using dtype = __nv_bfloat16;
  using alpha_dtype = float;
  setenv("QUANT_PROFILE", "0", 1);

  ModelConfig new_model_config = model_config;
  RuntimeConfig new_runtime_config = runtime_config;
  new_model_config.weight_data_type = TYPE_BF16;
  new_runtime_config.inter_data_type = new_model_config.weight_data_type;

  torch::manual_seed(42);

  // params
  MoeScaleNormMode moe_scale_norm_mode = MoeScaleNormMode::NO_NORM;  // 用不到
  size_t max_token_num = 4096;
  int layer_idx = 10;
  size_t expert_num = 32;
  size_t expert_hidden_size = 7168;
  size_t expert_inter_size = 2048;
  size_t expert_topk = 8;
  size_t tp_size = 1;
  bool use_vllm_moe = true;
  uint32_t num_expert_group = 8;
  uint32_t expert_groups_topk = 4;
  std::string scoring_func = "sigmoid";
  std::string topk_method = "";  // 用不到
  bool norm_topk_prob = true;
  float routed_scaling_factor = 2.5f;
  bool use_e_score_correction_bias = false;  // 关闭方便测试
  bool enable_full_shared_expert = false;
  DataType fp8_weight_dtype = DataType::TYPE_INVALID;
  DataType int_weight_dtype = DataType::TYPE_UINT4x2;
  int group_size = 128;
  bool apply_weight = false;  // 用不到

  std::vector<std::any> params;
  params.push_back(moe_scale_norm_mode);
  params.push_back(max_token_num);
  params.push_back(layer_idx);
  params.push_back(expert_num);
  params.push_back(expert_hidden_size);
  params.push_back(expert_inter_size);
  params.push_back(expert_topk);
  params.push_back(tp_size);
  params.push_back(use_vllm_moe);
  params.push_back(num_expert_group);
  params.push_back(expert_groups_topk);
  params.push_back(scoring_func);
  params.push_back(topk_method);
  params.push_back(norm_topk_prob);
  params.push_back(routed_scaling_factor);
  params.push_back(use_e_score_correction_bias);
  params.push_back(enable_full_shared_expert);
  params.push_back(fp8_weight_dtype);
  params.push_back(int_weight_dtype);
  params.push_back(group_size);
  params.push_back(apply_weight);

  auto int8_option = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA);
  auto int32_option = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto float_option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto dtype_option = torch::TensorOptions().dtype(GetTorchDataType<dtype>()).device(torch::kCUDA);
  auto alpha_dtype_option = torch::TensorOptions().dtype(GetTorchDataType<alpha_dtype>()).device(torch::kCUDA);

  auto updateInterleave = [](size_t hidden_size, size_t intermediate_size_per_partition) {
    std::vector<size_t> interleave;
    std::vector<size_t> k_shapes = {hidden_size, intermediate_size_per_partition};
    for (const size_t& k_shape : k_shapes) {
      if (k_shape % 512 == 0) {
        interleave.push_back(4);
      } else if (k_shape % 256 == 0) {
        interleave.push_back(2);
      } else if (k_shape % 128 == 0) {
        interleave.push_back(1);
      } else {
        throw std::runtime_error(fmt::format("K shape is required to be multiple of 128, received {}.", k_shape));
      }
    }
    return interleave;
  };

  // NOTE(jinxcwu) 这里-8是uint4转int4。权重都是uint4的，但moe_layer那边需要的是int4，为了对齐需要额外实现
  auto unpack_int4_packed_tensor_to_int8 = [](torch::Tensor weight) {
    std::vector<int64_t> int8_tensor_size(weight.dim());
    for (int i = 0; i < weight.dim(); ++i) {
      int8_tensor_size[i] = weight.size(i);
    }
    int8_tensor_size[weight.dim() - 1] *= 2;

    torch::Tensor unpacked_weight =
        torch::zeros(int8_tensor_size, torch::dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));

    int8_t* packed_ptr = static_cast<int8_t*>(weight.data_ptr());
    int8_t* unpacked_ptr = static_cast<int8_t*>(unpacked_weight.data_ptr());

    for (int64_t packed_idx = 0; packed_idx < weight.numel(); ++packed_idx) {
      int8_t packed_data = packed_ptr[packed_idx];

      int8_t elt_0 = (int8_t(packed_data << 4) >> 4);  // The double shift here is to ensure sign extension
      int8_t elt_1 = packed_data >> 4;

      unpacked_ptr[2 * packed_idx + 0] = elt_0;
      unpacked_ptr[2 * packed_idx + 1] = elt_1;
    }

    return unpacked_weight;
  };

  auto ConvertPackUint4ToPackInt4 = [](const torch::Tensor& qweight) {
    auto origin_shape = qweight.sizes();
    auto weight = qweight.view(torch::kUInt8).view({-1});
    auto low = torch::bitwise_and(((weight % 16).to(torch::kInt8) - 8), 0x0F);
    auto high = torch::bitwise_and(((weight / 16).to(torch::kInt8) - 8), 0x0F);
    weight = torch::bitwise_or(torch::bitwise_left_shift(high, 4), low);
    weight = weight.view(origin_shape);
    return weight;
  };

  size_t num_tokens = 4;
  // 创建输入
  torch::Tensor hidden_states =
      torch::randn({static_cast<int64_t>(num_tokens), static_cast<int64_t>(expert_hidden_size)}, dtype_option);
  torch::Tensor routing_out =
      torch::randn({static_cast<int64_t>(num_tokens), static_cast<int64_t>(expert_num)}, dtype_option);
  // 创建输出
  torch::Tensor cutlass_output =
      torch::zeros({static_cast<int64_t>(num_tokens), static_cast<int64_t>(expert_hidden_size)}, dtype_option);
  torch::Tensor triton_output =
      torch::zeros({static_cast<int64_t>(num_tokens), static_cast<int64_t>(expert_hidden_size)}, dtype_option);

  // 创建权重
  float affine_coeff = 0.005f;
  std::map<std::string, torch::Tensor> weights;
  for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
    weights[fmt::format("{}.w1.weight", expert_id)] = torch::randint(
        -128, 127, {static_cast<int64_t>(expert_inter_size), static_cast<int64_t>(expert_hidden_size / 2)},
        int8_option);
    weights[fmt::format("{}.w2.weight", expert_id)] = torch::randint(
        -128, 127, {static_cast<int64_t>(expert_hidden_size), static_cast<int64_t>(expert_inter_size / 2)},
        int8_option);
    weights[fmt::format("{}.w3.weight", expert_id)] = torch::randint(
        -128, 127, {static_cast<int64_t>(expert_inter_size), static_cast<int64_t>(expert_hidden_size / 2)},
        int8_option);

    weights[fmt::format("{}.w1.weight_scale_inv", expert_id)] =
        torch::randn({static_cast<int64_t>(expert_inter_size), static_cast<int64_t>(expert_hidden_size / group_size)},
                     dtype_option) *
        affine_coeff;
    weights[fmt::format("{}.w2.weight_scale_inv", expert_id)] =
        torch::randn({static_cast<int64_t>(expert_hidden_size), static_cast<int64_t>(expert_inter_size / group_size)},
                     dtype_option) *
        affine_coeff;
    weights[fmt::format("{}.w3.weight_scale_inv", expert_id)] =
        torch::randn({static_cast<int64_t>(expert_inter_size), static_cast<int64_t>(expert_hidden_size / group_size)},
                     dtype_option) *
        affine_coeff;

    // NOTE(jinxcwu) input_scale用1结果会更稳定
    // torch::Tensor input_scale = torch::randn({static_cast<int64_t>(1)}, float_option) * 0.02;
    torch::Tensor input_scale = torch::ones({static_cast<int64_t>(1)}, float_option);
    weights[fmt::format("{}.w1.input_scale", expert_id)] = input_scale.clone();
    weights[fmt::format("{}.w2.input_scale", expert_id)] = input_scale.clone();
    weights[fmt::format("{}.w3.input_scale", expert_id)] = input_scale.clone();
  }

  {
    std::vector<size_t> interleave = updateInterleave(expert_hidden_size, expert_inter_size);

    // 权重加载
    // 步骤1: 创建一份空权重
    torch::Tensor w1_w3_weight =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_inter_size * 2),
                      static_cast<int64_t>(expert_hidden_size / 2)},
                     int8_option);
    torch::Tensor w2_weight = torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_hidden_size),
                                            static_cast<int64_t>(expert_inter_size / 2)},
                                           int8_option);
    torch::Tensor fc13_act_scale = torch::empty({static_cast<int64_t>(1)}, dtype_option);
    torch::Tensor fc2_act_scale = torch::empty({static_cast<int64_t>(1)}, dtype_option);
    torch::Tensor fc13_weight_scale = torch::empty(
        {static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_hidden_size / (128 * interleave[0])),
         static_cast<int64_t>(expert_inter_size * 2 * interleave[0])},
        dtype_option);
    torch::Tensor fc2_weight_scale =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_inter_size / (128 * interleave[1])),
                      static_cast<int64_t>(expert_hidden_size * interleave[1])},
                     dtype_option);
    torch::Tensor fc13_alpha =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(1)}, alpha_dtype_option);
    torch::Tensor fc2_alpha =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(1)}, alpha_dtype_option);
    // 步骤2 处理weight
    for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
      w1_w3_weight[expert_id].copy_(ConvertPackUint4ToPackInt4(torch::cat(
          {weights[fmt::format("{}.w1.weight", expert_id)], weights[fmt::format("{}.w3.weight", expert_id)]}, 0)));
      w2_weight[expert_id].copy_(ConvertPackUint4ToPackInt4(weights[fmt::format("{}.w2.weight", expert_id)]));
    }
    // 步骤3 处理act scale和alpha
    float all_w2_input_scales_max = std::numeric_limits<float>::min();
    for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
      all_w2_input_scales_max = std::max(
          all_w2_input_scales_max, torch::max(weights[fmt::format("{}.w2.input_scale", expert_id)]).item<float>());
    }
    fc2_act_scale.copy_((torch::ones_like(fc2_act_scale) * (1 / all_w2_input_scales_max)).to(fc2_act_scale.dtype()));
    fc2_alpha.copy_((torch::ones_like(fc2_alpha) * all_w2_input_scales_max).to(fc2_alpha.dtype()));
    // 步骤4 处理act scale和alpha
    float all_w1_w3_input_scales_max = std::numeric_limits<float>::min();
    for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
      all_w1_w3_input_scales_max = std::max(
          all_w1_w3_input_scales_max, torch::max(weights[fmt::format("{}.w1.input_scale", expert_id)]).item<float>());
      all_w1_w3_input_scales_max = std::max(
          all_w1_w3_input_scales_max, torch::max(weights[fmt::format("{}.w3.input_scale", expert_id)]).item<float>());
    }
    fc13_act_scale.copy_(
        (torch::ones_like(fc13_act_scale) * (1 / all_w1_w3_input_scales_max)).to(fc13_act_scale.dtype()));
    fc13_alpha.copy_((torch::ones_like(fc13_alpha) * all_w1_w3_input_scales_max).to(fc13_alpha.dtype()));
    // 步骤5
    std::vector<torch::Tensor> all_w2_scales;
    for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
      all_w2_scales.push_back(weights[fmt::format("{}.w2.weight_scale_inv", expert_id)]);
    }
    torch::Tensor w2_scales = torch::stack(all_w2_scales).to(torch::kBFloat16).view(GetTorchDataType<dtype>());
    auto w2_s_shape = w2_scales.sizes();
    torch::Tensor w2_scales_interleaved =
        w2_scales.reshape({w2_s_shape[0], w2_s_shape[1], static_cast<int64_t>(w2_s_shape[2] / interleave[1]),
                           static_cast<int64_t>(interleave[1])});
    w2_scales_interleaved = w2_scales_interleaved.permute({0, 2, 1, 3});
    w2_scales_interleaved =
        w2_scales_interleaved.reshape({w2_s_shape[0], static_cast<int64_t>(w2_s_shape[2] / interleave[1]),
                                       static_cast<int64_t>(w2_s_shape[1] * interleave[1])});
    fc2_weight_scale.copy_(w2_scales_interleaved.contiguous());
    // 步骤6
    std::vector<torch::Tensor> all_w1_scales;
    std::vector<torch::Tensor> all_w3_scales;
    for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
      all_w1_scales.push_back(weights[fmt::format("{}.w1.weight_scale_inv", expert_id)]);
      all_w3_scales.push_back(weights[fmt::format("{}.w3.weight_scale_inv", expert_id)]);
    }
    torch::Tensor all_w1_scales_stack = torch::stack(all_w1_scales);
    torch::Tensor all_w3_scales_stack = torch::stack(all_w3_scales);
    torch::Tensor all_w1_w3_scales = torch::cat({all_w1_scales_stack, all_w3_scales_stack}, -2);
    torch::Tensor w1_w3_scales = all_w1_w3_scales.to(torch::kBFloat16).view(GetTorchDataType<dtype>());
    auto w1_w3_s_shape = w1_w3_scales.sizes();
    torch::Tensor w1_w3_scales_interleaved = w1_w3_scales.reshape(
        {w1_w3_s_shape[0], w1_w3_s_shape[1], static_cast<int64_t>(w1_w3_s_shape[2] / interleave[0]),
         static_cast<int64_t>(interleave[0])});
    w1_w3_scales_interleaved = w1_w3_scales_interleaved.permute({0, 2, 1, 3});
    w1_w3_scales_interleaved =
        w1_w3_scales_interleaved.reshape({w1_w3_s_shape[0], static_cast<int64_t>(w1_w3_s_shape[2] / interleave[0]),
                                          static_cast<int64_t>(w1_w3_s_shape[1] * interleave[0])});
    fc13_weight_scale.copy_(w1_w3_scales_interleaved.contiguous());
    // 创建 cutlass moe layer
    std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
    CutlassMoeLayer cutlass_moe_layer = CutlassMoeLayer();
    cutlass_moe_layer.Init(params, new_runtime_config, context_, kDeviceRank);
    size_t workspace_size = cutlass_moe_layer.GetWorkspaceSize();
    KLLM_LOG_INFO << fmt::format("CutlassMoeLayer WorkSpaceSize: {}", workspace_size);
    cutlass_moe_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(workspace_size));
    cutlass_moe_layer.Preprocess(new_model_config, new_runtime_config);
    // 构建输入 input_tensors: 0.hidden states 1.routing_out 2.up_gate_experts 3.down_experts 4.bias
    std::vector<Tensor> inputs(4);
    inputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(hidden_states.size(0)), static_cast<size_t>(hidden_states.size(1))},
                       hidden_states.data_ptr());
    inputs[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(routing_out.size(0)), static_cast<size_t>(routing_out.size(1))},
                       routing_out.data_ptr());
    inputs[2] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8,
                       {static_cast<size_t>(w1_w3_weight.size(0)), static_cast<size_t>(w1_w3_weight.size(1)),
                        static_cast<size_t>(w1_w3_weight.size(2))},
                       w1_w3_weight.data_ptr());
    inputs[3] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8,
                       {static_cast<size_t>(w2_weight.size(0)), static_cast<size_t>(w2_weight.size(1)),
                        static_cast<size_t>(w2_weight.size(2))},
                       w2_weight.data_ptr());
    std::vector<Tensor> scales(2);
    scales[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(fc13_weight_scale.size(0)), static_cast<size_t>(fc13_weight_scale.size(1)),
                        static_cast<size_t>(fc13_weight_scale.size(2))},
                       fc13_weight_scale.data_ptr());
    scales[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(fc2_weight_scale.size(0)), static_cast<size_t>(fc2_weight_scale.size(1)),
                        static_cast<size_t>(fc2_weight_scale.size(2))},
                       fc2_weight_scale.data_ptr());
    inputs[2].scales = &scales[0];
    inputs[3].scales = &scales[1];
    std::vector<Tensor> input_scales(2);
    input_scales[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {1}, fc13_act_scale.data_ptr());
    input_scales[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {1}, fc2_act_scale.data_ptr());
    inputs[2].input_scales = &input_scales[0];
    inputs[3].input_scales = &input_scales[1];
    std::vector<Tensor> alpha(2);
    alpha[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32,
                      {static_cast<size_t>(fc13_alpha.size(0)), static_cast<size_t>(fc13_alpha.size(1))},
                      fc13_alpha.data_ptr());
    alpha[1] =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32,
               {static_cast<size_t>(fc2_alpha.size(0)), static_cast<size_t>(fc2_alpha.size(1))}, fc2_alpha.data_ptr());
    inputs[2].input_alpha = &alpha[0];
    inputs[3].input_alpha = &alpha[1];
    // 构建输出
    std::vector<Tensor> outputs(1);
    outputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                        {static_cast<size_t>(cutlass_output.size(0)), static_cast<size_t>(cutlass_output.size(1))},
                        cutlass_output.data_ptr());
    // 推理
    EXPECT_TRUE(cutlass_moe_layer.Forward(inputs, outputs).OK());
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
  }

  {
    // 权重加载
    // 步骤1: 创建一份空权重
    torch::Tensor up_gate_weight =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_inter_size * 2),
                      static_cast<int64_t>(expert_hidden_size / 2)},
                     int8_option);
    torch::Tensor down_weight =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_hidden_size),
                      static_cast<int64_t>(expert_inter_size / 2)},
                     int8_option);
    torch::Tensor up_gate_weight_scale =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_inter_size * 2),
                      static_cast<int64_t>(expert_hidden_size / 128)},
                     dtype_option);
    torch::Tensor down_weight_scale =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_hidden_size),
                      static_cast<int64_t>(expert_inter_size / 128)},
                     dtype_option);
    // 步骤2 处理weight
    for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
      up_gate_weight[expert_id].copy_(torch::cat(
          {weights[fmt::format("{}.w1.weight", expert_id)], weights[fmt::format("{}.w3.weight", expert_id)]}, 0));
      down_weight[expert_id].copy_(weights[fmt::format("{}.w2.weight", expert_id)]);
    }
    // 步骤3 处理scale
    for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
      up_gate_weight_scale[expert_id].copy_(torch::cat({weights[fmt::format("{}.w1.weight_scale_inv", expert_id)],
                                                        weights[fmt::format("{}.w3.weight_scale_inv", expert_id)]},
                                                       0));
      down_weight_scale[expert_id].copy_(weights[fmt::format("{}.w2.weight_scale_inv", expert_id)]);
    }
    // 创建 triton moe layer
    std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
    MoeLayer moe_layer = MoeLayer();
    moe_layer.Init(params, new_runtime_config, context_, kDeviceRank);
    size_t workspace_size = moe_layer.GetWorkspaceSize();
    KLLM_LOG_INFO << fmt::format("MoeLayer WorkSpaceSize: {}", workspace_size);
    Tensor workspace_buffer = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8, {workspace_size});
    std::shared_ptr<Tensor> workspace_buffer_ptr = std::make_shared<Tensor>(workspace_buffer);
    moe_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(workspace_size));
    moe_layer.Preprocess(new_model_config, new_runtime_config);
    // 构建输入 input_tensors: 0.hidden states 1.routing_out 2.up_gate_experts 3.down_experts 4.bias
    std::vector<Tensor> inputs(4);
    inputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(hidden_states.size(0)), static_cast<size_t>(hidden_states.size(1))},
                       hidden_states.data_ptr());
    inputs[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(routing_out.size(0)), static_cast<size_t>(routing_out.size(1))},
                       routing_out.data_ptr());
    inputs[2] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8,
                       {static_cast<size_t>(up_gate_weight.size(0)), static_cast<size_t>(up_gate_weight.size(1)),
                        static_cast<size_t>(up_gate_weight.size(2))},
                       up_gate_weight.data_ptr());
    inputs[3] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8,
                       {static_cast<size_t>(down_weight.size(0)), static_cast<size_t>(down_weight.size(1)),
                        static_cast<size_t>(down_weight.size(2))},
                       down_weight.data_ptr());
    std::vector<Tensor> scales(2);
    scales[0] =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
               {static_cast<size_t>(up_gate_weight_scale.size(0)), static_cast<size_t>(up_gate_weight_scale.size(1)),
                static_cast<size_t>(up_gate_weight_scale.size(2))},
               up_gate_weight_scale.data_ptr());
    scales[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(down_weight_scale.size(0)), static_cast<size_t>(down_weight_scale.size(1)),
                        static_cast<size_t>(down_weight_scale.size(2))},
                       down_weight_scale.data_ptr());
    inputs[2].scales = &scales[0];
    inputs[3].scales = &scales[1];
    // 构建输出
    std::vector<Tensor> outputs(1);
    outputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                        {static_cast<size_t>(triton_output.size(0)), static_cast<size_t>(triton_output.size(1))},
                        triton_output.data_ptr());
    // 推理
    EXPECT_TRUE(moe_layer.Forward(inputs, outputs).OK());
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
  }

  torch::Tensor torch_output = torch::zeros_like(hidden_states);
  {
    GroupedTopkLayer grouped_topk_layer;
    GroupedTopkLayerParameters params{.topk = expert_topk,
                                      .renormalize = norm_topk_prob,
                                      .num_expert_group = num_expert_group,
                                      .topk_group = expert_groups_topk,
                                      .scoring_func = scoring_func,
                                      .routed_scaling_factor = routed_scaling_factor,
                                      .use_e_score_correction_bias = use_e_score_correction_bias,
                                      .score_threshold = 0.f};
    EXPECT_TRUE(grouped_topk_layer.Init({params}, runtime_config, context_, kDeviceRank).OK());
    torch::Tensor topk_ids =
        torch::zeros({static_cast<int64_t>(num_tokens), static_cast<int64_t>(expert_topk)}, int32_option);
    torch::Tensor topk_weights =
        torch::zeros({static_cast<int64_t>(num_tokens), static_cast<int64_t>(expert_topk)}, float_option);
    // 构建输入
    std::vector<Tensor> inputs(1);
    inputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(routing_out.size(0)), static_cast<size_t>(routing_out.size(1))},
                       routing_out.data_ptr());
    // 构建输出
    std::vector<Tensor> outputs(2);
    outputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32,
                        {static_cast<size_t>(topk_weights.size(0)), static_cast<size_t>(topk_weights.size(1))},
                        topk_weights.data_ptr());
    outputs[1] =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
               {static_cast<size_t>(topk_ids.size(0)), static_cast<size_t>(topk_ids.size(1))}, topk_ids.data_ptr());
    // 推理
    EXPECT_TRUE(grouped_topk_layer.Forward(inputs, outputs).OK());
    StreamSynchronize(context_->device->GetPrimaryComputeStream());
    // 计算参考结果
    torch::Tensor selected_experts = topk_ids.clone();
    torch::Tensor final_scales = topk_weights.clone();
    for (size_t e_idx = 0; e_idx < expert_num; e_idx++) {
      torch::Tensor mask = selected_experts == e_idx;
      torch::Tensor activated_tokens = mask.sum(1).to(torch::kBool);
      torch::Tensor act = hidden_states.index_select(0, activated_tokens.nonzero().squeeze());
      if (act.size(0) == 0) {
        continue;
      }
      torch::Tensor final_scale = (final_scales * mask.to(torch::kFloat))
                                      .sum(1)
                                      .index_select(0, activated_tokens.nonzero().squeeze())
                                      .unsqueeze(1);

      torch::Tensor w1 = weights[fmt::format("{}.w1.weight", e_idx)];
      w1 = unpack_int4_packed_tensor_to_int8(ConvertPackUint4ToPackInt4(w1).cpu()).t().contiguous().cuda();
      torch::Tensor w2 = weights[fmt::format("{}.w2.weight", e_idx)];
      w2 = unpack_int4_packed_tensor_to_int8(ConvertPackUint4ToPackInt4(w2).cpu()).t().contiguous().cuda();
      torch::Tensor w3 = weights[fmt::format("{}.w3.weight", e_idx)];
      w3 = unpack_int4_packed_tensor_to_int8(ConvertPackUint4ToPackInt4(w3).cpu()).t().contiguous().cuda();
      torch::Tensor w1_w3 = torch::cat({w1, w3}, -1);

      torch::Tensor s1 = weights[fmt::format("{}.w1.weight_scale_inv", e_idx)].t().contiguous();
      torch::Tensor s2 = weights[fmt::format("{}.w2.weight_scale_inv", e_idx)].t().contiguous();
      torch::Tensor s3 = weights[fmt::format("{}.w3.weight_scale_inv", e_idx)].t().contiguous();
      torch::Tensor s1_s3 = torch::cat({s1, s3}, -1);

      torch::Tensor p1 = weights[fmt::format("{}.w1.input_scale", e_idx)];
      torch::Tensor p2 = weights[fmt::format("{}.w2.input_scale", e_idx)];
      torch::Tensor p3 = weights[fmt::format("{}.w3.input_scale", e_idx)];
      torch::Tensor p1_p3 = torch::max(p1, p3);

      act = torch::clamp((act / p1_p3), -448.0, 448.0).to(torch::kFloat8_e4m3fn).to(GetTorchDataType<dtype>());
      w1_w3 =
          (w1_w3.to(torch::kFloat) * s1_s3.repeat_interleave(128, 0).to(torch::kFloat)).to(GetTorchDataType<dtype>());
      torch::Tensor fc1_gate = torch::matmul(act, w1_w3) * p1_p3;
      auto chunks = fc1_gate.chunk(2, -1);
      torch::Tensor gate = chunks[0];
      torch::Tensor fc1 = chunks[1];
      fc1 = fc1 * torch::nn::functional::silu(gate);

      act = torch::clamp((fc1 / p2), -448.0, 448.0).to(torch::kFloat8_e4m3fn).to(GetTorchDataType<dtype>());
      w2 = (w2.to(torch::kFloat) * s2.repeat_interleave(128, 0).to(torch::kFloat)).to(GetTorchDataType<dtype>());
      torch::Tensor fc2 = torch::matmul(act, w2) * p2;
      torch_output.index_add_(0, activated_tokens.nonzero().squeeze(), (fc2 * final_scale).to(torch_output.dtype()));
    }
  }

  unsetenv("QUANT_PROFILE");
#endif
}

TEST_F(LayerTest, CutlassMoeLayerOnlyTest) {
#ifdef ENABLE_CUDA
  if (context_->device->ext->GetComputeCapacity() != 90) {
    return;
  }
  constexpr int kDeviceRank = 0;
  using dtype = __nv_bfloat16;
  using alpha_dtype = float;
  setenv("QUANT_PROFILE", "3", 1);

  ModelConfig new_model_config = model_config;
  RuntimeConfig new_runtime_config = runtime_config;
  new_model_config.weight_data_type = TYPE_BF16;
  new_runtime_config.inter_data_type = new_model_config.weight_data_type;

  torch::manual_seed(666);

  // params
  MoeScaleNormMode moe_scale_norm_mode = MoeScaleNormMode::NO_NORM;  // 用不到
  size_t max_token_num = 65536;
  int layer_idx = 10;
  size_t expert_num = 32;
  size_t expert_hidden_size = 7168;
  size_t expert_inter_size = 2048;
  size_t expert_topk = 8;
  size_t tp_size = 1;
  bool use_vllm_moe = true;
  uint32_t num_expert_group = 8;
  uint32_t expert_groups_topk = 4;
  std::string scoring_func = "sigmoid";
  std::string topk_method = "";  // 用不到
  bool norm_topk_prob = true;
  float routed_scaling_factor = 2.5f;
  bool use_e_score_correction_bias = false;  // 关闭方便测试
  bool enable_full_shared_expert = false;
  DataType fp8_weight_dtype = DataType::TYPE_INVALID;
  DataType int_weight_dtype = DataType::TYPE_UINT4x2;
  int group_size = 128;
  bool apply_weight = false;  // 用不到

  std::vector<std::any> params;
  params.push_back(moe_scale_norm_mode);
  params.push_back(max_token_num);
  params.push_back(layer_idx);
  params.push_back(expert_num);
  params.push_back(expert_hidden_size);
  params.push_back(expert_inter_size);
  params.push_back(expert_topk);
  params.push_back(tp_size);
  params.push_back(use_vllm_moe);
  params.push_back(num_expert_group);
  params.push_back(expert_groups_topk);
  params.push_back(scoring_func);
  params.push_back(topk_method);
  params.push_back(norm_topk_prob);
  params.push_back(routed_scaling_factor);
  params.push_back(use_e_score_correction_bias);
  params.push_back(enable_full_shared_expert);
  params.push_back(fp8_weight_dtype);
  params.push_back(int_weight_dtype);
  params.push_back(group_size);
  params.push_back(apply_weight);

  auto int8_option = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA);
  auto int32_option = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto float_option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto dtype_option = torch::TensorOptions().dtype(GetTorchDataType<dtype>()).device(torch::kCUDA);
  auto alpha_dtype_option = torch::TensorOptions().dtype(GetTorchDataType<alpha_dtype>()).device(torch::kCUDA);

  auto updateInterleave = [](size_t hidden_size, size_t intermediate_size_per_partition) {
    std::vector<size_t> interleave;
    std::vector<size_t> k_shapes = {hidden_size, intermediate_size_per_partition};
    for (const size_t& k_shape : k_shapes) {
      if (k_shape % 512 == 0) {
        interleave.push_back(4);
      } else if (k_shape % 256 == 0) {
        interleave.push_back(2);
      } else if (k_shape % 128 == 0) {
        interleave.push_back(1);
      } else {
        throw std::runtime_error(fmt::format("K shape is required to be multiple of 128, received {}.", k_shape));
      }
    }
    return interleave;
  };

  // NOTE(jinxcwu) 这里-8是uint4转int4。权重都是uint4的，但moe_layer那边需要的是int4，为了对齐需要额外实现
  auto unpack_int4_packed_tensor_to_int8 = [](torch::Tensor weight) {
    std::vector<int64_t> int8_tensor_size(weight.dim());
    for (int i = 0; i < weight.dim(); ++i) {
      int8_tensor_size[i] = weight.size(i);
    }
    int8_tensor_size[weight.dim() - 1] *= 2;

    torch::Tensor unpacked_weight =
        torch::zeros(int8_tensor_size, torch::dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));

    int8_t* packed_ptr = static_cast<int8_t*>(weight.data_ptr());
    int8_t* unpacked_ptr = static_cast<int8_t*>(unpacked_weight.data_ptr());

    for (int64_t packed_idx = 0; packed_idx < weight.numel(); ++packed_idx) {
      int8_t packed_data = packed_ptr[packed_idx];

      int8_t elt_0 = (int8_t(packed_data << 4) >> 4);  // The double shift here is to ensure sign extension
      int8_t elt_1 = packed_data >> 4;

      unpacked_ptr[2 * packed_idx + 0] = elt_0;
      unpacked_ptr[2 * packed_idx + 1] = elt_1;
    }

    return unpacked_weight;
  };

  auto ConvertPackUint4ToPackInt4 = [](const torch::Tensor& qweight) {
    auto origin_shape = qweight.sizes();
    auto weight = qweight.view(torch::kUInt8).view({-1});
    auto low = torch::bitwise_and(((weight % 16).to(torch::kInt8) - 8), 0x0F);
    auto high = torch::bitwise_and(((weight / 16).to(torch::kInt8) - 8), 0x0F);
    weight = torch::bitwise_or(torch::bitwise_left_shift(high, 4), low);
    weight = weight.view(origin_shape);
    return weight;
  };

  // 创建权重（权重只需要创建一次，与 num_tokens 无关）
  float affine_coeff = 0.005f;
  std::map<std::string, torch::Tensor> weights;
  for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
    weights[fmt::format("{}.w1.weight", expert_id)] = torch::randint(
        -128, 127, {static_cast<int64_t>(expert_inter_size), static_cast<int64_t>(expert_hidden_size / 2)},
        int8_option);
    weights[fmt::format("{}.w2.weight", expert_id)] = torch::randint(
        -128, 127, {static_cast<int64_t>(expert_hidden_size), static_cast<int64_t>(expert_inter_size / 2)},
        int8_option);
    weights[fmt::format("{}.w3.weight", expert_id)] = torch::randint(
        -128, 127, {static_cast<int64_t>(expert_inter_size), static_cast<int64_t>(expert_hidden_size / 2)},
        int8_option);

    weights[fmt::format("{}.w1.weight_scale_inv", expert_id)] =
        torch::randn({static_cast<int64_t>(expert_inter_size), static_cast<int64_t>(expert_hidden_size / group_size)},
                     dtype_option) *
        affine_coeff;
    weights[fmt::format("{}.w2.weight_scale_inv", expert_id)] =
        torch::randn({static_cast<int64_t>(expert_hidden_size), static_cast<int64_t>(expert_inter_size / group_size)},
                     dtype_option) *
        affine_coeff;
    weights[fmt::format("{}.w3.weight_scale_inv", expert_id)] =
        torch::randn({static_cast<int64_t>(expert_inter_size), static_cast<int64_t>(expert_hidden_size / group_size)},
                     dtype_option) *
        affine_coeff;

    // NOTE(jinxcwu) input_scale用1结果会更稳定
    // torch::Tensor input_scale = torch::randn({static_cast<int64_t>(1)}, float_option) * 0.02;
    torch::Tensor input_scale = torch::ones({static_cast<int64_t>(1)}, float_option);
    weights[fmt::format("{}.w1.input_scale", expert_id)] = input_scale.clone();
    weights[fmt::format("{}.w2.input_scale", expert_id)] = input_scale.clone();
    weights[fmt::format("{}.w3.input_scale", expert_id)] = input_scale.clone();
  }

  {
    std::vector<size_t> interleave = updateInterleave(expert_hidden_size, expert_inter_size);

    // 权重加载
    // 步骤1: 创建一份空权重
    torch::Tensor w1_w3_weight =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_inter_size * 2),
                      static_cast<int64_t>(expert_hidden_size / 2)},
                     int8_option);
    torch::Tensor w2_weight = torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_hidden_size),
                                            static_cast<int64_t>(expert_inter_size / 2)},
                                           int8_option);
    torch::Tensor fc13_act_scale = torch::empty({static_cast<int64_t>(1)}, dtype_option);
    torch::Tensor fc2_act_scale = torch::empty({static_cast<int64_t>(1)}, dtype_option);
    torch::Tensor fc13_weight_scale = torch::empty(
        {static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_hidden_size / (128 * interleave[0])),
         static_cast<int64_t>(expert_inter_size * 2 * interleave[0])},
        dtype_option);
    torch::Tensor fc2_weight_scale =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_inter_size / (128 * interleave[1])),
                      static_cast<int64_t>(expert_hidden_size * interleave[1])},
                     dtype_option);
    torch::Tensor fc13_alpha =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(1)}, alpha_dtype_option);
    torch::Tensor fc2_alpha =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(1)}, alpha_dtype_option);
    // 步骤2 处理weight
    for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
      w1_w3_weight[expert_id].copy_(ConvertPackUint4ToPackInt4(torch::cat(
          {weights[fmt::format("{}.w1.weight", expert_id)], weights[fmt::format("{}.w3.weight", expert_id)]}, 0)));
      w2_weight[expert_id].copy_(ConvertPackUint4ToPackInt4(weights[fmt::format("{}.w2.weight", expert_id)]));
    }
    // 步骤3 处理act scale和alpha
    float all_w2_input_scales_max = std::numeric_limits<float>::min();
    for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
      all_w2_input_scales_max = std::max(
          all_w2_input_scales_max, torch::max(weights[fmt::format("{}.w2.input_scale", expert_id)]).item<float>());
    }
    fc2_act_scale.copy_((torch::ones_like(fc2_act_scale) * (1 / all_w2_input_scales_max)).to(fc2_act_scale.dtype()));
    fc2_alpha.copy_((torch::ones_like(fc2_alpha) * all_w2_input_scales_max).to(fc2_alpha.dtype()));
    // 步骤4 处理act scale和alpha
    float all_w1_w3_input_scales_max = std::numeric_limits<float>::min();
    for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
      all_w1_w3_input_scales_max = std::max(
          all_w1_w3_input_scales_max, torch::max(weights[fmt::format("{}.w1.input_scale", expert_id)]).item<float>());
      all_w1_w3_input_scales_max = std::max(
          all_w1_w3_input_scales_max, torch::max(weights[fmt::format("{}.w3.input_scale", expert_id)]).item<float>());
    }
    fc13_act_scale.copy_(
        (torch::ones_like(fc13_act_scale) * (1 / all_w1_w3_input_scales_max)).to(fc13_act_scale.dtype()));
    fc13_alpha.copy_((torch::ones_like(fc13_alpha) * all_w1_w3_input_scales_max).to(fc13_alpha.dtype()));
    // 步骤5
    std::vector<torch::Tensor> all_w2_scales;
    for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
      all_w2_scales.push_back(weights[fmt::format("{}.w2.weight_scale_inv", expert_id)]);
    }
    torch::Tensor w2_scales = torch::stack(all_w2_scales).to(torch::kBFloat16).view(GetTorchDataType<dtype>());
    auto w2_s_shape = w2_scales.sizes();
    torch::Tensor w2_scales_interleaved =
        w2_scales.reshape({w2_s_shape[0], w2_s_shape[1], static_cast<int64_t>(w2_s_shape[2] / interleave[1]),
                           static_cast<int64_t>(interleave[1])});
    w2_scales_interleaved = w2_scales_interleaved.permute({0, 2, 1, 3});
    w2_scales_interleaved =
        w2_scales_interleaved.reshape({w2_s_shape[0], static_cast<int64_t>(w2_s_shape[2] / interleave[1]),
                                       static_cast<int64_t>(w2_s_shape[1] * interleave[1])});
    fc2_weight_scale.copy_(w2_scales_interleaved.contiguous());
    // 步骤6
    std::vector<torch::Tensor> all_w1_scales;
    std::vector<torch::Tensor> all_w3_scales;
    for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
      all_w1_scales.push_back(weights[fmt::format("{}.w1.weight_scale_inv", expert_id)]);
      all_w3_scales.push_back(weights[fmt::format("{}.w3.weight_scale_inv", expert_id)]);
    }
    torch::Tensor all_w1_scales_stack = torch::stack(all_w1_scales);
    torch::Tensor all_w3_scales_stack = torch::stack(all_w3_scales);
    torch::Tensor all_w1_w3_scales = torch::cat({all_w1_scales_stack, all_w3_scales_stack}, -2);
    torch::Tensor w1_w3_scales = all_w1_w3_scales.to(torch::kBFloat16).view(GetTorchDataType<dtype>());
    auto w1_w3_s_shape = w1_w3_scales.sizes();
    torch::Tensor w1_w3_scales_interleaved = w1_w3_scales.reshape(
        {w1_w3_s_shape[0], w1_w3_s_shape[1], static_cast<int64_t>(w1_w3_s_shape[2] / interleave[0]),
         static_cast<int64_t>(interleave[0])});
    w1_w3_scales_interleaved = w1_w3_scales_interleaved.permute({0, 2, 1, 3});
    w1_w3_scales_interleaved =
        w1_w3_scales_interleaved.reshape({w1_w3_s_shape[0], static_cast<int64_t>(w1_w3_s_shape[2] / interleave[0]),
                                          static_cast<int64_t>(w1_w3_s_shape[1] * interleave[0])});
    fc13_weight_scale.copy_(w1_w3_scales_interleaved.contiguous());
    // 创建 cutlass moe layer（只需要创建一次）
    std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
    CutlassMoeLayer cutlass_moe_layer = CutlassMoeLayer();
    cutlass_moe_layer.Init(params, new_runtime_config, context_, kDeviceRank);
    size_t workspace_size = cutlass_moe_layer.GetWorkspaceSize();
    KLLM_LOG_INFO << fmt::format("CutlassMoeLayer WorkSpaceSize: {}", workspace_size);
    cutlass_moe_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(workspace_size));
    cutlass_moe_layer.Preprocess(new_model_config, new_runtime_config);

    // 构建权重相关的输入（与 num_tokens 无关，只需要创建一次）
    std::vector<Tensor> scales(2);
    scales[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(fc13_weight_scale.size(0)), static_cast<size_t>(fc13_weight_scale.size(1)),
                        static_cast<size_t>(fc13_weight_scale.size(2))},
                       fc13_weight_scale.data_ptr());
    scales[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(fc2_weight_scale.size(0)), static_cast<size_t>(fc2_weight_scale.size(1)),
                        static_cast<size_t>(fc2_weight_scale.size(2))},
                       fc2_weight_scale.data_ptr());
    std::vector<Tensor> input_scales(2);
    input_scales[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {1}, fc13_act_scale.data_ptr());
    input_scales[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {1}, fc2_act_scale.data_ptr());
    std::vector<Tensor> alpha(2);
    alpha[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32,
                      {static_cast<size_t>(fc13_alpha.size(0)), static_cast<size_t>(fc13_alpha.size(1))},
                      fc13_alpha.data_ptr());
    alpha[1] =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32,
               {static_cast<size_t>(fc2_alpha.size(0)), static_cast<size_t>(fc2_alpha.size(1))}, fc2_alpha.data_ptr());

    // 循环测试不同的 num_tokens
    std::vector<size_t> num_tokens_list = {320, 704, 1088};
    for (size_t num_tokens : num_tokens_list) {
      // 创建输入
      torch::Tensor hidden_states =
          torch::randn({static_cast<int64_t>(num_tokens), static_cast<int64_t>(expert_hidden_size)}, dtype_option);
      torch::Tensor routing_out =
          torch::randn({static_cast<int64_t>(num_tokens), static_cast<int64_t>(expert_num)}, dtype_option);
      // 创建输出
      torch::Tensor cutlass_output =
          torch::zeros({static_cast<int64_t>(num_tokens), static_cast<int64_t>(expert_hidden_size)}, dtype_option);

      // 构建输入 input_tensors: 0.hidden states 1.routing_out 2.up_gate_experts 3.down_experts
      std::vector<Tensor> inputs(4);
      inputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                         {static_cast<size_t>(hidden_states.size(0)), static_cast<size_t>(hidden_states.size(1))},
                         hidden_states.data_ptr());
      inputs[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                         {static_cast<size_t>(routing_out.size(0)), static_cast<size_t>(routing_out.size(1))},
                         routing_out.data_ptr());
      inputs[2] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8,
                         {static_cast<size_t>(w1_w3_weight.size(0)), static_cast<size_t>(w1_w3_weight.size(1)),
                          static_cast<size_t>(w1_w3_weight.size(2))},
                         w1_w3_weight.data_ptr());
      inputs[3] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8,
                         {static_cast<size_t>(w2_weight.size(0)), static_cast<size_t>(w2_weight.size(1)),
                          static_cast<size_t>(w2_weight.size(2))},
                         w2_weight.data_ptr());
      inputs[2].scales = &scales[0];
      inputs[3].scales = &scales[1];
      inputs[2].input_scales = &input_scales[0];
      inputs[3].input_scales = &input_scales[1];
      inputs[2].input_alpha = &alpha[0];
      inputs[3].input_alpha = &alpha[1];

      // 构建输出
      std::vector<Tensor> outputs(1);
      outputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                          {static_cast<size_t>(cutlass_output.size(0)), static_cast<size_t>(cutlass_output.size(1))},
                          cutlass_output.data_ptr());

      // 推理并测试性能（warmup/iterations 可按需调大以做 benchmark）
      auto forward_func = [&]() { cutlass_moe_layer.Forward(inputs, outputs); };
      float avg_time_ms = MeasureCudaExecutionTime(forward_func, context_->device->GetPrimaryComputeStream().Get(),
                                                   /*warmups=*/1, /*iterations=*/1);
    }
  }

  unsetenv("QUANT_PROFILE");
#endif
}

TEST_F(LayerTest, CutlassMoeSearchStatusTest) {
#ifdef ENABLE_CUDA
  if (context_->device->ext->GetComputeCapacity() != 90) {
    return;
  }
  constexpr int kDeviceRank = 0;

  setenv("QUANT_PROFILE", "3", 1);

  // params
  MoeScaleNormMode moe_scale_norm_mode = MoeScaleNormMode::NO_NORM;  // 用不到
  size_t max_token_num = 4096;
  int layer_idx = 10;
  size_t expert_num = 32;
  size_t expert_hidden_size = 7168;
  size_t expert_inter_size = 2048;
  size_t expert_topk = 8;
  size_t tp_size = 1;
  bool use_vllm_moe = true;
  uint32_t num_expert_group = 8;
  uint32_t expert_groups_topk = 4;
  std::string scoring_func = "sigmoid";
  std::string topk_method = "";  // 用不到
  bool norm_topk_prob = true;
  float routed_scaling_factor = 2.5f;
  bool use_e_score_correction_bias = false;  // 关闭方便测试
  bool enable_full_shared_expert = false;
  DataType fp8_weight_dtype = DataType::TYPE_INVALID;
  DataType int_weight_dtype = DataType::TYPE_UINT4x2;
  int group_size = 128;
  bool apply_weight = false;  // 用不到

  std::vector<std::any> params;
  params.push_back(moe_scale_norm_mode);
  params.push_back(max_token_num);
  params.push_back(layer_idx);
  params.push_back(expert_num);
  params.push_back(expert_hidden_size);
  params.push_back(expert_inter_size);
  params.push_back(expert_topk);
  params.push_back(tp_size);
  params.push_back(use_vllm_moe);
  params.push_back(num_expert_group);
  params.push_back(expert_groups_topk);
  params.push_back(scoring_func);
  params.push_back(topk_method);
  params.push_back(norm_topk_prob);
  params.push_back(routed_scaling_factor);
  params.push_back(use_e_score_correction_bias);
  params.push_back(enable_full_shared_expert);
  params.push_back(fp8_weight_dtype);
  params.push_back(int_weight_dtype);
  params.push_back(group_size);
  params.push_back(apply_weight);

  std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
  auto t1 = std::chrono::high_resolution_clock::now();
  Singleton<CutlassMoeSearchStatus>::GetInstance()->ClearCutlassMoeSchedule();
  {
    CutlassMoeLayer cutlass_moe_layer = CutlassMoeLayer();
    cutlass_moe_layer.Init(params, runtime_config, context_, kDeviceRank);
    cutlass_moe_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(cutlass_moe_layer.GetWorkspaceSize()));
    cutlass_moe_layer.Preprocess(model_config, runtime_config);
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  Singleton<CutlassMoeSearchStatus>::GetInstance()->ClearCutlassMoeSchedule();
  {
    constexpr size_t kNumLayers = 2;
    constexpr int kNumThreads = 2;
    auto func = [&]() {
      for (size_t layer_idx = 0; layer_idx < kNumLayers; layer_idx++) {
        CutlassMoeLayer cutlass_moe_layer = CutlassMoeLayer();
        cutlass_moe_layer.Init(params, runtime_config, context_, kDeviceRank);
        cutlass_moe_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(cutlass_moe_layer.GetWorkspaceSize()));
        cutlass_moe_layer.Preprocess(model_config, runtime_config);
      }
    };
    std::vector<std::thread> threads;
    for (int i = 0; i < kNumThreads; ++i) {
      threads.emplace_back(func);
    }
    for (auto& t : threads) {
      t.join();
    }
  }

  auto t3 = std::chrono::high_resolution_clock::now();
  auto duration12 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  auto duration23 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);

  printf("time1: %ld, time2: %ld\n", duration12.count(), duration23.count());

  // 有缓存，创建多次耗时不应该增加太多
  EXPECT_TRUE(2 * duration12.count() > duration23.count());

  unsetenv("QUANT_PROFILE");
#endif
}

}  // namespace ksana_llm
