/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#include "layers/layer_test_base.h"

#include "layers/cutlass_group_wint4afp8_moe_layer.h"
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
  int layer_idx = 0;
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
  float swiglu_limit = 0.0f;
  uint32_t num_hash_layers = 0;
  uint32_t vocab_size = 0;
  float situ_beta = 0.0f;
  float situ_linear_beta = 0.0f;

  MoeLayerParameters moe_params;
  moe_params.moe_scale_norm_mode = moe_scale_norm_mode;
  moe_params.max_token_num = max_token_num;
  moe_params.layer_idx = layer_idx;
  moe_params.expert_num_per_node = expert_num;
  moe_params.expert_hidden_size = expert_hidden_size;
  moe_params.expert_inter_size = expert_inter_size;
  moe_params.expert_topk = expert_topk;
  moe_params.tp_size = tp_size;
  moe_params.use_vllm_moe = use_vllm_moe;
  moe_params.num_expert_group = num_expert_group;
  moe_params.expert_groups_topk = expert_groups_topk;
  moe_params.scoring_func = scoring_func;
  moe_params.topk_method = topk_method;
  moe_params.norm_topk_prob = norm_topk_prob;
  moe_params.routed_scaling_factor = routed_scaling_factor;
  moe_params.use_e_score_correction_bias = use_e_score_correction_bias;
  moe_params.enable_full_shared_expert = enable_full_shared_expert;
  moe_params.fp8_weight_dtype = fp8_weight_dtype;
  moe_params.int_weight_dtype = int_weight_dtype;
  moe_params.group_size = group_size;
  moe_params.apply_weight = apply_weight;
  moe_params.swiglu_limit = swiglu_limit;
  moe_params.num_hash_layers = num_hash_layers;
  moe_params.vocab_size = vocab_size;
  moe_params.situ_beta = situ_beta;
  moe_params.situ_linear_beta = situ_linear_beta;
  std::vector<std::any> params = {moe_params};

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
  EXPECT_TRUE(moe_layer.Forward(&inputs[0], &inputs[1], &inputs[2], &inputs[3], nullptr, &outputs[0], nullptr).OK());

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
  int layer_idx = 0;
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
  float swiglu_limit = 0.0f;
  uint32_t num_hash_layers = 0;
  uint32_t vocab_size = 0;
  float situ_beta = 0.0f;
  float situ_linear_beta = 0.0f;

  MoeLayerParameters moe_params;
  moe_params.moe_scale_norm_mode = moe_scale_norm_mode;
  moe_params.max_token_num = max_token_num;
  moe_params.layer_idx = layer_idx;
  moe_params.expert_num_per_node = expert_num;
  moe_params.expert_hidden_size = expert_hidden_size;
  moe_params.expert_inter_size = expert_inter_size;
  moe_params.expert_topk = expert_topk;
  moe_params.tp_size = tp_size;
  moe_params.use_vllm_moe = use_vllm_moe;
  moe_params.num_expert_group = num_expert_group;
  moe_params.expert_groups_topk = expert_groups_topk;
  moe_params.scoring_func = scoring_func;
  moe_params.topk_method = topk_method;
  moe_params.norm_topk_prob = norm_topk_prob;
  moe_params.routed_scaling_factor = routed_scaling_factor;
  moe_params.use_e_score_correction_bias = use_e_score_correction_bias;
  moe_params.enable_full_shared_expert = enable_full_shared_expert;
  moe_params.fp8_weight_dtype = fp8_weight_dtype;
  moe_params.int_weight_dtype = int_weight_dtype;
  moe_params.group_size = group_size;
  moe_params.apply_weight = apply_weight;
  moe_params.swiglu_limit = swiglu_limit;
  moe_params.num_hash_layers = num_hash_layers;
  moe_params.vocab_size = vocab_size;
  moe_params.situ_beta = situ_beta;
  moe_params.situ_linear_beta = situ_linear_beta;
  std::vector<std::any> params = {moe_params};

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
  EXPECT_TRUE(moe_layer.Forward(&inputs[0], &inputs[1], &inputs[2], &inputs[3], nullptr, &outputs[0], nullptr).OK());

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
  float swiglu_limit = 0.0f;
  uint32_t num_hash_layers = 0;
  uint32_t vocab_size = 0;
  float situ_beta = 0.0f;
  float situ_linear_beta = 0.0f;

  MoeLayerParameters moe_params;
  moe_params.moe_scale_norm_mode = moe_scale_norm_mode;
  moe_params.max_token_num = max_token_num;
  moe_params.layer_idx = layer_idx;
  moe_params.expert_num_per_node = expert_num;
  moe_params.expert_hidden_size = expert_hidden_size;
  moe_params.expert_inter_size = expert_inter_size;
  moe_params.expert_topk = expert_topk;
  moe_params.tp_size = tp_size;
  moe_params.use_vllm_moe = use_vllm_moe;
  moe_params.num_expert_group = num_expert_group;
  moe_params.expert_groups_topk = expert_groups_topk;
  moe_params.scoring_func = scoring_func;
  moe_params.topk_method = topk_method;
  moe_params.norm_topk_prob = norm_topk_prob;
  moe_params.routed_scaling_factor = routed_scaling_factor;
  moe_params.use_e_score_correction_bias = use_e_score_correction_bias;
  moe_params.enable_full_shared_expert = enable_full_shared_expert;
  moe_params.fp8_weight_dtype = fp8_weight_dtype;
  moe_params.int_weight_dtype = int_weight_dtype;
  moe_params.group_size = group_size;
  moe_params.apply_weight = apply_weight;
  moe_params.swiglu_limit = swiglu_limit;
  moe_params.num_hash_layers = num_hash_layers;
  moe_params.vocab_size = vocab_size;
  moe_params.situ_beta = situ_beta;
  moe_params.situ_linear_beta = situ_linear_beta;
  std::vector<std::any> params = {moe_params};

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
    CutlassGroupWint4Afp8MoeLayer cutlass_group_wint4afp8_moe_layer = CutlassGroupWint4Afp8MoeLayer();
    cutlass_group_wint4afp8_moe_layer.Init(params, new_runtime_config, context_, kDeviceRank);
    size_t workspace_size = cutlass_group_wint4afp8_moe_layer.GetWorkspaceSize();
    KLLM_LOG_INFO << fmt::format("CutlassGroupWint4Afp8MoeLayer WorkSpaceSize: {}", workspace_size);
    cutlass_group_wint4afp8_moe_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(workspace_size));
    cutlass_group_wint4afp8_moe_layer.Preprocess(new_model_config, new_runtime_config);
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
    EXPECT_TRUE(cutlass_group_wint4afp8_moe_layer
                    .Forward(&inputs[0], &inputs[1], &inputs[2], &inputs[3], nullptr, &outputs[0], nullptr)
                    .OK());
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
    EXPECT_TRUE(moe_layer.Forward(&inputs[0], &inputs[1], &inputs[2], &inputs[3], nullptr, &outputs[0], nullptr).OK());
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
    EXPECT_TRUE(grouped_topk_layer.Forward(&inputs[0], &outputs[0], &outputs[1]).OK());
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

TEST_F(LayerTest, CutlassGroupWint4Afp8MoeLayerOnlyTest) {
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
  float swiglu_limit = 0.0f;
  uint32_t num_hash_layers = 0;
  uint32_t vocab_size = 0;
  float situ_beta = 0.0f;
  float situ_linear_beta = 0.0f;

  MoeLayerParameters moe_params;
  moe_params.moe_scale_norm_mode = moe_scale_norm_mode;
  moe_params.max_token_num = max_token_num;
  moe_params.layer_idx = layer_idx;
  moe_params.expert_num_per_node = expert_num;
  moe_params.expert_hidden_size = expert_hidden_size;
  moe_params.expert_inter_size = expert_inter_size;
  moe_params.expert_topk = expert_topk;
  moe_params.tp_size = tp_size;
  moe_params.use_vllm_moe = use_vllm_moe;
  moe_params.num_expert_group = num_expert_group;
  moe_params.expert_groups_topk = expert_groups_topk;
  moe_params.scoring_func = scoring_func;
  moe_params.topk_method = topk_method;
  moe_params.norm_topk_prob = norm_topk_prob;
  moe_params.routed_scaling_factor = routed_scaling_factor;
  moe_params.use_e_score_correction_bias = use_e_score_correction_bias;
  moe_params.enable_full_shared_expert = enable_full_shared_expert;
  moe_params.fp8_weight_dtype = fp8_weight_dtype;
  moe_params.int_weight_dtype = int_weight_dtype;
  moe_params.group_size = group_size;
  moe_params.apply_weight = apply_weight;
  moe_params.swiglu_limit = swiglu_limit;
  moe_params.num_hash_layers = num_hash_layers;
  moe_params.vocab_size = vocab_size;
  moe_params.situ_beta = situ_beta;
  moe_params.situ_linear_beta = situ_linear_beta;
  std::vector<std::any> params = {moe_params};

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
    CutlassGroupWint4Afp8MoeLayer cutlass_group_wint4afp8_moe_layer = CutlassGroupWint4Afp8MoeLayer();
    cutlass_group_wint4afp8_moe_layer.Init(params, new_runtime_config, context_, kDeviceRank);
    size_t workspace_size = cutlass_group_wint4afp8_moe_layer.GetWorkspaceSize();
    KLLM_LOG_INFO << fmt::format("CutlassGroupWint4Afp8MoeLayer WorkSpaceSize: {}", workspace_size);
    cutlass_group_wint4afp8_moe_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(workspace_size));
    cutlass_group_wint4afp8_moe_layer.Preprocess(new_model_config, new_runtime_config);

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
      auto forward_func = [&]() {
        cutlass_group_wint4afp8_moe_layer.Forward(&inputs[0], &inputs[1], &inputs[2], &inputs[3], nullptr, &outputs[0],
                                                  nullptr);
      };
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
  float swiglu_limit = 0.0f;
  uint32_t num_hash_layers = 0;
  uint32_t vocab_size = 0;
  float situ_beta = 0.0f;
  float situ_linear_beta = 0.0f;

  MoeLayerParameters moe_params;
  moe_params.moe_scale_norm_mode = moe_scale_norm_mode;
  moe_params.max_token_num = max_token_num;
  moe_params.layer_idx = layer_idx;
  moe_params.expert_num_per_node = expert_num;
  moe_params.expert_hidden_size = expert_hidden_size;
  moe_params.expert_inter_size = expert_inter_size;
  moe_params.expert_topk = expert_topk;
  moe_params.tp_size = tp_size;
  moe_params.use_vllm_moe = use_vllm_moe;
  moe_params.num_expert_group = num_expert_group;
  moe_params.expert_groups_topk = expert_groups_topk;
  moe_params.scoring_func = scoring_func;
  moe_params.topk_method = topk_method;
  moe_params.norm_topk_prob = norm_topk_prob;
  moe_params.routed_scaling_factor = routed_scaling_factor;
  moe_params.use_e_score_correction_bias = use_e_score_correction_bias;
  moe_params.enable_full_shared_expert = enable_full_shared_expert;
  moe_params.fp8_weight_dtype = fp8_weight_dtype;
  moe_params.int_weight_dtype = int_weight_dtype;
  moe_params.group_size = group_size;
  moe_params.apply_weight = apply_weight;
  moe_params.swiglu_limit = swiglu_limit;
  moe_params.num_hash_layers = num_hash_layers;
  moe_params.vocab_size = vocab_size;
  moe_params.situ_beta = situ_beta;
  moe_params.situ_linear_beta = situ_linear_beta;
  std::vector<std::any> params = {moe_params};

  std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
  auto t1 = std::chrono::high_resolution_clock::now();
  Singleton<CutlassMoeSearchStatus>::GetInstance()->ClearCutlassMoeSchedule();
  {
    CutlassGroupWint4Afp8MoeLayer cutlass_group_wint4afp8_moe_layer = CutlassGroupWint4Afp8MoeLayer();
    cutlass_group_wint4afp8_moe_layer.Init(params, runtime_config, context_, kDeviceRank);
    cutlass_group_wint4afp8_moe_layer.SetWorkspaceBuffer(
        workspace_mgr->GetWorkspace(cutlass_group_wint4afp8_moe_layer.GetWorkspaceSize()));
    cutlass_group_wint4afp8_moe_layer.Preprocess(model_config, runtime_config);
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  Singleton<CutlassMoeSearchStatus>::GetInstance()->ClearCutlassMoeSchedule();
  {
    constexpr size_t kNumLayers = 2;
    constexpr int kNumThreads = 2;
    auto func = [&]() {
      for (size_t layer_idx = 0; layer_idx < kNumLayers; layer_idx++) {
        CutlassGroupWint4Afp8MoeLayer cutlass_group_wint4afp8_moe_layer = CutlassGroupWint4Afp8MoeLayer();
        cutlass_group_wint4afp8_moe_layer.Init(params, runtime_config, context_, kDeviceRank);
        cutlass_group_wint4afp8_moe_layer.SetWorkspaceBuffer(
            workspace_mgr->GetWorkspace(cutlass_group_wint4afp8_moe_layer.GetWorkspaceSize()));
        cutlass_group_wint4afp8_moe_layer.Preprocess(model_config, runtime_config);
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

// --- hpc-ops fused MoE through MoeLayer (both FP8 expert formats) ---
//
// These drive MoeLayer as a deployment would once setting.moe_backend.moe_impl has resolved to hpc,
// with the weight/scale layouts the KsanaLLM weight loaders produce.
//
// Routing is made irrelevant by construction: every expert dequantizes to the *same* logical
// weights, so with norm_topk_prob (routing weights sum to 1) the output must equal a single dense
// feed-forward pass no matter which experts win. The per-tensor case still stores each expert's
// payload against a different power-of-two scale, so an off-by-one in the per-expert scale
// indexing — or a swapped gate/up half — still shows up as a large error.
namespace {

#if defined(ENABLE_CUDA) && defined(ENABLE_FP8)
constexpr float kFp8E4m3Max = 448.0f;

// MoeLayer::Init parameters, with the knobs these tests care about; everything else keeps the
// MoeLayerParameters default.
std::vector<std::any> MakeHpcMoeLayerParams(size_t expert_num, size_t hidden, size_t inter, size_t topk,
                                            DataType fp8_weight_dtype) {
  MoeLayerParameters moe_params;
  moe_params.moe_scale_norm_mode = MoeScaleNormMode::NO_NORM;
  moe_params.max_token_num = 512;
  moe_params.layer_idx = 0;
  moe_params.expert_num_per_node = expert_num;
  moe_params.expert_hidden_size = hidden;
  moe_params.expert_inter_size = inter;
  moe_params.expert_topk = topk;
  moe_params.tp_size = 1;
  moe_params.use_vllm_moe = true;  // -> [gate, up] weight layout
  moe_params.num_expert_group = 1;
  moe_params.expert_groups_topk = 1;
  moe_params.scoring_func = "softmax";
  moe_params.topk_method = "greedy";
  moe_params.norm_topk_prob = true;  // -> routing weights sum to 1
  moe_params.routed_scaling_factor = 1.0f;
  moe_params.use_e_score_correction_bias = false;
  moe_params.enable_full_shared_expert = false;
  moe_params.fp8_weight_dtype = fp8_weight_dtype;
  moe_params.int_weight_dtype = DataType::TYPE_INVALID;
  moe_params.group_size = 0;
  moe_params.apply_weight = false;
  moe_params.swiglu_limit = 0.0f;
  moe_params.num_hash_layers = 0;
  moe_params.vocab_size = 0;
  moe_params.situ_beta = 0.0f;
  moe_params.situ_linear_beta = 0.0f;
  return {moe_params};
}

// fp8 payload with magnitudes kept in [2^-3, 2^3], so that rescaling it by a power of two (up to
// 8x here) stays inside the normal e4m3 range and remains bit-exact.
torch::Tensor MakeFp8Payload(const std::vector<int64_t>& shape, const torch::TensorOptions& float_option) {
  torch::Tensor t = torch::randn(shape, float_option);
  torch::Tensor mag = torch::clamp(torch::abs(t), 0.125f, 8.0f);
  return torch::where(t < 0, -mag, mag).to(torch::kFloat8_e4m3fn).to(torch::kFloat32);
}

// Round-trip [m, n] through per-token 128-group fp8 quantization, mirroring
// InvokePerTokenGroupQuantFp8E4m3.
torch::Tensor PerTokenGroupQuantDequant(const torch::Tensor& x, int64_t group_size) {
  torch::Tensor grouped = x.reshape({x.size(0), x.size(1) / group_size, group_size});
  torch::Tensor scale =
      std::get<0>(torch::max(torch::abs(grouped), -1, /*keepdim=*/true)).clamp_min(1e-10f) / kFp8E4m3Max;
  torch::Tensor q = (grouped / scale).to(torch::kFloat8_e4m3fn).to(torch::kFloat32);
  return (q * scale).reshape(x.sizes());
}

// Mean |actual - expected| relative to mean |expected|.
float RelativeL1Error(const torch::Tensor& actual, const torch::Tensor& expected) {
  const float expected_l1 = torch::mean(torch::abs(expected)).item<float>();
  EXPECT_GT(expected_l1, 1e-3f) << "degenerate reference, the comparison would not discriminate";
  return torch::mean(torch::abs(actual - expected)).item<float>() / expected_l1;
}
#endif

}  // namespace

TEST_F(LayerTest, HpcPerTensorFp8MoeLayerTest) {
#ifdef ENABLE_CUDA
#  ifdef ENABLE_FP8
  if (context_->device->ext->GetComputeCapacity() != 90) {
    GTEST_SKIP() << "hpc-ops MoE kernels are built for sm90a only";
  }
  constexpr int kDeviceRank = 0;

  ModelConfig new_model_config = model_config;
  RuntimeConfig new_runtime_config = runtime_config;
  new_model_config.weight_data_type = TYPE_BF16;
  new_runtime_config.inter_data_type = TYPE_BF16;
  // What setting.moe_backend.moe_impl resolves to for a per-tensor FP8 MoE model on Hopper.
  new_runtime_config.moe_backend_config.moe_impl_choice = MoeBackendConfig::MoeImplChoice::AUTO;
  new_runtime_config.moe_backend_config.use_hpc_moe = true;

  torch::manual_seed(1234);

  // hpc-ops needs hidden_size and moe_inter_size to be multiples of 128.
  const size_t expert_num = 8;
  const size_t expert_hidden_size = 512;
  const size_t expert_inter_size = 256;
  const size_t expert_topk = 2;
  const int64_t num_tokens = 16;
  const int64_t hidden = static_cast<int64_t>(expert_hidden_size);
  const int64_t inter = static_cast<int64_t>(expert_inter_size);
  const int64_t num_experts = static_cast<int64_t>(expert_num);

  std::vector<std::any> params =
      MakeHpcMoeLayerParams(expert_num, expert_hidden_size, expert_inter_size, expert_topk, TYPE_FP8_E4M3);

  auto float_option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, kDeviceRank);
  auto bf16_option = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, kDeviceRank);
  auto fp8_option = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA, kDeviceRank);

  torch::Tensor hidden_states = torch::randn({num_tokens, hidden}, bf16_option) * 0.5f;
  torch::Tensor routing_out = torch::randn({num_tokens, num_experts}, bf16_option);

  // Per-expert scales: powers of two, so that re-expressing the same logical weight against a
  // different expert's scale is an exact fp8 operation (a pure exponent shift).
  constexpr int64_t kNumDistinctScales = 4;
  torch::Tensor expert_scale = torch::empty({num_experts}, float_option);
  for (int64_t e = 0; e < num_experts; ++e) {
    expert_scale[e] = std::ldexp(1.0f, static_cast<int>(e % kNumDistinctScales) - 2);  // 0.25 .. 2
  }
  const float max_expert_scale = torch::max(expert_scale).item<float>();

  torch::Tensor gate_up_payload = MakeFp8Payload({2 * inter, hidden}, float_option);
  torch::Tensor down_payload = MakeFp8Payload({hidden, inter}, float_option);

  // The logical (dequantized) weights every expert shares.
  torch::Tensor gate_up_deq = gate_up_payload * max_expert_scale;
  torch::Tensor down_deq = down_payload * max_expert_scale;

  torch::Tensor gate_up_weight = torch::empty({num_experts, 2 * inter, hidden}, fp8_option);
  torch::Tensor down_weight = torch::empty({num_experts, hidden, inter}, fp8_option);
  for (int64_t e = 0; e < num_experts; ++e) {
    const float rescale = max_expert_scale / expert_scale[e].item<float>();
    gate_up_weight[e].copy_((gate_up_payload * rescale).to(torch::kFloat8_e4m3fn));
    down_weight[e].copy_((down_payload * rescale).to(torch::kFloat8_e4m3fn));
  }
  for (int64_t e = 0; e < num_experts; ++e) {
    ASSERT_TRUE(torch::equal(gate_up_weight[e].to(torch::kFloat32) * expert_scale[e], gate_up_deq))
        << "expert " << e << " does not dequantize to the shared reference weights";
    ASSERT_TRUE(torch::equal(down_weight[e].to(torch::kFloat32) * expert_scale[e], down_deq))
        << "expert " << e << " does not dequantize to the shared reference weights";
  }

  // Reference forward, replaying the quantization steps hpc performs (fp8 activations, bf16 GEMM
  // outputs). It also fixes the down projection's static activation scale.
  const float x_input_scale = torch::max(torch::abs(hidden_states)).item<float>() / kFp8E4m3Max;
  torch::Tensor x_deq =
      (hidden_states.to(torch::kFloat32) / x_input_scale).to(torch::kFloat8_e4m3fn).to(torch::kFloat32) *
      x_input_scale;
  torch::Tensor gate_up = torch::matmul(x_deq, gate_up_deq.t()).to(torch::kBFloat16).to(torch::kFloat32);
  torch::Tensor act = torch::silu(gate_up.slice(1, 0, inter)) * gate_up.slice(1, inter, 2 * inter);
  const float act_input_scale = torch::max(torch::abs(act)).item<float>() / kFp8E4m3Max;
  torch::Tensor act_deq =
      (act / act_input_scale).to(torch::kFloat8_e4m3fn).to(torch::kFloat32) * act_input_scale;
  torch::Tensor expected = torch::matmul(act_deq, down_deq.t());

  // Scale tensors in the layout QuantWeight::BindFp8E4m3ScaleOfMoeWeight produces:
  //   up_gate.weight_scale[e] = up_gate.input_scale * per-expert weight scale
  //   up_gate.input_scale     = activation dequant scale
  //   down.weight_scale[e]    = down.input_scale * per-expert weight scale
  //   down.input_scale        = 1 / down activation scale (a multiplier, already reciprocated)
  torch::Tensor gate_up_scale = expert_scale * x_input_scale;
  torch::Tensor down_scale = expert_scale * act_input_scale;
  torch::Tensor gate_up_input_scale = torch::full({1}, x_input_scale, float_option);
  torch::Tensor down_input_scale = torch::full({1}, 1.0f / act_input_scale, float_option);

  torch::Tensor moe_output = torch::zeros({num_tokens, hidden}, bf16_option);

  std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
  MoeLayer moe_layer = MoeLayer();
  ASSERT_TRUE(moe_layer.Init(params, new_runtime_config, context_, kDeviceRank).OK());
  moe_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(moe_layer.GetWorkspaceSize()));
  ASSERT_TRUE(moe_layer.Preprocess(new_model_config, new_runtime_config).OK());

  std::vector<Tensor> inputs(4);
  inputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                     {static_cast<size_t>(num_tokens), static_cast<size_t>(hidden)}, hidden_states.data_ptr());
  inputs[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                     {static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)}, routing_out.data_ptr());
  inputs[2] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP8_E4M3,
                     {static_cast<size_t>(num_experts), static_cast<size_t>(2 * inter), static_cast<size_t>(hidden)},
                     gate_up_weight.data_ptr());
  inputs[3] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP8_E4M3,
                     {static_cast<size_t>(num_experts), static_cast<size_t>(hidden), static_cast<size_t>(inter)},
                     down_weight.data_ptr());

  std::vector<Tensor> weight_scales(2);
  weight_scales[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {static_cast<size_t>(num_experts)},
                            gate_up_scale.data_ptr());
  weight_scales[1] =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {static_cast<size_t>(num_experts)}, down_scale.data_ptr());
  std::vector<Tensor> input_scales(2);
  input_scales[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {1}, gate_up_input_scale.data_ptr());
  input_scales[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {1}, down_input_scale.data_ptr());
  inputs[2].weight_scales = &weight_scales[0];
  inputs[3].weight_scales = &weight_scales[1];
  inputs[2].input_scales = &input_scales[0];
  inputs[3].input_scales = &input_scales[1];

  std::vector<Tensor> outputs(1);
  outputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                      {static_cast<size_t>(num_tokens), static_cast<size_t>(hidden)}, moe_output.data_ptr());

  EXPECT_TRUE(moe_layer.Forward(&inputs[0], &inputs[1], &inputs[2], &inputs[3], nullptr, &outputs[0], nullptr).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  torch::Tensor actual = moe_output.to(torch::kFloat32);
  ASSERT_TRUE(torch::all(torch::isfinite(actual)).item<bool>()) << "hpc per-tensor MoE produced non-finite output";
  const float rel_err = RelativeL1Error(actual, expected);
  EXPECT_LT(rel_err, 0.02f) << "hpc per-tensor MoE relative L1 error " << rel_err;

#  endif
#endif
}

// Companion of the above for the blockwise-128x128 FP8 format, which shares the same MoeLayer
// dispatch branch. Weight scales are a [E, N/128, K/128] grid and the activation is quantized
// per-token-group at runtime, so there are no input_scales. down_weight_scale's last dim is
// inter/128 = 2 here, which exercises the pad-to-4 repacking inside the hpc bridge.
TEST_F(LayerTest, HpcBlockwiseFp8MoeLayerTest) {
#ifdef ENABLE_CUDA
#  ifdef ENABLE_FP8
  if (context_->device->ext->GetComputeCapacity() != 90) {
    GTEST_SKIP() << "hpc-ops MoE kernels are built for sm90a only";
  }
  constexpr int kDeviceRank = 0;
  constexpr int64_t kBlock = 128;

  ModelConfig new_model_config = model_config;
  RuntimeConfig new_runtime_config = runtime_config;
  new_model_config.weight_data_type = TYPE_BF16;
  new_runtime_config.inter_data_type = TYPE_BF16;
  new_runtime_config.moe_backend_config.moe_impl_choice = MoeBackendConfig::MoeImplChoice::AUTO;
  new_runtime_config.moe_backend_config.use_hpc_moe = true;

  torch::manual_seed(4321);

  const size_t expert_num = 8;
  const size_t expert_hidden_size = 512;
  const size_t expert_inter_size = 256;
  const size_t expert_topk = 2;
  const int64_t num_tokens = 16;
  const int64_t hidden = static_cast<int64_t>(expert_hidden_size);
  const int64_t inter = static_cast<int64_t>(expert_inter_size);
  const int64_t num_experts = static_cast<int64_t>(expert_num);

  std::vector<std::any> params = MakeHpcMoeLayerParams(expert_num, expert_hidden_size, expert_inter_size, expert_topk,
                                                       TYPE_BLOCK_FP8_E4M3);

  auto float_option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, kDeviceRank);
  auto bf16_option = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, kDeviceRank);
  auto fp8_option = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA, kDeviceRank);

  torch::Tensor hidden_states = torch::randn({num_tokens, hidden}, bf16_option) * 0.5f;
  torch::Tensor routing_out = torch::randn({num_tokens, num_experts}, bf16_option);

  // One logical expert replicated across all experts, so the output is routing-independent.
  torch::Tensor gate_up_payload = MakeFp8Payload({2 * inter, hidden}, float_option);
  torch::Tensor down_payload = MakeFp8Payload({hidden, inter}, float_option);
  torch::Tensor gate_up_block_scale = torch::rand({2 * inter / kBlock, hidden / kBlock}, float_option) * 0.02f + 0.01f;
  torch::Tensor down_block_scale = torch::rand({hidden / kBlock, inter / kBlock}, float_option) * 0.02f + 0.01f;

  // Dequantized weights: payload[n, k] * block_scale[n/128, k/128].
  auto expand_block_scale = [&](const torch::Tensor& s, int64_t n, int64_t k) {
    return s.unsqueeze(1).unsqueeze(3).expand({s.size(0), kBlock, s.size(1), kBlock}).reshape({n, k});
  };
  torch::Tensor gate_up_deq = gate_up_payload * expand_block_scale(gate_up_block_scale, 2 * inter, hidden);
  torch::Tensor down_deq = down_payload * expand_block_scale(down_block_scale, hidden, inter);

  torch::Tensor gate_up_weight =
      gate_up_payload.to(torch::kFloat8_e4m3fn).unsqueeze(0).repeat({num_experts, 1, 1}).contiguous();
  torch::Tensor down_weight =
      down_payload.to(torch::kFloat8_e4m3fn).unsqueeze(0).repeat({num_experts, 1, 1}).contiguous();
  torch::Tensor gate_up_weight_scale = gate_up_block_scale.unsqueeze(0).repeat({num_experts, 1, 1}).contiguous();
  torch::Tensor down_weight_scale = down_block_scale.unsqueeze(0).repeat({num_experts, 1, 1}).contiguous();

  // Reference forward with per-token-group activation quantization at both GEMM inputs.
  torch::Tensor x_deq = PerTokenGroupQuantDequant(hidden_states.to(torch::kFloat32), kBlock);
  torch::Tensor gate_up = torch::matmul(x_deq, gate_up_deq.t()).to(torch::kBFloat16).to(torch::kFloat32);
  torch::Tensor act = torch::silu(gate_up.slice(1, 0, inter)) * gate_up.slice(1, inter, 2 * inter);
  torch::Tensor act_deq = PerTokenGroupQuantDequant(act, kBlock);
  torch::Tensor expected = torch::matmul(act_deq, down_deq.t());

  torch::Tensor moe_output = torch::zeros({num_tokens, hidden}, bf16_option);

  std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
  MoeLayer moe_layer = MoeLayer();
  ASSERT_TRUE(moe_layer.Init(params, new_runtime_config, context_, kDeviceRank).OK());
  moe_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(moe_layer.GetWorkspaceSize()));
  ASSERT_TRUE(moe_layer.Preprocess(new_model_config, new_runtime_config).OK());

  std::vector<Tensor> inputs(4);
  inputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                     {static_cast<size_t>(num_tokens), static_cast<size_t>(hidden)}, hidden_states.data_ptr());
  inputs[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                     {static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)}, routing_out.data_ptr());
  inputs[2] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP8_E4M3,
                     {static_cast<size_t>(num_experts), static_cast<size_t>(2 * inter), static_cast<size_t>(hidden)},
                     gate_up_weight.data_ptr());
  inputs[3] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP8_E4M3,
                     {static_cast<size_t>(num_experts), static_cast<size_t>(hidden), static_cast<size_t>(inter)},
                     down_weight.data_ptr());

  std::vector<Tensor> weight_scales(2);
  weight_scales[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32,
                            {static_cast<size_t>(num_experts), static_cast<size_t>(2 * inter / kBlock),
                             static_cast<size_t>(hidden / kBlock)},
                            gate_up_weight_scale.data_ptr());
  weight_scales[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32,
                            {static_cast<size_t>(num_experts), static_cast<size_t>(hidden / kBlock),
                             static_cast<size_t>(inter / kBlock)},
                            down_weight_scale.data_ptr());
  inputs[2].weight_scales = &weight_scales[0];
  inputs[3].weight_scales = &weight_scales[1];

  std::vector<Tensor> outputs(1);
  outputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                      {static_cast<size_t>(num_tokens), static_cast<size_t>(hidden)}, moe_output.data_ptr());

  EXPECT_TRUE(moe_layer.Forward(&inputs[0], &inputs[1], &inputs[2], &inputs[3], nullptr, &outputs[0], nullptr).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  torch::Tensor actual = moe_output.to(torch::kFloat32);
  ASSERT_TRUE(torch::all(torch::isfinite(actual)).item<bool>()) << "hpc blockwise MoE produced non-finite output";
  const float rel_err = RelativeL1Error(actual, expected);
  EXPECT_LT(rel_err, 0.02f) << "hpc blockwise MoE relative L1 error " << rel_err;

#  endif
#endif
}

}  // namespace ksana_llm
