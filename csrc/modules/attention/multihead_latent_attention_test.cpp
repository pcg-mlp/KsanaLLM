/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>

#include "cache_manager/prefix_cache_manager.h"
#include "helpers/block_cache_layout_helper.h"
#include "helpers/model_test_helper.h"
#ifdef ENABLE_CUDA
#  include "kernels/nvidia/kernel_wrapper.h"
#  include "layers/cast_layer.h"
#endif
#ifdef ENABLE_KUNLUN
#  include "layers/kunlun/host_fp_helpers.h"
#endif
#include "models/base/model_weight.h"
#include "models/deepseek_v3/attention_meta/deepseek_v3_attention_meta_creator.h"
#include "modules/attention/multihead_latent_attention.h"
#include "samplers/sampler.h"
#include "tests/test.h"
#include "utils/attention_backend/attention_backend_manager.h"
#include "utils/calc_intvec_hash.h"
#include "utils/memory_allocator.h"
#include "utils/search_path.h"
#include "utils/singleton.h"

using namespace ksana_llm;

template <typename T>
DataType GetKsanaDataType();
#define GET_KSANA_DATA_TYPE(T, KSANA_TYPE) \
  template <>                              \
  DataType GetKsanaDataType<T>() {         \
    return KSANA_TYPE;                     \
  }
GET_KSANA_DATA_TYPE(int32_t, TYPE_INT32);
GET_KSANA_DATA_TYPE(float, TYPE_FP32);
// float16/bfloat16 是各平台的可移植 typedef(CUDA: half/__nv_bfloat16,昆仑芯: int16_t/uint16_t)。
GET_KSANA_DATA_TYPE(float16, TYPE_FP16);
GET_KSANA_DATA_TYPE(bfloat16, TYPE_BF16);
#undef GET_KSANA_DATA_TYPE

size_t schedule_id = 0;

Status CastDeviceTensorType(Tensor& input_tensor, DataType new_dtype, int dev_rank, std::shared_ptr<Context>& context) {
#ifdef ENABLE_ACL
  KLLM_THROW(fmt::format("Unsupported tensor cast for Ascend device"));
#elif ENABLE_TOPS
  KLLM_THROW(fmt::format("Unsupported tensor cast for Tops device"));
#endif
  if (input_tensor.dtype == new_dtype) {
    return Status();
  }
#ifdef ENABLE_CUDA
  if (input_tensor.dtype == DataType::TYPE_FP32 && new_dtype == DataType::TYPE_FP16) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, nullptr,
                               &(context->device->GetMemoryManageStream()));
    CUDA_CHECK_LAST_ERROR(
        llm_kernels::nvidia::FloatToHalf(input_tensor.GetPtr<float>(), input_tensor.GetElementNumber(),
                                         new_tensor.GetPtr<float16>(), context->device->GetMemoryManageStream().Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_FP32 && new_dtype == DataType::TYPE_BF16) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, nullptr,
                               &(context->device->GetMemoryManageStream()));
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::FloatToBFloat16(
        input_tensor.GetPtr<float>(), input_tensor.GetElementNumber(), new_tensor.GetPtr<bfloat16>(),
        context->device->GetMemoryManageStream().Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_FP16 && new_dtype == DataType::TYPE_FP32) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, nullptr,
                               &(context->device->GetMemoryManageStream()));
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::HalfToFloat(input_tensor.GetPtr<float16>(),
                                                           input_tensor.GetElementNumber(), new_tensor.GetPtr<float>(),
                                                           context->device->GetMemoryManageStream().Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_BF16 && new_dtype == DataType::TYPE_FP32) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, nullptr,
                               &(context->device->GetMemoryManageStream()));
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::BFloat16ToFloat(
        input_tensor.GetPtr<bfloat16>(), input_tensor.GetElementNumber(), new_tensor.GetPtr<float>(),
        context->device->GetMemoryManageStream().Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_BF16 && new_dtype == DataType::TYPE_FP16) {
    input_tensor.dtype = new_dtype;
    // Inplace cast
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::BFloat16ToHalf(
        input_tensor.GetPtr<void>(), input_tensor.GetElementNumber(), context->device->GetMemoryManageStream().Get()));
  } else if (input_tensor.dtype == DataType::TYPE_FP16 && new_dtype == DataType::TYPE_BF16) {
    input_tensor.dtype = new_dtype;
    // Inplace cast
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::HalfToBFloat16(
        input_tensor.GetPtr<void>(), input_tensor.GetElementNumber(), context->device->GetMemoryManageStream().Get()));
  } else {
    KLLM_THROW(fmt::format("Unsupported tensor cast from type {} to {}", input_tensor.dtype, new_dtype));
  }
#endif
#ifdef ENABLE_KUNLUN
  // 昆仑芯无对应的 device 端 cast kernel,与真机权重加载一致走 host 端转换:
  // 把 fp32 数据拷回 host,用 CPU 转成 fp16/bf16 的 uint16 位模式后再拷回 device。
  if (input_tensor.dtype == DataType::TYPE_FP32 &&
      (new_dtype == DataType::TYPE_FP16 || new_dtype == DataType::TYPE_BF16)) {
    const size_t element_num = input_tensor.GetElementNumber();
    // 等待此前对该 tensor 的写入(AssignFromVector / AddMlaWeight 都用 manage stream)。
    StreamSynchronize(context->device->GetMemoryManageStream());
    std::vector<float> host_fp32(element_num);
    MemcpyAsync(host_fp32.data(), input_tensor.GetPtr<void>(), element_num * sizeof(float), MEMCPY_DEVICE_TO_HOST,
                context->device->GetMemoryManageStream());
    StreamSynchronize(context->device->GetMemoryManageStream());

    std::vector<uint16_t> host_cast(element_num);
    if (new_dtype == DataType::TYPE_FP16) {
      for (size_t i = 0; i < element_num; i++) host_cast[i] = kunlun_host::Fp32ToHalf<float16>(host_fp32[i]);
    } else {
      for (size_t i = 0; i < element_num; i++) host_cast[i] = kunlun_host::Fp32ToHalf<bfloat16>(host_fp32[i]);
    }

    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, nullptr,
                               &(context->device->GetMemoryManageStream()));
    MemcpyAsync(new_tensor.GetPtr<void>(), host_cast.data(), element_num * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                context->device->GetMemoryManageStream());
    StreamSynchronize(context->device->GetMemoryManageStream());
    input_tensor = new_tensor;
  } else {
    KLLM_THROW(fmt::format("Unsupported tensor cast from type {} to {} on Kunlun", input_tensor.dtype, new_dtype));
  }
#endif
  return Status();
}

void AssignFromVector(Tensor& tensor, const std::vector<float>& f_vector, std::shared_ptr<Context>& context) {
  int device_rank;
  GetDevice(&device_rank);

  if (f_vector.size() != tensor.GetElementNumber()) {
    KLLM_THROW("Vector size does not match tensor element count");
  }

  Tensor fp32_cast = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, tensor.shape, nullptr,
                            &(context->device->GetMemoryManageStream()));
  MemcpyAsync(fp32_cast.GetPtr<void>(), f_vector.data(), f_vector.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context->device->GetMemoryManageStream());
  CastDeviceTensorType(fp32_cast, tensor.dtype, device_rank, context);
  MemcpyAsync(tensor.GetPtr<void>(), fp32_cast.GetPtr<void>(), tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE,
              context->device->GetMemoryManageStream());
  DeviceSynchronize();
}

template <typename T>
class MultiHeadLatentAttentionTestModel : public CommonModel {
 public:
  using CommonModel::context_;
  using CommonModel::forwarding_context_;
  using CommonModel::model_config_;

  using CommonModel::cast_layer_;

  std::shared_ptr<MultiHeadLatentAttention> mla_;
  MlaBuffers mla_buffers_;

  MultiHeadLatentAttentionTestModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                    const int rank, std::shared_ptr<Context> context,
                                    std::shared_ptr<BaseWeight> base_weight,
                                    std::shared_ptr<BaseModelCacheLayout> block_cache_layout)
      : CommonModel(model_config, runtime_config, rank, context) {
    ModelRunConfig model_run_config;
    model_run_config.position_encoding = PositionEncoding::ROPE;
    CommonModel::InitRunConfig(model_run_config, base_weight, block_cache_layout);
  }

  ~MultiHeadLatentAttentionTestModel() {}

  std::unique_ptr<AttentionMetaCreator> CreateCustomAttentionMetaCreator(
      const AttentionMetaCreator::Config& config) override {
    if (model_config_.use_mla) {
      return std::make_unique<DeepSeekV3AttentionMetaCreator>(config);
    }
    return nullptr;
  }

  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override {
    MultiHeadLatentAttention::CreateBuffers(CommonModel::GetBufferManager(), model_creation_config.attn_config,
                                            creation_context.runtime_config, mla_buffers_);
    bool is_neox = true;
    int layer_idx = 0;
    mla_ = std::make_shared<MultiHeadLatentAttention>(layer_idx, is_neox, creation_context, model_creation_config,
                                                      mla_buffers_);
    return Status();
  }

  Status LayerForward(ForwardingContext& forwarding_context_, const RunMode run_mode = RunMode::kMain) override {
    return Status();
  }

  Status CommonAttention(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                         const bool is_multi_token_forward, const std::vector<ForwardRequest>& forward_reqs) {
    // ForwardingContext 改为按 active slot 持有 ModelInput; 单测通过 GetModelInput() 访问当前 slot.
    auto& mi = forwarding_context_->GetModelInput();
    mi->ParseFromRequests(forward_reqs);

    // create forward shape tensor (CUDA 13D + Kunlun [13])
    forwarding_context_->GetAttentionForwardContext().forward_shape.shape = {
        mi->multi_token_request_num,     mi->multi_token_request_max_tokens,     mi->context_kv_cache_block_num,
        mi->single_token_request_num,    mi->single_token_request_max_tokens,    mi->decode_kv_cache_block_num,
        mi->dp_max_forwarding_tokens,    mi->dp_multi_token_request_num,         mi->dp_multi_token_request_max_tokens,
        mi->dp_single_token_request_num, mi->dp_single_token_request_max_tokens, mi->dp_total_prefix_len,
        mi->dp_decode_tokens,            mi->dp_max_prefix_per_req};

    CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context_->buffers_->hidden_buffer_0);
    CREATE_BUFFER_SCOPE(reduce_buffer_tensors, forwarding_context_->buffers_->shared_buffer);

    forwarding_context_->GetAttentionForwardContext().flag_tensor.template GetPtr<bool>()[0] = mi->use_cache;
    hidden_buffer_tensors_0[0].shape = {mi->input_ids.shape[0], model_config_.hidden_units};
    hidden_buffer_tensors_0[0].dtype = model_config_.weight_data_type;
    std::vector<float> input_data;
    input_data.reserve(hidden_buffer_tensors_0[0].GetElementNumber());
    size_t count = 0;
    for (const auto& req : forward_reqs) {
      // Skip cached prefix
      count += req.prefix_cache_len * model_config_.hidden_units;
      for (size_t i = 0; i < req.GetInputIdsLength(); i++) {
        for (size_t j = 0; j < model_config_.hidden_units; j++) {
          input_data.push_back(1.0f / (count % 97 * 0.1f + 1.0f) * pow(-1, (count % 7)));
          ++count;
        }
      }
    }
    AssignFromVector(hidden_buffer_tensors_0[0], input_data, context_);
    Status status = mla_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, nullptr, is_multi_token_forward,
                                  *forwarding_context_);
    // MultiHeadLatentAttention::Forward routes its final output by whether a ModelCommunicator
    // exists (created iff tensor_para_size > 1), NOT by backend. With a communicator (the Kunlun
    // test runs tp=2) the real MLA output is placed in reduce_buffer_tensors[0] for the caller's
    // AllReduce, while hidden_buffer_tensors_0 only holds intermediate scratch after the module's
    // internal double-buffer swap. Without a communicator (tp==1, e.g. the CUDA deepseekv2 test)
    // the output stays in hidden_buffer_tensors_0[0]. Pick the output buffer by comm presence.
    std::vector<Tensor>& output_tensors =
        forwarding_context_->GetModelCommunicator() ? reduce_buffer_tensors : hidden_buffer_tensors_0;
    output_tensors[0].dtype = model_config_.weight_data_type;
    forwarding_context_->GetAttentionForwardContext().forward_shape.shape = {0, 1, 1};
    {
      CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context_->buffers_->hidden_buffer_1);
      STATUS_CHECK_RETURN(static_cast<CastLayer*>(cast_layer_.get())
                              ->Forward(&output_tensors[0], 0, 1, 1, &hidden_buffer_tensors_1[0]));
      StreamSynchronize(context_->device->GetPrimaryComputeStream());
      Memcpy(input_data.data(), hidden_buffer_tensors_1[0].template GetPtr<void>(),
             sizeof(float) * output_tensors[0].GetElementNumber(), MEMCPY_DEVICE_TO_HOST);
      // Reset dtype
      hidden_buffer_tensors_1[0].dtype = model_config_.weight_data_type;
    }
#ifdef ENABLE_KUNLUN
    // 昆仑芯 W4A8(int8 权重 + fp32 scale)MLA 前向的期望输出。三种前向由
    // (is_multi_token_forward, dp_total_prefix_len) 区分:prefill(无 prefix)/prefill(带 prefix cache)/decode。
    // 输入与权重均为确定性构造,故可硬编码逐元素对比。这里的值取自模块的真实输出 buffer
    // (上方 output_tensors:tp>1 时为 reduce_buffer_tensors[0],即 o_proj 结果),而非内部
    // 中间 scratch。若需重新采集,把 KUNLUN_MLA_DUMP=1 传入环境即可打印实际 output[0:6]。
    if (std::getenv("KUNLUN_MLA_DUMP") != nullptr) {
      std::cout << "[Kunlun MLA] is_multi_token_forward=" << is_multi_token_forward
                << ", dp_total_prefix_len=" << forwarding_context_->GetModelInput()->dp_total_prefix_len
                << ", kv_cache_dtype=" << static_cast<int>(runtime_config_.attn_backend_config.kv_cache_dtype)
                << ", output[0:6]={";
      for (size_t i = 0; i < 6 && i < input_data.size(); i++) {
        std::cout << input_data[i] << (i + 1 < 6 ? ", " : "");
      }
      std::cout << "}" << std::endl;
    }
    std::vector<float> kunlun_expected;
    if (is_multi_token_forward) {
      if (forwarding_context_->GetModelInput()->dp_total_prefix_len == 0) {
        kunlun_expected = {-0.0152435f, -0.00582123f, 0.0150986f, 0.00939941f, -0.0230865f, -0.007267f};
      } else {
        kunlun_expected = {-0.0125275f, -0.00996399f, 0.0114136f, 0.0125504f, -0.0190735f, -0.0125504f};
      }
    } else {
      kunlun_expected = {-0.0143967f, 0.00323296f, 0.0186462f, -0.00833893f, -0.0189362f, 0.00421143f};
    }
    for (size_t i = 0; i < kunlun_expected.size(); i++) {
      // 数值很小且经 fp16 + int8 动态量化,近零值用相对+绝对混合容差避免相对误差发散。
      const float tol = std::fabs(kunlun_expected[i]) * 0.05f + 2e-5f;
      EXPECT_NEAR(input_data[i], kunlun_expected[i], tol);
    }
    return status;
#endif
    std::vector<float> output;
    if (is_multi_token_forward) {
      if (model_config_.use_dsa && std::getenv("CONTEXT_DSA_THRESHOLD") != nullptr) {
        // Prefill phase (no prefix cache for DSA test)
        // The numerical difference here is due to sparse mla applies weight absorption for prefill tokens,
        // while in this test, weights w_uk_t/w_uv and kv_b_nope_proj/v_head_proj are not equivalent
        if (forwarding_context_->GetModelInput()->dp_total_prefix_len == 0) {
          if (runtime_config_.attn_backend_config.kv_cache_dtype == TYPE_FP8_DS_MLA) {
            output = {1416, 65.5, -1248, -2672, 856, -692};
          } else {
            output = {1408, 61.75, -1256, -2672, 856, -688};
          }
        } else {
          output = {604, 880, 374, -474, 134, -708};
        }
      } else if (runtime_config_.attn_backend_config.kv_cache_dtype == TYPE_FP8_E4M3) {
        if (forwarding_context_->GetModelInput()->dp_total_prefix_len == 0) {
          output = {-1984, 2096, -3392, 784, 117.5, -564};
        } else {
          output = {346, 588, -1008, 764, -648, 936};
        }
      } else if (runtime_config_.attn_backend_config.kv_cache_dtype == TYPE_FP8_DS_MLA) {
        if (forwarding_context_->GetModelInput()->dp_total_prefix_len == 0) {
          output = {-1976, 2096, -3392, 776, 126, -576};
        } else {
          output = {318, 632, -1032, 756, -672, 924};
        }
      } else {
        if (forwarding_context_->GetModelInput()->dp_total_prefix_len == 0) {
          output = {-1977, 2088, -3386, 772.5, 125.75, -575.5};
        } else {
          output = {349, 585, -1004, 759, -643.5, 928};
        }
      }
    } else {
      if (model_config_.use_dsa) {
        if (runtime_config_.attn_backend_config.kv_cache_dtype == TYPE_FP8_DS_MLA) {
          output = {744, 748, 118, -684, 201, -716};
        } else {
          output = {752, 716, -57, -920, 276, -728};
        }
      } else if (runtime_config_.attn_backend_config.kv_cache_dtype == TYPE_FP8_E4M3) {
        output = {548, 940, 362, -45.75, -75.5, -564};
      } else {
        output = {559.5, 935.5, 366.25, -55.0312, -76.1875, -564};
      }
    }

    for (size_t i = 0; i < output.size(); i++) {
      const float diff = std::fabs((input_data[i] - output[i]) / output[i]);
      EXPECT_LT(diff, 0.05);
    }

    return status;
  }
};

template <typename T>
class TestWeight : public BaseWeight {
 public:
  using ksana_llm::BaseWeight::weights_map_;
  TestWeight() {}
  explicit TestWeight(const ksana_llm::ModelConfig& model_config, const ksana_llm::RuntimeConfig& runtime_config,
                      int rank, std::shared_ptr<ksana_llm::Context> context)
      : ksana_llm::BaseWeight(model_config, runtime_config, rank, context) {}
  ~TestWeight() override {}

  Tensor GetModelWeights(const std::string& weight_name) override {
    auto it = weights_map_.find(weight_name);
    if (it == weights_map_.end()) {
      KLLM_LOG_WARNING << fmt::format("weight_name: {} not in weights map", weight_name);
      return Tensor();
    }
    return it->second;
  }

  void AddMlaWeight(int device_id = 0) {
#ifdef ENABLE_KUNLUN
    // Kunlun/Kimi-K2.6:昆仑芯 MLA 只实现了 W4A8(int8 权重 + fp32 per-channel scale)路径,
    // 非量化 fp16 GEMM 根本没实现;因此 GEMM/BMM 权重必须是 TYPE_INT8 且带 weight_scale,
    // 只有 norm/layernorm 权重与激活用 fp16(真机用 bf16,测试放宽为 fp16 即可)。
    //
    // 权重形状取自真机 tp8/ep8 dump(trans_b=true,2D GEMM 布局 {out=n, in=k};
    // BMM w_uk_t/w_uv 布局 {B, N, K}),从真实 model_config 推导。
    // 参考目录:ksana_vs_xsglang_dump_weights/Kimi-K2.6-klx-w4a8-gptq-trans_b_true/0
    //
    // scale 约定见 kunlun_w4a8_layer_test.cpp:564——weight_scale 已被预乘 127
    // (kernel 内部再除 127);此处只需给出量级合理的正数即可,数值待真机 dump 后硬编码。
    // scale 形状:2D GEMM 为 per-channel [n];BMM 为 [B, N]。绑定方式见同文件 547 行:
    // int8 权重张量的 weight_scales 指针指向对应 scale 张量。
    //
    // 昆仑 MLA 走 fused_lora_a_proj + rmsnorm_split_col 融合路径(见
    // multihead_latent_attention.cpp:66),因此这里造 fused_lora_a_proj 而非未融合的
    // q_a_proj/kv_a_lora_proj/kv_a_rope_proj 三件套。
    const size_t hidden = model_config_.hidden_units;
    const size_t head_num = model_config_.head_num;
    // head_per_tp 必须与 forwarding_context.cpp:48 用同一公式、同一 attn_tensor_parallel_size,
    // 否则前向缓冲区(head_num_per_tp)与权重(o_proj/q_b 等)维度对不上。
    const size_t attn_tp = runtime_config_.parallel_basic_config.attn_tensor_parallel_size;
    const size_t head_per_tp = head_num / attn_tp;
    const size_t q_lora = model_config_.mla_config.q_lora_rank;
    const size_t kv_lora = model_config_.mla_config.kv_lora_rank;
    const size_t nope = model_config_.mla_config.qk_nope_head_dim;
    const size_t rope = model_config_.mla_config.qk_rope_head_dim;
    const size_t v_head = model_config_.mla_config.v_head_dim;
    // 测试只创建 layer 0(见 MultiHeadLatentAttentionTestModel::CreateLayers),Kimi 权重维度很大
    // (hidden=7168),只造 layer 0 即可,避免为 61 层造巨量无用权重。
    const int layer_num = 1;

    auto& stream = context_->device->GetMemoryManageStream();
    const DataType act_type = GetKsanaDataType<T>();  // fp16:激活/norm dtype

    // fp16 norm/layernorm 权重:fp32 host 填充后 cast 到激活 dtype(与真机 host 端加载一致)。
    auto add_norm = [&](const std::string& name, const std::vector<size_t>& shape) {
      weights_map_[name] = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, shape, nullptr, &stream);
      Tensor& t = weights_map_.at(name);
      std::vector<float> data(t.GetElementNumber());
      for (size_t j = 0; j < data.size(); j++) data[j] = 1.0f / (j % 97 * 0.1f + 1.0f) * pow(-1, (j % 7));
      MemcpyAsync(t.template GetPtr<void>(), data.data(), t.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE, stream);
      CastDeviceTensorType(t, act_type, device_id, context_);
    };

    // int8 权重 + fp32 scale。weight_shape 首维为 n(输出通道),scale 形状 scale_shape。
    // 先建张量,scale 绑定放到后面统一做(避免依赖插入顺序)。
    auto add_int8 = [&](const std::string& wname, const std::vector<size_t>& weight_shape,
                        const std::vector<size_t>& scale_shape) {
      weights_map_[wname] =
          Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_INT8, weight_shape, nullptr, &stream);
      Tensor& w = weights_map_.at(wname);
      std::vector<int8_t> wdata(w.GetElementNumber());
      for (size_t j = 0; j < wdata.size(); j++) wdata[j] = static_cast<int8_t>(static_cast<int>(j % 127) - 64);
      MemcpyAsync(w.template GetPtr<void>(), wdata.data(), wdata.size() * sizeof(int8_t), MEMCPY_HOST_TO_DEVICE,
                  stream);

      const std::string sname = wname + "_scale";  // 仅用于在 weights_map_ 中持有,MLA 通过指针取用
      weights_map_[sname] = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, scale_shape, nullptr, &stream);
      Tensor& s = weights_map_.at(sname);
      std::vector<float> sdata(s.GetElementNumber());
      for (size_t j = 0; j < sdata.size(); j++) sdata[j] = 0.01f + (j % 13) * 0.001f;
      MemcpyAsync(s.template GetPtr<void>(), sdata.data(), sdata.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE, stream);
    };

    std::vector<std::pair<std::string, std::string>> scale_bindings;  // {weight_name, scale_name}
    for (int i = 0; i < layer_num; i++) {
      const std::string p = fmt::format("model.layers.{}", i);
      add_norm(p + ".input_layernorm.weight", {hidden});
      add_norm(p + ".post_attention_layernorm.weight", {hidden});
      add_norm(p + ".self_attn.q_a_layernorm.weight", {q_lora});
      add_norm(p + ".self_attn.kv_a_layernorm.weight", {kv_lora});

      // 2D GEMM 权重 {n, k} + per-channel scale {n}
      const std::vector<std::pair<std::string, std::vector<size_t>>> gemm = {
          {p + ".self_attn.fused_lora_a_proj.weight",
           { q_lora + kv_lora + rope,
             hidden }},
          {p + ".self_attn.q_b_nope_rope_proj.weight",
           { head_per_tp * (nope + rope),
             q_lora }},
          {p + ".self_attn.kv_b_nope_proj.weight",
           { head_per_tp * nope,
             kv_lora }},
          {p + ".self_attn.kv_b_combined_proj.weight",
           { head_per_tp * (nope + v_head),
             kv_lora }},
          {p + ".self_attn.v_head_proj.weight",
           { head_per_tp * v_head,
             kv_lora }},
          {p + ".self_attn.o_proj.weight",
           { hidden,
             head_per_tp * v_head }},
      };
      for (const auto& [name, shape] : gemm) {
        add_int8(name, shape, {shape[0]});
        scale_bindings.emplace_back(name, name + "_scale");
      }

      // BMM 权重 {B, N, K} + scale {B, N}
      const std::string uk = p + ".self_attn.w_uk_t.weight";  // {head_per_tp, kv_lora, nope}
      const std::string uv = p + ".self_attn.w_uv.weight";    // {head_per_tp, nope, kv_lora}
      add_int8(uk, {head_per_tp, kv_lora, nope}, {head_per_tp, kv_lora});
      add_int8(uv, {head_per_tp, nope, kv_lora}, {head_per_tp, nope});
      scale_bindings.emplace_back(uk, uk + "_scale");
      scale_bindings.emplace_back(uv, uv + "_scale");
    }

    // 统一绑定 weight_scales。weights_map_(std::unordered_map)对元素的指针/引用在
    // 插入/rehash 后保持稳定,因此这里存 &weights_map_.at(sname) 安全。
    for (const auto& [wname, sname] : scale_bindings) {
      weights_map_.at(wname).weight_scales = &weights_map_.at(sname);
    }
    StreamSynchronize(stream);
    return;
#else
    std::unordered_map<std::string, std::vector<size_t>> add_tensor_map;
    const int layer_num = 4;
    for (int i = 0; i < layer_num; i++) {
      add_tensor_map[fmt::format("model.layers.{}.self_attn.kv_a_layernorm.weight", i)] = {512};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.o_proj.weight", i)] = {2048, 2048};
      add_tensor_map[fmt::format("model.layers.{}.post_attention_layernorm.weight", i)] = {2048};
      add_tensor_map[fmt::format("model.layers.{}.input_layernorm.weight", i)] = {2048};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.kv_a_rope_proj.weight", i)] = {2048, 64};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.v_head_proj.weight", i)] = {512, 2048};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.kv_a_lora_proj.weight", i)] = {2048, 512};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.kv_b_nope_proj.weight", i)] = {512, 2048};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.w_uk_t.weight", i)] = {16, 128, 512};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.w_uv.weight", i)] = {16, 512, 128};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.q_b_nope_rope_proj.weight", i)] = {2048, 3072};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.q_a_proj.weight", i)] = {2048, 1536};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.q_a_layernorm.weight", i)] = {1536};
      // Add indexer weights
      add_tensor_map[fmt::format("model.layers.{}.self_attn.indexer.wq_b.weight", i)] = {4096, 1536};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.indexer.wk.weight", i)] = {128, 2048};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.indexer.weights_proj.weight", i)] = {32, 2048};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.indexer.k_norm.weight", i)] = {128};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.indexer.k_norm.bias", i)] = {128};
    }
    DataType weight_type = GetKsanaDataType<T>();
    for (auto& [tensor_name, shape] : add_tensor_map) {
      weights_map_[tensor_name] = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, shape, nullptr,
                                         &(context_->device->GetMemoryManageStream()));
      Tensor& tensor = weights_map_.at(tensor_name);
      std::vector<float> input_data(tensor.GetElementNumber());
      for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = 1.0f / (i % 97 * 0.1f + 1.0f) * pow(-1, (i % 7));
      }
      MemcpyAsync(tensor.template GetPtr<void>(), reinterpret_cast<void*>(input_data.data()), tensor.GetTotalBytes(),
                  MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
      CastDeviceTensorType(tensor, weight_type, device_id, context_);
    }
#endif
  }
};

// 定义一个 MlaTest 类,继承自 testing::Test
class MlaTest : public testing::Test {
 protected:
  void SetUp() override {
    origin_stderr_verbosity = loguru::g_stderr_verbosity;
    loguru::g_stderr_verbosity = loguru::Verbosity_MAX;

    const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    test_name = test_info->name();

    // Kunlun 平台复用与 CUDA 相同的 MLA 前向流程,由 Kimi-K2.6 的配置驱动。这里使用
    // 专用的单卡 light 配置 ksana_llm_kimi_k26_kunlun_light.yaml(tensor_para_size=1),
    // 与 CUDA 侧 deepseekv2 的做法一致:让测试可以用 XPU_VISIBLE_DEVICES 钉在单卡运行,
    // 而不会被生产用 8卡 yaml 的 tensor_para_size=8 卡在 ParseConfig 的 device_num 守卫。
    const bool is_kunlun_test = test_name.find("Kunlun") != std::string::npos;

    // 解析 config.json,初始化 ModelConfig 以及 BlockManager
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate =
        parent_path / (is_kunlun_test ? "../../../examples/ksana_llm_kimi_k26_kunlun_light.yaml"
                                      : "../../../examples/ksana_llm_deepseekv2.yaml");
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    AttentionBackendManager::GetInstance()->Initialize();
    const auto& env = Singleton<Environment>::GetInstance();
    context = std::make_shared<Context>();
    context->global.reset(new GlobalContext(1, 1, 1, env));
    context->global->InitDistributedConfig();

    context->device = std::make_shared<DeviceContext>(1);
#ifdef ENABLE_CUDA
    // NCCL 仅在 NVIDIA 平台需要;昆仑芯单卡(GlobalContext tp=1)跑 MLA 不涉及跨卡通信。
    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context->device->ext->InitNcclParam(nccl_uid);
#endif

    if (is_kunlun_test) {
      // light yaml 里 model_dir 已指向真实 Kimi-K2.6 模型目录,无需 override。
      env->ParseConfig(config_path);
      // 复用同一套流程,但按昆仑芯 2 卡分片场景把并行度设成 tp=2 / ep=2。
      env->SetTensorParallelSize(2);
      env->SetExpertParallelSize(2);
      // light yaml 的 tensor_para_size=1 会让 ParseConfig 把 attn_tensor_parallel_size 算成 1
      // (attn_dp=1),而 SetTensorParallelSize 不会重算它。MLA 权重形状与下面硬编码的期望值
      // 是在 attn_tensor_parallel_size=8(head_per_tp=64/8=8,即真实 8卡部署的单卡分片)下抓取的,
      // 因此这里显式钉成 8,使单卡 light 配置下的分片形状与期望值保持一致。
      env->schedule_config_parser_.runtime_config_.parallel_basic_config.attn_tensor_parallel_size = 8;
    } else {
      env->ParseConfig(config_path,
                       std::filesystem::absolute(config_path_relate.parent_path() / "deepseekv2").string());
    }
    env->GetModelConfig(model_config);
    InitPipelineConfigForTest(env, model_config);

    // kv_cache_dtype must be set before calling env->InitializeBlockManagerConfig()
    if (test_name.find("FlashMlaKvFP8") != std::string::npos) {
      env->schedule_config_parser_.runtime_config_.attn_backend_config.kv_cache_dtype = TYPE_FP8_E4M3;
    } else if (test_name.find("FlashDsa") != std::string::npos) {
      // For DSA test, we need to set use_dsa BEFORE InitializeBlockManagerConfig
      // so that indexer kv_cache_config can be properly initialized
      model_config.use_dsa = true;
      model_config.dsa_config.index_head_dim = 128;
      model_config.dsa_config.index_n_heads = 32;
      model_config.dsa_config.index_topk = 2048;
      model_config.mla_config.q_lora_rank = 1536;
      // Update the environment's model config
      env->SetModelConfig(model_config);
      if (test_name.find("FP8") != std::string::npos) {
        env->schedule_config_parser_.runtime_config_.attn_backend_config.kv_cache_dtype = TYPE_FP8_E4M3;
      } else {
        env->schedule_config_parser_.runtime_config_.attn_backend_config.kv_cache_dtype = TYPE_FP8_DS_MLA;
      }
    }
    block_cache_layout_ = GetTestModelCacheLayout(env);
    env->InitializeBlockManagerConfig(block_cache_layout_);
    BlockManagerConfig block_manager_config;
    env->GetBlockManagerConfig(block_manager_config);
    block_manager_config.block_host_memory_factor = 0;
    block_manager_config.reserved_device_memory_ratio = 0.98;
    block_manager_config.host_allocator_config.blocks_num = 0;
    block_manager_config.device_allocator_config.blocks_num = 10;
    env->SetBlockManagerConfig(block_manager_config);
    KLLM_LOG_DEBUG << fmt::format("block_size {}", block_manager_config.device_allocator_config.block_size);

    env->GetRuntimeConfig(runtime_config);
    runtime_config.max_batch_size = 5;
    runtime_config.max_seq_len = 20;
    runtime_config.max_step_token_num = 40;
    runtime_config.enable_prefix_caching = true;
    env->schedule_config_parser_.runtime_config_ = runtime_config;

    env->GetPipelineConfig(pipeline_config);

    BlockAllocatorGroupConfig group_0_config;
    group_0_config.devices = {0};
    group_0_config.device_block_num = block_manager_config.device_allocator_config.blocks_num;
    group_0_config.host_block_num = block_manager_config.host_allocator_config.blocks_num;
    group_0_config.block_size = block_manager_config.device_allocator_config.block_size;

    BlockAllocatorManagerConfig block_allocator_manager_config;
    block_allocator_manager_config[0] = group_0_config;

    auto memory_allocator_ = std::make_shared<MemoryAllocator>();
    engine_block_manager =
        std::make_shared<BlockAllocatorManager>(true, block_allocator_manager_config, memory_allocator_, context);
    engine_block_group = engine_block_manager->GetBlockAllocatorGroup(0);
    executor_block_manager =
        std::make_shared<BlockAllocatorManager>(false, block_allocator_manager_config, memory_allocator_, context);
    executor_block_group = executor_block_manager->GetBlockAllocatorGroup(0);

    CacheManagerConfig cache_manager_config;
    cache_manager_config.block_token_num = block_manager_config.device_allocator_config.block_token_num;
    cache_manager_config.tensor_para_size = 1;
    cache_manager_config.enable_prefix_caching = true;
    cache_manager = std::make_shared<PrefixCacheManager>(cache_manager_config, engine_block_group);
  }

  void TearDown() override { loguru::g_stderr_verbosity = origin_stderr_verbosity; }

 protected:
  int origin_stderr_verbosity = loguru::Verbosity_MAX;
  std::string test_name;
  ModelConfig model_config;
  RuntimeConfig runtime_config;
  PipelineConfig pipeline_config;
  std::shared_ptr<BlockAllocatorManager> engine_block_manager;
  std::shared_ptr<BlockAllocatorManager> executor_block_manager;
  std::shared_ptr<BlockAllocatorGroupInterface> engine_block_group;
  std::shared_ptr<BlockAllocatorGroupInterface> executor_block_group;
  std::shared_ptr<Context> context;
  std::shared_ptr<BaseModelCacheLayout> block_cache_layout_ = nullptr;

  std::shared_ptr<PrefixCacheManager> cache_manager;

  template <typename weight_data_type>
  void TestMlaForward(int device_id = 0) {
    SetDevice(device_id);
    std::shared_ptr<TestWeight<weight_data_type>> base_weight =
        std::make_shared<TestWeight<weight_data_type>>(model_config, runtime_config, 0, context);
    base_weight->AddMlaWeight(device_id);

    std::shared_ptr<ksana_llm::BaseWeight> bs1 = base_weight;
    std::shared_ptr<MultiHeadLatentAttentionTestModel<weight_data_type>> test_mla_model =
        std::make_shared<MultiHeadLatentAttentionTestModel<weight_data_type>>(model_config, runtime_config, 0, context,
                                                                              bs1, block_cache_layout_);

    // ContextDecode
    ForwardRequest forward;
    forward.block_allocator_manager = executor_block_manager;
    std::vector<int> input_ids = {233, 1681};
    forward.forwarding_tokens = std::make_shared<std::vector<int>>(input_ids);
    std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks;
    forward.flexible_cached_copy_tasks = &flexible_cached_copy_tasks;
    std::vector<int> input_refit_pos;
    std::vector<std::vector<float>> input_refit_embedding;
    EmbeddingSlice embedding_slice;
    embedding_slice.pos = input_refit_pos;
    embedding_slice.embeddings = input_refit_embedding;
    forward.input_refit_embedding = &embedding_slice;
    std::vector<int> block_ids;
    engine_block_group->GetDeviceBlockAllocator(0)->AllocateBlocks(1, block_ids);
    executor_block_group->GetDeviceBlockAllocator(0)->GetBlockPtrs(block_ids, forward.kv_cache_ptrs);
    forward.atb_kv_cache_base_blk_ids.assign(1, {});
    BuildFlatKVCacheBlockIds(GetTotalLayerNum(model_config, pipeline_config), {block_ids},
                             forward.atb_kv_cache_base_blk_ids, executor_block_group->GetDeviceBlockAllocator(0));

    test_mla_model->CommonAttention(0, bs1, true, {forward});

    // Test Prefix caching
    forward.forwarding_tokens->push_back(321);
    forward.prefix_cache_len = 1;
    forward.kv_cached_token_num = 1;
    test_mla_model->CommonAttention(0, bs1, true, {forward});
    forward.forwarding_tokens->pop_back();

    // Decode
    forward.forwarding_tokens->push_back(321);
    forward.prefix_cache_len = 0;
    forward.kv_cached_token_num = 2;
    test_mla_model->CommonAttention(0, bs1, false, {forward});
  }
};

// 运行时检测是否是 Hopper 架构的辅助函数
bool IsHopperSupported() {
#ifdef ENABLE_CUDA
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return false;
  }

  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
    return false;
  }

  int compute_capability = major * 10 + minor;
  // Hopper 架构是 SM 9.0 (compute capability 90)
  return compute_capability >= 90;
#else
  return false;
#endif
}

TEST_F(MlaTest, ForwardWithFlashMlaTest) {
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif
  // 运行时检测 GPU 架构，只有在 Hopper 架构时才运行测试
  if (!IsHopperSupported()) {
    GTEST_SKIP_("Test requires FA3 support or Hopper architecture (SM >= 9.0)");
    return;
  }
  // fp16 forward
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_FP16;
  model_config.quant_config.method = QUANT_NONE;
  runtime_config.inter_data_type = model_config.weight_data_type;
  const auto& env = Singleton<Environment>::GetInstance();
  env->schedule_config_parser_.runtime_config_.inter_data_type = model_config.weight_data_type;
  std::cout << "Test TYPE_FP16 weight_data_type forward." << std::endl;
  TestMlaForward<float16>();
  return;
}

TEST_F(MlaTest, ForwardWithFlashMlaKvFP8Test) {
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif
  // 运行时检测 GPU 架构，只有在支持 FA3 或 Hopper 架构时才运行测试
  if (!IsHopperSupported()) {
    GTEST_SKIP_("Test requires FA3 support or Hopper architecture (SM >= 9.0)");
    return;
  }
  // bf16 forward, FA3 only supports BF16 output for FP8 input
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_BF16;
  model_config.quant_config.method = QUANT_NONE;
  runtime_config.inter_data_type = model_config.weight_data_type;
  const auto& env = Singleton<Environment>::GetInstance();
  env->schedule_config_parser_.runtime_config_.inter_data_type = model_config.weight_data_type;
  std::cout << "Test TYPE_BF16 weight_data_type forward." << std::endl;
  TestMlaForward<bfloat16>();

  // test kv scales, FA3 requires special scale input
  model_config.k_scales[0] = 0.5f;
  model_config.v_scales[0] = 0.5f;
  std::cout << "Test TYPE_BF16 weight_data_type forward with kv scales." << std::endl;
  TestMlaForward<bfloat16>();
  return;
}

TEST_F(MlaTest, ForwardWithFlashDsaTest) {
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif

  // 运行时检测 GPU 架构，只有在 Hopper 架构时才运行测试
  if (!IsHopperSupported()) {
    GTEST_SKIP_("Test requires Hopper architecture (SM >= 9.0)");
    return;
  }
  // BF16 forward, DeepSeek Sparse MLA only supports BF16
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_BF16;
  model_config.quant_config.method = QUANT_NONE;
  runtime_config.inter_data_type = model_config.weight_data_type;
  const auto& env = Singleton<Environment>::GetInstance();
  env->schedule_config_parser_.runtime_config_.inter_data_type = model_config.weight_data_type;
  std::cout << "Test TYPE_BF16 weight_data_type forward." << std::endl;
  // Adaptive fall back
  TestMlaForward<bfloat16>();
  // Force DSA
  setenv("CONTEXT_DSA_THRESHOLD", "0", 1);
  TestMlaForward<bfloat16>();
  // Unset the env
  unsetenv("CONTEXT_DSA_THRESHOLD");
  return;
}

TEST_F(MlaTest, ForwardWithFlashDsaFP8Test) {
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif

  // 运行时检测 GPU 架构，只有在 Hopper 架构时才运行测试
  if (!IsHopperSupported()) {
    GTEST_SKIP_("Test requires Hopper architecture (SM >= 9.0)");
    return;
  }
  // BF16 forward, DeepSeek Sparse MLA only supports BF16
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_BF16;
  model_config.quant_config.method = QUANT_NONE;
  runtime_config.inter_data_type = model_config.weight_data_type;
  const auto& env = Singleton<Environment>::GetInstance();
  env->schedule_config_parser_.runtime_config_.inter_data_type = model_config.weight_data_type;
  std::cout << "Test TYPE_BF16 weight_data_type forward with kv fp8." << std::endl;
  // Force DSA
  setenv("CONTEXT_DSA_THRESHOLD", "0", 1);
  TestMlaForward<bfloat16>();
  // Unset the env
  unsetenv("CONTEXT_DSA_THRESHOLD");
  return;
}

#ifdef ENABLE_KUNLUN
// 昆仑芯 P800 上复用与 CUDA 相同的 MLA 前向流程:
// SetUp 会识别测试名中的 "Kunlun",改用 ksana_llm_kimi_k26_kunlun.yaml,
// 并把 tp_size / ep_size 覆盖为 2。
//
// 这是 Kimi-K2.6 MLA 模型,昆仑芯上要跑的就是 W4A8:GEMM/BMM 权重是 int8 + fp32
// per-channel scale,非量化 fp16 GEMM 根本没实现。因此这里保持 quant_config.method
// = QUANT_W4A8_KLX(matmul_layer_factory 据此 + 权重实际 dtype=INT8 路由到
// KunlunInt8MatMulLayer),权重由 AddMlaWeight 造成 int8+scale;激活/norm 仍用 fp16。
TEST_F(MlaTest, ForwardWithKunlunMlaTest) {
  model_config.is_quant = true;
  model_config.weight_data_type = TYPE_FP16;  // 激活/norm dtype;GEMM 权重是 int8
  model_config.quant_config.method = QUANT_W4A8_KLX;
  runtime_config.inter_data_type = model_config.weight_data_type;
  const auto& env = Singleton<Environment>::GetInstance();
  env->schedule_config_parser_.runtime_config_.inter_data_type = model_config.weight_data_type;
  std::cout << "Test Kunlun W4A8_KLX (int8 weight + fp32 scale) MLA forward." << std::endl;
  TestMlaForward<float16>();
  return;
}
#endif
