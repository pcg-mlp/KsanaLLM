/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/flash_sparse_mla_indexer_layer.h"

#include <fmt/core.h>
#include <torch/torch.h>

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/fused_rope_quant_scale.h"
#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/rope.h"
#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/store.h"
#include "kernels/nvidia/deepseek_deepgemm_wrapper.h"
#include "layers/attention_layer.h"  // For yarn mscale functions and InitYarnRotaryEmbedding
#include "layers/topk_layer.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"

namespace ksana_llm {

// 解析输入张量的辅助结构
template <typename SCALAR_T>
struct FlashMlaInputData {
  SCALAR_T* q_ptr;
  SCALAR_T* k_ptr;
  SCALAR_T* weights_ptr;
  uint8_t* quant_workspace_ptr;
  const Tensor& rotary_pos;
  const int* store_out_loc_ptr;
  const Tensor& extend_lens_tensor;
  const Tensor& seq_len_offset_tensor;  // dp_input_offset: full sequence cumsum
  const Tensor* block_table_tensor;
  const Tensor* cur_seq_len_start_tensor;
  const Tensor* cur_seq_len_end_tensor;
  const Tensor* layer_indexer_kv_cache_ptr_tensor;

  size_t total_q_tokens;  // new tokens without prefix
  size_t total_k_tokens;  // k may be compressed
  size_t batch_size;
  size_t total_prefix_tokens;  // prefix tokens
  size_t total_tokens;         // new tokens + prefix tokens
  cudaStream_t stream;
};

// FP8 量化工作空间布局
struct FlashQuantWorkspace {
  uint8_t* q_fp8_ptr;
  float* q_scale_ptr;
  uint8_t* k_fp8_ptr;
  float* k_scale_ptr;
  float* logits_ptr;
  size_t aligned_seq_len_kv;
};

Status FlashSparseMlaIndexerLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                        std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  KLLM_CHECK_WITH_INFO(
      inter_data_type_ == DataType::TYPE_BF16,
      fmt::format("FlashSparseMlaIndexerLayer: unsupported inter_data_type {}", GetTypeString(inter_data_type_)));

  // Parse parameters
  int param_idx = 0;
  n_heads_ = std::any_cast<int>(parameters[param_idx++]);
  head_dim_ = std::any_cast<int>(parameters[param_idx++]);
  rope_head_dim_ = std::any_cast<int>(parameters[param_idx++]);
  index_topk_ = std::any_cast<int>(parameters[param_idx++]);
  KLLM_CHECK_WITH_INFO(index_topk_ == 512 || index_topk_ == 1024 || index_topk_ == 2048,
                       "FlashSparseMlaIndexerLayer requires index_topk in {512, 1024, 2048}");
  block_size_ = std::any_cast<int>(parameters[param_idx++]);
  quant_block_size_ = std::any_cast<int>(parameters[param_idx++]);
  KLLM_CHECK_WITH_INFO(quant_block_size_ == 128, "FlashSparseMlaIndexerLayer requires quant_block_size == 128");

  n_heads_inv_sqrt_ = 1.0f / std::sqrt(static_cast<float>(n_heads_));
  softmax_scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  // Initialize RoPE
  PositionEncoding position_encoding = std::any_cast<PositionEncoding>(parameters[param_idx++]);
  if (position_encoding != PositionEncoding::ROPE) {
    KLLM_THROW(fmt::format("FlashSparseMlaIndexer only supports ROPE, got {}", position_encoding));
  }

  // Initialize RoPE using YARN scaling (only for DeepSeek models)
  float rope_theta = std::any_cast<float>(parameters[param_idx++]);
  void* cos_sin_cache_ptr = std::any_cast<void*>(parameters[param_idx++]);
  auto rope_scaling_factor_config = std::any_cast<RoPEScalingFactor>(parameters[param_idx++]);
  KLLM_CHECK_WITH_INFO(
      rope_scaling_factor_config.type == "yarn" || rope_scaling_factor_config.type == "default",
      fmt::format("FlashSparseMlaIndexer only supports yarn/default, got {}", rope_scaling_factor_config.type));
  param_idx++;  // max_batch_size unused
  const int max_position_embeddings = std::any_cast<int>(parameters[param_idx++]);
  layer_index_ = std::any_cast<int>(parameters[param_idx++]);
  layer_num_ = std::any_cast<int>(parameters[param_idx++]);
  is_neox_ = !(std::any_cast<bool>(parameters[param_idx++]));
  scale_ue8m0_ = (std::any_cast<std::string>(parameters[param_idx++]) == "ue8m0");

  compress_ratio_ = static_cast<size_t>(std::any_cast<uint32_t>(parameters[param_idx++]));
  KLLM_CHECK_WITH_INFO(compress_ratio_ == 1 || compress_ratio_ == 4,
                       "FlashSparseMlaIndexerLayer requires compress_ratio in {1, 4}");
  max_seq_len_ = runtime_config.max_seq_len / compress_ratio_;
  KLLM_CHECK_WITH_INFO(block_size_ % compress_ratio_ == 0, "block_size must be divisible by compress_ratio");
  block_size_ /= compress_ratio_;
  KLLM_CHECK(block_size_ > 0 && (block_size_ & (block_size_ - 1)) == 0);
  KLLM_CHECK_WITH_INFO(block_size_ == 64, "FlashSparseMlaIndexerLayer requires block_size == 64");
  page_bits_ = __builtin_ctz(block_size_);

  max_step_token_num_ = runtime_config.max_step_token_num;  // Save max_step_token_num from runtime_config
  block_stride_ = runtime_config.attn_backend_config.block_size;

  // Initialize TopKLayer for shared topk dispatch
  topk_layer_ = std::make_unique<TopKLayer>();
  topk_layer_->Init({static_cast<uint32_t>(index_topk_), static_cast<uint32_t>(block_size_)}, runtime_config, context,
                    rank);

  // Initialize RoPE based on data type
  DISPATCH_BY_2_DTYPE(inter_data_type_, InitYarnRotaryEmbedding, rotary_embedding_cuda_, rope_scaling_factor_config,
                      cos_sin_cache_ptr, rope_theta, rope_head_dim_, max_position_embeddings, head_dim_, n_heads_,
                      is_neox_, context_->device->GetPrimaryComputeStream().Get());
}

Status FlashSparseMlaIndexerLayer::Forward(const std::vector<Tensor>& input_tensors,
                                           std::vector<Tensor>& output_tensors) {
  return ForwardT<__nv_bfloat16>(input_tensors, output_tensors);
}

template <typename SCALAR_T>
static FlashMlaInputData<SCALAR_T> ParseInputTensors(const std::vector<Tensor>& input_tensors,
                                                     std::shared_ptr<Context> context, int rank) {
  auto input_iter = input_tensors.cbegin();

  // 提取所有引用类型的 tensor
  const Tensor& q_tensor = *input_iter++;                              // query
  const Tensor& k_tensor = *input_iter++;                              // key
  SCALAR_T* weights_ptr = (input_iter++)->GetPtr<SCALAR_T>();          // weights
  const Tensor& quant_workspace_tensor = *input_iter++;                // quant_workspace
  const Tensor& rotary_pos = *input_iter++;                            // rotary pos
  const int* store_out_loc_ptr = (input_iter++)->GetPtr<int>();        // store_out_loc
  const Tensor& extend_lens_tensor = *input_iter++;                    // extend_lens
  const Tensor& seq_len_offset_tensor = *input_iter++;                 // dp_input_offset (seq_len)
  const Tensor* block_table_tensor = &(*input_iter++);                 // block_table
  const Tensor* cur_seq_len_start_tensor = &(*input_iter++);           // cur_seq_len_start
  const Tensor* cur_seq_len_end_tensor = &(*input_iter++);             // cur_seq_len_end
  const Tensor* layer_indexer_kv_cache_ptr_tensor = &(*input_iter++);  // layer_indexer_kv_cache_ptr
  const Tensor* forward_shape_tensor = &(*input_iter++);               // forward_shape

  // 使用聚合初始化创建对象
  FlashMlaInputData<SCALAR_T> data{.q_ptr = q_tensor.GetPtr<SCALAR_T>(),
                                   .k_ptr = k_tensor.GetPtr<SCALAR_T>(),
                                   .weights_ptr = weights_ptr,
                                   .quant_workspace_ptr = quant_workspace_tensor.GetPtr<uint8_t>(),
                                   .rotary_pos = rotary_pos,
                                   .store_out_loc_ptr = store_out_loc_ptr,
                                   .extend_lens_tensor = extend_lens_tensor,
                                   .seq_len_offset_tensor = seq_len_offset_tensor,
                                   .block_table_tensor = block_table_tensor,
                                   .cur_seq_len_start_tensor = cur_seq_len_start_tensor,
                                   .cur_seq_len_end_tensor = cur_seq_len_end_tensor,
                                   .layer_indexer_kv_cache_ptr_tensor = layer_indexer_kv_cache_ptr_tensor,
                                   .total_q_tokens = q_tensor.shape[0],
                                   .total_k_tokens = k_tensor.shape[0],
                                   .batch_size = forward_shape_tensor->shape[7],
                                   .total_prefix_tokens = forward_shape_tensor->shape[11],
                                   .total_tokens = forward_shape_tensor->shape[11] + q_tensor.shape[0],
                                   .stream = context->device->GetPrimaryComputeStream().Get()};

  return data;
}

static void PrepareQuantWorkspace(size_t total_q_tokens, size_t total_tokens, FlashQuantWorkspace& workspace,
                                  uint8_t* quant_workspace_ptr, float* logits_workspace_ptr, int n_heads, int head_dim,
                                  int quant_block_size) {
  // Calculate FP8 quantization buffer sizes
  size_t q_fp8_bytes = total_q_tokens * n_heads * head_dim;
  size_t q_scale_groups = DivRoundUp(head_dim, quant_block_size);
  KLLM_CHECK_WITH_INFO(q_scale_groups == 1,
                       fmt::format("q_scale_groups must be 1 for weight scaling, got {}", q_scale_groups));
  size_t q_scale_bytes = total_q_tokens * n_heads * q_scale_groups * sizeof(float);
  // K/V buffers need to accommodate all tokens (new + prefix)
  size_t k_fp8_bytes = total_tokens * head_dim;

  constexpr int block_kv = 256;
  const size_t aligned_seq_len_kv = RoundUp(total_tokens + block_kv, 4);

  // Setup workspace pointers
  workspace.q_fp8_ptr = quant_workspace_ptr;
  workspace.q_scale_ptr = reinterpret_cast<float*>(quant_workspace_ptr + q_fp8_bytes);
  workspace.k_fp8_ptr = quant_workspace_ptr + q_fp8_bytes + q_scale_bytes;
  workspace.k_scale_ptr = reinterpret_cast<float*>(quant_workspace_ptr + q_fp8_bytes + q_scale_bytes + k_fp8_bytes);
  workspace.logits_ptr = logits_workspace_ptr;
  workspace.aligned_seq_len_kv = aligned_seq_len_kv;
}

template <typename SCALAR_T>
static Status StoreAndGatherKV(SCALAR_T* k_ptr, const FlashMlaInputData<SCALAR_T>& input_data,
                               FlashQuantWorkspace& workspace, int layer_index, int block_size, int head_dim,
                               int page_bits, int block_stride, bool context_use_dsa, int compress_ratio) {
  // For compress_ratio == 1, we still need to store new K to cache here (no external compressor).
  // For compress_ratio > 1, the compressor already stored K via fused_norm_rope_store_indexer.
  if (compress_ratio == 1 && input_data.total_k_tokens > 0) {
    DISPATCH_STORE_BY_PAGE_BITS(
        page_bits, llm_kernels::nvidia::fused_store_indexer_cache, k_ptr,
        input_data.layer_indexer_kv_cache_ptr_tensor->template GetPtr<void*>()[1 + layer_index * 2],
        input_data.store_out_loc_ptr, input_data.total_k_tokens, block_stride, input_data.stream);
  }

  if (!context_use_dsa) {  // No attention needed, nothing to gather
    return Status();
  }

  // Gather k_fp8 and k_scale from cache (unified path for all compress_ratios, with or without prefix cache)
  KLLM_CHECK_WITH_INFO(input_data.total_prefix_tokens % static_cast<size_t>(compress_ratio) == 0,
                       "FlashSparseMlaIndexerLayer requires prefix_len to be a multiple of compress_ratio");
  const size_t prefix_compressed = input_data.total_prefix_tokens / static_cast<size_t>(compress_ratio);
  const uint32_t total_gather_tokens = static_cast<uint32_t>(input_data.total_k_tokens + prefix_compressed);

  if (total_gather_tokens > 0) {
    void* cache = input_data.layer_indexer_kv_cache_ptr_tensor->template GetPtr<void*>()[1 + layer_index * 2];
    const int32_t max_num_blocks_per_seq =
        static_cast<int32_t>(input_data.block_table_tensor->GetElementNumber() / input_data.batch_size);

    DISPATCH_STORE_BY_PAGE_BITS(page_bits, llm_kernels::nvidia::fused_gather_indexer_cache, workspace.k_fp8_ptr,
                                workspace.k_scale_ptr, cache, input_data.block_table_tensor->template GetPtr<void>(),
                                input_data.seq_len_offset_tensor.template GetPtr<size_t>(),
                                static_cast<uint32_t>(input_data.batch_size), total_gather_tokens,
                                static_cast<int64_t>(block_stride), max_num_blocks_per_seq, compress_ratio,
                                input_data.stream);
  }

  return Status();
}

template <typename SCALAR_T>
static Status ComputeMqaLogits(const FlashMlaInputData<SCALAR_T>& input_data, FlashQuantWorkspace& workspace, int rank,
                               int n_heads, int head_dim, size_t chunk_start, size_t chunk_size, int compress_ratio) {
  constexpr int64_t seq_len_alignment = 4;
  const int64_t aligned_seq_len = RoundUp(chunk_size, seq_len_alignment);
  // Reverse-copy 输出按"压缩后 entry"维度展开:
  //   seq_len_kv = total_k_tokens + total_prefix_tokens / compress_ratio
  const int seq_len_kv = static_cast<int>(input_data.total_k_tokens +
                                          input_data.total_prefix_tokens / static_cast<size_t>(compress_ratio));

  const auto device = torch::Device(torch::kCUDA, rank);

  // Create torch tensors from workspace pointers
  auto q_torch = torch::from_blob(
      const_cast<uint8_t*>(workspace.q_fp8_ptr + chunk_start * n_heads * head_dim),
      {static_cast<int64_t>(chunk_size), static_cast<int64_t>(n_heads), static_cast<int64_t>(head_dim)},
      torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device));

  auto weights_torch = torch::from_blob(workspace.q_scale_ptr + chunk_start * n_heads,
                                        {static_cast<int64_t>(chunk_size), static_cast<int64_t>(n_heads)},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(device));

  auto cur_seq_len_start_torch = torch::from_blob(
      const_cast<int32_t*>(input_data.cur_seq_len_start_tensor->template GetPtr<int32_t>() + chunk_start),
      {static_cast<int64_t>(chunk_size)}, torch::TensorOptions().dtype(torch::kInt32).device(device));

  auto cur_seq_len_end_torch = torch::from_blob(
      const_cast<int32_t*>(input_data.cur_seq_len_end_tensor->template GetPtr<int32_t>() + chunk_start),
      {static_cast<int64_t>(chunk_size)}, torch::TensorOptions().dtype(torch::kInt32).device(device));

  auto logits_torch_aligned = torch::from_blob(
      workspace.logits_ptr,
      std::vector<int64_t>{static_cast<int64_t>(aligned_seq_len), static_cast<int64_t>(workspace.aligned_seq_len_kv)},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));

  auto k_fp8_torch = torch::from_blob(const_cast<uint8_t*>(workspace.k_fp8_ptr),
                                      {static_cast<int64_t>(seq_len_kv), static_cast<int64_t>(head_dim)},
                                      torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device));

  auto k_scale_torch = torch::from_blob(workspace.k_scale_ptr, {static_cast<int64_t>(seq_len_kv)},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(device));

  auto kv_fp8 = std::make_pair(k_fp8_torch, k_scale_torch);

  // Call DeepGEMM wrapper
  auto wrapper = ksana_llm::nvidia::DeepSeekDeepGEMMWrapper::GetInstance();
  wrapper->Fp8MqaLogits(q_torch, kv_fp8, weights_torch, cur_seq_len_start_torch, cur_seq_len_end_torch,
                        logits_torch_aligned, input_data.stream,
                        /*clean_logits=*/false);

  return Status();
}

template <typename SCALAR_T>
static Status SelectTopK(BaseLayer* topk_layer, const FlashMlaInputData<SCALAR_T>& input_data,
                         FlashQuantWorkspace& workspace, Tensor& topk_indices, int index_topk, size_t chunk_start,
                         size_t chunk_size) {
  Tensor scores(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, {chunk_size, workspace.aligned_seq_len_kv},
                workspace.logits_ptr);

  Tensor row_starts_chunk = input_data.cur_seq_len_start_tensor->GetView({chunk_size}, chunk_start);
  Tensor row_ends_chunk = input_data.cur_seq_len_end_tensor->GetView({chunk_size}, chunk_start);
  Tensor topk_indices_chunk =
      topk_indices.GetView({chunk_size, static_cast<size_t>(index_topk)}, chunk_start * index_topk);

  std::vector<Tensor> outputs = {topk_indices_chunk};
  return topk_layer->Forward(
      {scores, row_starts_chunk, row_ends_chunk, *input_data.block_table_tensor, input_data.extend_lens_tensor},
      outputs);
}

template <typename SCALAR_T>
Status FlashSparseMlaIndexerLayer::ForwardT(const std::vector<Tensor>& input_tensors,
                                            std::vector<Tensor>& output_tensors) {
  // Step 1: Parse input tensors
  FlashMlaInputData<SCALAR_T> input_data = ParseInputTensors<SCALAR_T>(input_tensors, context_, rank_);
  Tensor& topk_indices = output_tensors[0];

  // When context_use_dsa is false, only the KV needs to be stored
  const bool context_use_dsa = (input_data.weights_ptr != nullptr);

  // Step 2: Prepare workspace
  FlashQuantWorkspace workspace;
  PrepareQuantWorkspace(input_data.total_q_tokens, input_data.total_k_tokens + input_data.total_prefix_tokens,
                        workspace, input_data.quant_workspace_ptr, workspace_buffer_->GetPtr<float>(), n_heads_,
                        head_dim_, quant_block_size_);

  // Step 3: DSA indexer only — apply RoPE to K
  if (compress_ratio_ == 1) {
    const int64_t k_stride_batch = static_cast<int64_t>(head_dim_) * sizeof(SCALAR_T);
    const int64_t k_stride_head = k_stride_batch;

    llm_kernels::nvidia::fused_qk_rope(
        /*q=*/nullptr, /*k=*/input_data.k_ptr,  // rope_dim comes first
        /*freqs_cis=*/rotary_embedding_cuda_->cos_sin_cache_,
        /*positions=*/input_data.rotary_pos.template GetPtr<int64_t>(),
        /*q_stride_batch=*/0, k_stride_batch, /*q_stride_head=*/0, k_stride_head,
        /*num_q_heads=*/0u,
        /*num_k_heads=*/1u,
        /*batch_size=*/static_cast<uint32_t>(input_data.total_q_tokens), input_data.stream, /*inverse=*/false,
        is_neox_);
  }

  // Step 4: Store (compress_ratio==1 only) and gather k_fp8/k_scale from cache
  StoreAndGatherKV<SCALAR_T>(input_data.k_ptr, input_data, workspace, layer_index_, block_size_, head_dim_, page_bits_,
                             block_stride_, context_use_dsa, static_cast<int>(compress_ratio_));

  if (context_use_dsa) {
    // Step 5: Fused RoPE + FP8 quantization + weight scaling for Q
    // rope_offset: DSA has [rope, nope] layout (offset=0), C4 has [nope, rope] layout (offset=64)
    const uint32_t rope_offset = (compress_ratio_ == 1) ? 0u : static_cast<uint32_t>(head_dim_ - rope_head_dim_);

    llm_kernels::nvidia::fused_rope_quant_scale(
        /*q=*/input_data.q_ptr,
        /*weights=*/input_data.weights_ptr,
        /*freqs_cis=*/rotary_embedding_cuda_->cos_sin_cache_,
        /*positions=*/input_data.rotary_pos.template GetPtr<int64_t>(),
        /*output_q_fp8=*/workspace.q_fp8_ptr,
        /*output_scale=*/workspace.q_scale_ptr, n_heads_inv_sqrt_, softmax_scale_, static_cast<uint32_t>(n_heads_),
        static_cast<uint32_t>(input_data.total_q_tokens), rope_offset, input_data.stream, is_neox_);

    // Step 6: Process in chunks: [chunk_start, chunk_start + current_chunk_size)
    for (size_t chunk_start = 0; chunk_start < input_data.total_q_tokens; chunk_start += chunk_size_) {
      const size_t current_chunk_size = std::min(chunk_size_, input_data.total_q_tokens - chunk_start);

      // Step 6.1: Compute MQA logits
      ComputeMqaLogits<SCALAR_T>(input_data, workspace, rank_, n_heads_, head_dim_, chunk_start, current_chunk_size,
                                 static_cast<int>(compress_ratio_));

      // Step 6.2: TopK selection
      SelectTopK<SCALAR_T>(topk_layer_.get(), input_data, workspace, topk_indices, index_topk_, chunk_start,
                           current_chunk_size);
    }
  }

  return Status();
}

size_t FlashSparseMlaIndexerLayer::GetWorkspaceSize() {
  // Calculate workspace size needed for logits (with alignment)
  // Layout: [logits(aligned to 256 bytes)]
  // Set chunk_size_ to roughly cap the logits buffer size at 4 GB
  constexpr size_t kMaxLogitsBytes = 4ul << 30;
  chunk_size_ = kMaxLogitsBytes / max_seq_len_ / sizeof(float);
  const size_t max_total_tokens = std::min(max_step_token_num_, chunk_size_);

  // Logits data: [aligned_seq_len, aligned_seq_len_kv] * sizeof(float), aligned to 256 bytes
  constexpr int block_kv = 256;
  constexpr int seq_len_alignment = 4;
  const size_t aligned_seq_len = RoundUp(max_total_tokens, seq_len_alignment);
  const size_t aligned_seq_len_kv = RoundUp(max_seq_len_ + block_kv, 4);
  const size_t logits_bytes = aligned_seq_len * aligned_seq_len_kv * sizeof(float);

  KLLM_LOG_DEBUG << fmt::format("FlashSparseMlaIndexerLayer GetWorkspaceSize: max_total_tokens={}, logits={}",
                                max_total_tokens, logits_bytes);

  return logits_bytes;
}

}  // namespace ksana_llm
