/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/paged_sparse_mla_indexer_layer.h"

#include <fmt/core.h>
#include <torch/torch.h>

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/fused_rope_quant_scale.h"
#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/rope.h"
#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/store.h"
#include "kernels/nvidia/deepseek_deepgemm_wrapper.h"
#include "layers/attention_layer.h"  // For yarn mscale functions and InitYarnRotaryEmbedding
#include "layers/topk_layer.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"  // For DivRoundUp

namespace ksana_llm {

// 解析输入张量的辅助结构
template <typename SCALAR_T>
struct PagedMlaInputData {
  SCALAR_T* q_ptr;
  SCALAR_T* k_ptr;
  SCALAR_T* weights_ptr;
  uint8_t* quant_workspace_ptr;
  const Tensor& rotary_pos;
  const int* store_out_loc_ptr;
  const Tensor& input_length_tensor;
  const Tensor& expanded_input_length_tensor;
  const Tensor* block_table_tensor;
  const Tensor* expanded_block_table_tensor;
  const Tensor* cu_seq_len_start_tensor;
  const Tensor* cu_seq_len_end_tensor;
  const Tensor* layer_indexer_kv_cache_ptr_tensor;
  const Tensor& schedule_meta_tensor;

  size_t total_q_tokens;  // new tokens without prefix
  size_t total_k_tokens;  // k may be compressed
  size_t batch_size;
  size_t max_compressed_seq_len;  // max(seq_len / compress_ratio) across decode requests in this group
  cudaStream_t stream;
};

// FP8 量化工作空间布局
struct PagedQuantWorkspace {
  uint8_t* q_fp8_ptr;
  float* q_scale_ptr;
  float* logits_ptr;
  uint8_t* topkv2_ws_ptr;
  uint8_t* topkv2_metadata_ptr;
  size_t aligned_max_seq_len;
};

Status PagedSparseMlaIndexerLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                        std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  KLLM_CHECK_WITH_INFO(
      inter_data_type_ == DataType::TYPE_BF16,
      fmt::format("PagedSparseMlaIndexerLayer: unsupported inter_data_type {}", GetTypeString(inter_data_type_)));

  // Parse parameters
  int param_idx = 0;
  n_heads_ = std::any_cast<int>(parameters[param_idx++]);
  head_dim_ = std::any_cast<int>(parameters[param_idx++]);
  rope_head_dim_ = std::any_cast<int>(parameters[param_idx++]);
  index_topk_ = std::any_cast<int>(parameters[param_idx++]);
  KLLM_CHECK_WITH_INFO(index_topk_ == 512 || index_topk_ == 1024 || index_topk_ == 2048,
                       "PagedSparseMlaIndexerLayer requires index_topk in {512, 1024, 2048}");
  block_size_ = std::any_cast<int>(parameters[param_idx++]);
  const int quant_block_size = std::any_cast<int>(parameters[param_idx++]);
  KLLM_CHECK_WITH_INFO(quant_block_size == 128, "PagedSparseMlaIndexerLayer requires quant_block_size == 128");

  n_heads_inv_sqrt_ = 1.0f / std::sqrt(static_cast<float>(n_heads_));
  softmax_scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  // Initialize RoPE
  PositionEncoding position_encoding = std::any_cast<PositionEncoding>(parameters[param_idx++]);
  if (position_encoding != PositionEncoding::ROPE) {
    KLLM_THROW(fmt::format("PagedSparseMlaIndexer only supports ROPE, got {}", position_encoding));
  }

  // Initialize RoPE using YARN scaling (only for DeepSeek models)
  float rope_theta = std::any_cast<float>(parameters[param_idx++]);
  void* cos_sin_cache_ptr = std::any_cast<void*>(parameters[param_idx++]);
  auto rope_scaling_factor_config = std::any_cast<RoPEScalingFactor>(parameters[param_idx++]);
  KLLM_CHECK_WITH_INFO(
      rope_scaling_factor_config.type == "yarn" || rope_scaling_factor_config.type == "default",
      fmt::format("FlashSparseMlaIndexer only supports yarn/default, got {}", rope_scaling_factor_config.type));
  max_batch_size_ = std::any_cast<int>(parameters[param_idx++]);
  int max_position_embeddings = std::any_cast<int>(parameters[param_idx++]);
  layer_index_ = std::any_cast<int>(parameters[param_idx++]);  // Add layer_index initialization
  layer_num_ = std::any_cast<int>(parameters[param_idx++]);
  is_neox_ = !(std::any_cast<bool>(parameters[param_idx++]));
  scale_ue8m0_ = (std::any_cast<std::string>(parameters[param_idx++]) == "ue8m0");

  compress_ratio_ = static_cast<size_t>(std::any_cast<uint32_t>(parameters[param_idx++]));
  KLLM_CHECK_WITH_INFO(compress_ratio_ == 1 || compress_ratio_ == 4,
                       "PagedSparseMlaIndexerLayer requires compress_ratio in {1, 4}");
  max_seq_len_ = runtime_config.max_seq_len / compress_ratio_;
  KLLM_CHECK_WITH_INFO(block_size_ % compress_ratio_ == 0, "block_size must be divisible by compress_ratio");
  block_size_ /= compress_ratio_;
  KLLM_CHECK(block_size_ > 0 && (block_size_ & (block_size_ - 1)) == 0);
  KLLM_CHECK_WITH_INFO(block_size_ == 64, "PagedSparseMlaIndexerLayer requires block_size == 64");
  page_bits_ = __builtin_ctz(block_size_);

  // Paged only params
  max_decode_tokens_per_req_ = std::any_cast<int>(parameters[param_idx++]);
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

Status PagedSparseMlaIndexerLayer::Forward(const std::vector<Tensor>& input_tensors,
                                           std::vector<Tensor>& output_tensors,
                         const LayerExecContext& exec_ctx) {
  return ForwardT<__nv_bfloat16>(input_tensors, output_tensors);
}

template <typename SCALAR_T>
static PagedMlaInputData<SCALAR_T> ParseInputTensors(const std::vector<Tensor>& input_tensors,
                                                     std::shared_ptr<Context> context, int rank) {
  auto input_iter = input_tensors.cbegin();

  // 提取所有引用类型的 tensor
  const Tensor& q_tensor = *input_iter++;                              // query
  const Tensor& k_tensor = *input_iter++;                              // key
  SCALAR_T* weights_ptr = (input_iter++)->GetPtr<SCALAR_T>();          // weights
  const Tensor& quant_workspace_tensor = *input_iter++;                // quant_workspace
  const Tensor& rotary_pos = *input_iter++;                            // rotary pos
  const int* store_out_loc_ptr = (input_iter++)->GetPtr<int>();        // store_out_loc
  const Tensor& input_length_tensor = *input_iter++;                   // input_length
  const Tensor& expanded_input_length_tensor = *input_iter++;          // expanded_input_length
  const Tensor* block_table_tensor = &(*input_iter++);                 // block_table
  const Tensor* expanded_block_table_tensor = &(*input_iter++);        // expanded_block_table
  const Tensor* cu_seq_len_start_tensor = &(*input_iter++);            // cu_seq_len_start
  const Tensor* cu_seq_len_end_tensor = &(*input_iter++);              // cu_seq_len_end
  const Tensor* layer_indexer_kv_cache_ptr_tensor = &(*input_iter++);  // layer_indexer_kv_cache_ptr
  const Tensor& schedule_meta_tensor = *input_iter++;                  // schedule_meta
  const size_t max_compressed_seq_len =
      static_cast<size_t>((input_iter++)->template GetPtr<int32_t>()[0]);  // max_compressed_seq_len (HOST INT32)

  // 使用聚合初始化创建对象
  PagedMlaInputData<SCALAR_T> data{.q_ptr = q_tensor.GetPtr<SCALAR_T>(),
                                   .k_ptr = k_tensor.GetPtr<SCALAR_T>(),
                                   .weights_ptr = weights_ptr,
                                   .quant_workspace_ptr = quant_workspace_tensor.GetPtr<uint8_t>(),
                                   .rotary_pos = rotary_pos,
                                   .store_out_loc_ptr = store_out_loc_ptr,
                                   .input_length_tensor = input_length_tensor,
                                   .expanded_input_length_tensor = expanded_input_length_tensor,
                                   .block_table_tensor = block_table_tensor,
                                   .expanded_block_table_tensor = expanded_block_table_tensor,
                                   .cu_seq_len_start_tensor = cu_seq_len_start_tensor,
                                   .cu_seq_len_end_tensor = cu_seq_len_end_tensor,
                                   .layer_indexer_kv_cache_ptr_tensor = layer_indexer_kv_cache_ptr_tensor,
                                   .schedule_meta_tensor = schedule_meta_tensor,
                                   .total_q_tokens = q_tensor.shape[0],
                                   .total_k_tokens = k_tensor.shape[0],
                                   .batch_size = input_length_tensor.shape[0],
                                   .max_compressed_seq_len = max_compressed_seq_len,
                                   .stream = context->device->GetPrimaryComputeStream().Get()};

  return data;
}

static void PrepareQuantWorkspace(size_t total_q_tokens, PagedQuantWorkspace& workspace, uint8_t* quant_workspace_ptr,
                                  float* layer_workspace_ptr, int n_heads, int head_dim, int max_seq_len,
                                  int block_size) {
  // Calculate FP8 quantization buffer sizes
  size_t q_fp8_bytes = total_q_tokens * n_heads * head_dim;

  constexpr int num_math_warp_groups = 4;
  const size_t aligned_max_seq_len =
      RoundUp(static_cast<size_t>(max_seq_len), static_cast<size_t>(num_math_warp_groups * block_size));

  // Setup workspace pointers
  workspace.topkv2_ws_ptr = quant_workspace_ptr;
  workspace.q_fp8_ptr = quant_workspace_ptr;
  workspace.q_scale_ptr = reinterpret_cast<float*>(quant_workspace_ptr + q_fp8_bytes);
  uint8_t* layer_base = reinterpret_cast<uint8_t*>(layer_workspace_ptr);
  workspace.topkv2_metadata_ptr = layer_base;
  size_t topkv2_metadata_bytes = (total_q_tokens + 1) * kTopkV2MetadataIntsPerBatch * sizeof(int32_t);
  workspace.logits_ptr = reinterpret_cast<float*>(layer_base + topkv2_metadata_bytes);
  workspace.aligned_max_seq_len = aligned_max_seq_len;
}

template <typename SCALAR_T>
static Status StoreKVCache(SCALAR_T* k_ptr, const PagedMlaInputData<SCALAR_T>& input_data,
                           PagedQuantWorkspace& workspace, int layer_index, int page_bits, int block_stride) {
  // Quant and store K to KV cache if needed
  if (input_data.total_k_tokens > 0) {
    DISPATCH_STORE_BY_PAGE_BITS(
        page_bits, llm_kernels::nvidia::fused_store_indexer_cache, k_ptr,
        /*cache*/ input_data.layer_indexer_kv_cache_ptr_tensor->template GetPtr<void*>()[1 + layer_index * 2],
        input_data.store_out_loc_ptr, input_data.total_k_tokens, block_stride, input_data.stream);
  }

  return Status();
}

template <typename SCALAR_T>
static Status ComputeMqaLogits(const PagedMlaInputData<SCALAR_T>& input_data, PagedQuantWorkspace& workspace, int rank,
                               int n_heads, int head_dim, int block_size, int block_stride, int layer_index,
                               int max_seq_len) {
  const int next_n = input_data.total_q_tokens / input_data.batch_size;
  const Tensor& input_length_tensor =
      next_n > 1 ? input_data.expanded_input_length_tensor : input_data.input_length_tensor;
  const Tensor* block_table_tensor = next_n > ksana_llm::nvidia::kDeepGemmMaxQSeqLen
                                         ? input_data.expanded_block_table_tensor
                                         : input_data.block_table_tensor;
  const int64_t expanded_batch_size = block_table_tensor->shape[0];
  const int64_t expanded_next_n = input_data.total_q_tokens / expanded_batch_size;

  // Prepare 2D context_lens tensor
  auto context_lens_torch = torch::from_blob(const_cast<void*>(input_length_tensor.template GetPtr<void>()),
                                             {expanded_batch_size, expanded_next_n},
                                             torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, rank));

  // Prepare schedule_meta tensor
  auto schedule_meta_torch =
      torch::from_blob(const_cast<void*>(input_data.schedule_meta_tensor.template GetPtr<void>()),
                       {static_cast<int64_t>(input_data.schedule_meta_tensor.shape[0]),
                        static_cast<int64_t>(input_data.schedule_meta_tensor.shape[1])},
                       torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, rank));

  // Prepare Q tensor: reshape [total_q_tokens, n_heads, head_dim] to [batch_size, next_n, n_heads, head_dim]
  auto q_torch = torch::from_blob(workspace.q_fp8_ptr, {expanded_batch_size, expanded_next_n, n_heads, head_dim},
                                  torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA, rank));

  // Prepare fused_kv_cache tensor
  const int num_heads_kv = 1;
  const int head_dim_with_sf = head_dim + sizeof(float);
  const int num_kv_blocks = input_data.layer_indexer_kv_cache_ptr_tensor->template GetPtr<int64_t>()[0];

  auto fused_kv_cache_torch =
      torch::from_blob(input_data.layer_indexer_kv_cache_ptr_tensor->template GetPtr<void*>()[1 + layer_index * 2],
                       {num_kv_blocks, block_size, num_heads_kv, head_dim_with_sf},
                       {block_stride, num_heads_kv * head_dim_with_sf, head_dim_with_sf, 1},
                       torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA, rank));

  // Prepare weights tensor (combined scale from fused_rope_quant_scale)
  auto weights_torch =
      torch::from_blob(workspace.q_scale_ptr, {static_cast<int64_t>(input_data.total_q_tokens), n_heads},
                       torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, rank));

  // Prepare block_table tensor
  auto block_table_torch = torch::from_blob(block_table_tensor->template GetPtr<int>(),
                                            {expanded_batch_size, static_cast<int64_t>(block_table_tensor->shape[1])},
                                            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, rank));

  // Prepare logits tensor — use dynamic max_compressed_seq_len instead of static max
  constexpr int num_math_warp_groups = 4;
  const size_t dynamic_aligned_max_seq_len =
      RoundUp(input_data.max_compressed_seq_len, static_cast<size_t>(num_math_warp_groups * block_size));
  auto logits_torch = torch::from_blob(
      workspace.logits_ptr,
      {static_cast<int64_t>(input_data.total_q_tokens), static_cast<int64_t>(dynamic_aligned_max_seq_len)},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, rank));

  auto deepseek_deepgemm_wrapper = ksana_llm::nvidia::DeepSeekDeepGEMMWrapper::GetInstance();

  // Initialize schedule_meta according to init flag (shape[2])
  if (!input_data.schedule_meta_tensor.shape[2]) {
    deepseek_deepgemm_wrapper->PagedMqaLogitsMetadata(context_lens_torch, schedule_meta_torch, expanded_batch_size,
                                                      block_size, input_data.stream);
  }

  // Call DeepGEMM MQA
  deepseek_deepgemm_wrapper->Fp8PagedMqaLogits(q_torch, fused_kv_cache_torch, weights_torch, context_lens_torch,
                                               block_table_torch, schedule_meta_torch, logits_torch,
                                               dynamic_aligned_max_seq_len, input_data.stream);

  return Status();
}

template <typename SCALAR_T>
static Status SelectTopK(BaseLayer* topk_layer, const PagedMlaInputData<SCALAR_T>& input_data,
                         PagedQuantWorkspace& workspace, Tensor& topk_indices, int index_topk, int block_size) {
  constexpr int num_math_warp_groups = 4;
  const size_t dynamic_aligned_max_seq_len =
      RoundUp(input_data.max_compressed_seq_len, static_cast<size_t>(num_math_warp_groups * block_size));
  Tensor scores(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32,
                {input_data.total_q_tokens, dynamic_aligned_max_seq_len}, workspace.logits_ptr);
  Tensor topkv2_workspace(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_INT32,
                          {input_data.total_q_tokens, kWrokspaceIntsPerBatch}, workspace.topkv2_ws_ptr);
  Tensor topkv2_metadata(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_INT32,
                         {input_data.total_q_tokens + 1, kTopkV2MetadataIntsPerBatch}, workspace.topkv2_metadata_ptr);

  std::vector<Tensor> outputs = {topk_indices};
  return topk_layer->Forward({scores, *input_data.cu_seq_len_start_tensor, *input_data.cu_seq_len_end_tensor,
                              *input_data.block_table_tensor, topkv2_workspace, topkv2_metadata},
                             outputs);
}

template <typename SCALAR_T>
Status PagedSparseMlaIndexerLayer::ForwardT(const std::vector<Tensor>& input_tensors,
                                            std::vector<Tensor>& output_tensors) {
  // Step 1: Parse input tensors
  PagedMlaInputData<SCALAR_T> input_data = ParseInputTensors<SCALAR_T>(input_tensors, context_, rank_);
  Tensor& topk_indices = output_tensors[0];

  // Step 2: Prepare workspace
  PagedQuantWorkspace workspace;
  PrepareQuantWorkspace(input_data.total_q_tokens, workspace, input_data.quant_workspace_ptr,
                        workspace_buffer_->GetPtr<float>(), n_heads_, head_dim_, max_seq_len_, block_size_);

  if (compress_ratio_ == 1) {
    // DSA indexer path: additionally apply RoPE to K and store to KV cache.
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

    StoreKVCache<SCALAR_T>(input_data.k_ptr, input_data, workspace, layer_index_, page_bits_, block_stride_);
  }

  // Step 3: Fused RoPE + FP8 quantization + weight scaling for Q (both paths)
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

  // Step 4: Compute MQA logits
  ComputeMqaLogits<SCALAR_T>(input_data, workspace, rank_, n_heads_, head_dim_, block_size_, block_stride_,
                             layer_index_, max_seq_len_);

  // Step 5: TopK selection
  SelectTopK<SCALAR_T>(topk_layer_.get(), input_data, workspace, topk_indices, index_topk_, block_size_);

  return Status();
}

size_t PagedSparseMlaIndexerLayer::GetWorkspaceSize() {
  // Calculate workspace size needed for logits (with alignment)
  // Layout: [logits(aligned to 256 bytes)]
  const size_t max_decode_tokens = max_batch_size_ * max_decode_tokens_per_req_;

  // Logits data: [max_total_tokens, aligned_max_seq_len] * sizeof(float), aligned to 256 bytes
  constexpr int num_math_warp_groups = 4;
  const size_t aligned_max_seq_len =
      RoundUp(static_cast<size_t>(max_seq_len_), static_cast<size_t>(num_math_warp_groups * block_size_));
  const size_t logits_bytes = max_decode_tokens * aligned_max_seq_len * sizeof(float);

  KLLM_LOG_DEBUG << fmt::format("PagedSparseMlaIndexerLayer GetWorkspaceSize: max_decode_tokens={}, logits={}",
                                max_decode_tokens, logits_bytes);

  return logits_bytes;
}

}  // namespace ksana_llm
