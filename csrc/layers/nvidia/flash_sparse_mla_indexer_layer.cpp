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

// Input data assembled from typed params (avoids vector indexing overhead)
template <typename SCALAR_T>
struct FlashMlaInputData {
  SCALAR_T* q_ptr;
  SCALAR_T* k_ptr;
  SCALAR_T* weights_ptr;
  uint8_t* quant_workspace_ptr;
  const Tensor& rotary_pos;
  const int* store_out_loc_ptr;
  const Tensor& seq_len_offset_tensor;  // dp_input_offset: full sequence cumsum
  const Tensor* block_table_tensor;
  const Tensor* cu_seq_len_start_tensor;
  const Tensor* cu_seq_len_end_tensor;
  const Tensor* batch_indices_tensor;
  const Tensor* layer_indexer_kv_cache_ptr_tensor;
  const Tensor* chunk_token_starts_host;            // HOST [num_chunks+1]: Q token boundaries
  const Tensor* chunk_kv_token_ranges_host;         // HOST [2*num_chunks]: KV [start, end) per chunk (pairs)
  const Tensor* chunk_max_compressed_kv_lens_host;  // HOST [num_chunks]: per-chunk max(seq_len/compress_ratio)

  size_t total_q_tokens;  // new tokens without prefix
  size_t total_k_tokens;  // k may be compressed
  size_t batch_size;
  size_t total_prefix_tokens;  // prefix tokens
  size_t total_tokens;         // new tokens + prefix tokens
  cudaStream_t stream;
};

// FP8 quantization workspace layout
struct FlashQuantWorkspace {
  uint8_t* q_fp8_ptr;
  float* q_scale_ptr;
  uint8_t* k_fp8_ptr;
  float* k_scale_ptr;
  float* logits_ptr;
  uint8_t* topkv2_ws_ptr;
  uint8_t* topkv2_metadata_ptr;
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
  const int quant_block_size = std::any_cast<int>(parameters[param_idx++]);
  KLLM_CHECK_WITH_INFO(quant_block_size == 128, "FlashSparseMlaIndexerLayer requires quant_block_size == 128");

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

  max_step_token_num_ = runtime_config.max_step_token_num;        // Save max_step_token_num from runtime_config
  max_chunked_kv_tokens_ = runtime_config.max_chunked_kv_tokens;  // Save for k_fp8/k_scale sizing
  block_stride_ = runtime_config.attn_backend_config.block_size;

  // Initialize TopKLayer for shared topk dispatch
  topk_layer_ = std::make_unique<TopKLayer>();
  topk_layer_->Init({static_cast<uint32_t>(index_topk_), static_cast<uint32_t>(block_size_)}, runtime_config, context,
                    rank);

  // Initialize RoPE based on data type
  DISPATCH_BY_2_DTYPE(inter_data_type_, InitYarnRotaryEmbedding, rotary_embedding_cuda_, rope_scaling_factor_config,
                      cos_sin_cache_ptr, rope_theta, rope_head_dim_, max_seq_len_, head_dim_, n_heads_, is_neox_,
                      context_->device->GetPrimaryComputeStream().Get());
}

static void PrepareQuantWorkspace(size_t total_q_tokens, size_t max_chunked_kv_tokens, size_t max_seq_len,
                                  FlashQuantWorkspace& workspace, uint8_t* quant_workspace_ptr,
                                  float* layer_workspace_ptr, int n_heads, int head_dim) {
  // Calculate FP8 quantization buffer sizes
  size_t q_fp8_bytes = total_q_tokens * n_heads * head_dim;
  // K/V buffers sized to max_chunked_kv_tokens (reused per chunk), placed at front of layer_workspace
  size_t k_fp8_bytes = max_chunked_kv_tokens * head_dim;
  size_t k_scale_bytes = max_chunked_kv_tokens * sizeof(float);

  size_t topkv2_metadata_bytes = (total_q_tokens + 1) * kTopkV2MetadataIntsPerBatch * sizeof(int32_t);
  size_t max_k_topkmeta_bytes = std::max(k_fp8_bytes + k_scale_bytes, topkv2_metadata_bytes);

  // Setup workspace pointers
  workspace.topkv2_ws_ptr = quant_workspace_ptr;
  workspace.q_fp8_ptr = quant_workspace_ptr;
  workspace.q_scale_ptr = reinterpret_cast<float*>(quant_workspace_ptr + q_fp8_bytes);
  uint8_t* layer_base = reinterpret_cast<uint8_t*>(layer_workspace_ptr);
  workspace.topkv2_metadata_ptr = layer_base;
  workspace.k_fp8_ptr = layer_base;
  workspace.k_scale_ptr = reinterpret_cast<float*>(layer_base + k_fp8_bytes);
  workspace.logits_ptr = reinterpret_cast<float*>(layer_base + max_k_topkmeta_bytes);
}

template <typename SCALAR_T>
static Status ComputeMqaLogits(const FlashMlaInputData<SCALAR_T>& input_data, FlashQuantWorkspace& workspace, int rank,
                               int n_heads, int head_dim, size_t chunk_start, size_t chunk_size, int64_t chunk_kv_size,
                               int64_t dynamic_aligned_seq_len_kv) {
  constexpr int kBlockQH = 128;
  const int block_q = kBlockQH / n_heads;
  const int64_t aligned_seq_len = RoundUp(chunk_size, block_q);
  const int64_t seq_len_kv = chunk_kv_size;

  const auto device = torch::Device(torch::kCUDA, rank);

  // Create torch tensors from workspace pointers
  auto q_torch = torch::from_blob(
      const_cast<uint8_t*>(workspace.q_fp8_ptr + chunk_start * n_heads * head_dim),
      {static_cast<int64_t>(chunk_size), static_cast<int64_t>(n_heads), static_cast<int64_t>(head_dim)},
      torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device));

  auto weights_torch = torch::from_blob(workspace.q_scale_ptr + chunk_start * n_heads,
                                        {static_cast<int64_t>(chunk_size), static_cast<int64_t>(n_heads)},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(device));

  auto cu_seq_len_start_torch = torch::from_blob(
      const_cast<int32_t*>(input_data.cu_seq_len_start_tensor->template GetPtr<int32_t>() + chunk_start),
      {static_cast<int64_t>(chunk_size)}, torch::TensorOptions().dtype(torch::kInt32).device(device));

  auto cu_seq_len_end_torch =
      torch::from_blob(const_cast<int32_t*>(input_data.cu_seq_len_end_tensor->template GetPtr<int32_t>() + chunk_start),
                       {static_cast<int64_t>(chunk_size)}, torch::TensorOptions().dtype(torch::kInt32).device(device));

  auto logits_torch_aligned =
      torch::from_blob(workspace.logits_ptr, std::vector<int64_t>{aligned_seq_len, dynamic_aligned_seq_len_kv},
                       torch::TensorOptions().dtype(torch::kFloat32).device(device));

  auto k_fp8_torch =
      torch::from_blob(const_cast<uint8_t*>(workspace.k_fp8_ptr), {seq_len_kv, static_cast<int64_t>(head_dim)},
                       torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device));

  auto k_scale_torch = torch::from_blob(workspace.k_scale_ptr, {seq_len_kv},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(device));

  auto kv_fp8 = std::make_pair(k_fp8_torch, k_scale_torch);

  // Call DeepGEMM wrapper — max_seqlen_k must match logits column count
  auto wrapper = ksana_llm::nvidia::DeepSeekDeepGEMMWrapper::GetInstance();
  wrapper->Fp8MqaLogits(q_torch, kv_fp8, weights_torch, cu_seq_len_start_torch, cu_seq_len_end_torch,
                        logits_torch_aligned, dynamic_aligned_seq_len_kv, input_data.stream);

  return Status();
}

template <typename SCALAR_T>
static Status SelectTopK(BaseLayer* topk_layer, const FlashMlaInputData<SCALAR_T>& input_data,
                         FlashQuantWorkspace& workspace, Tensor& topk_indices, int index_topk, size_t chunk_start,
                         size_t chunk_size, int64_t dynamic_aligned_seq_len_kv) {
  Tensor scores(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32,
                {chunk_size, static_cast<size_t>(dynamic_aligned_seq_len_kv)}, workspace.logits_ptr);
  Tensor topkv2_workspace(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_INT32, {chunk_size, kWrokspaceIntsPerBatch},
                          workspace.topkv2_ws_ptr);
  Tensor topkv2_metadata(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_INT32,
                         {chunk_size + 1, kTopkV2MetadataIntsPerBatch}, workspace.topkv2_metadata_ptr);

  Tensor row_starts_chunk = input_data.cu_seq_len_start_tensor->GetView({chunk_size}, chunk_start);
  Tensor row_ends_chunk = input_data.cu_seq_len_end_tensor->GetView({chunk_size}, chunk_start);
  Tensor topk_indices_chunk =
      topk_indices.GetView({chunk_size, static_cast<size_t>(index_topk)}, chunk_start * index_topk);
  Tensor batch_indices_chunk = input_data.batch_indices_tensor->GetView({chunk_size}, chunk_start);

  auto* typed = static_cast<TopKLayer*>(topk_layer);
  return typed->ForwardDispatch(&scores, &row_starts_chunk, &row_ends_chunk, input_data.block_table_tensor,
                                &topkv2_workspace, &topkv2_metadata, &topk_indices_chunk, &batch_indices_chunk);
}

template <typename SCALAR_T>
Status FlashSparseMlaIndexerLayer::ForwardT(const FlashSparseMlaIndexerParams& p, Tensor* output) {
  // Step 1: Build input data directly from typed params (no vector indexing)
  FlashMlaInputData<SCALAR_T> input_data{
      .q_ptr = p.q->GetPtr<SCALAR_T>(),
      .k_ptr = p.k->GetPtr<SCALAR_T>(),
      .weights_ptr = p.weights->GetPtr<SCALAR_T>(),
      .quant_workspace_ptr = p.quant_workspace->GetPtr<uint8_t>(),
      .rotary_pos = *p.rotary_embedding_pos,
      .store_out_loc_ptr = p.store_out_loc->GetPtr<int>(),
      .seq_len_offset_tensor = *p.seq_len_offset,
      .block_table_tensor = p.block_table,
      .cu_seq_len_start_tensor = p.cu_seq_len_start,
      .cu_seq_len_end_tensor = p.cu_seq_len_end,
      .batch_indices_tensor = p.batch_indices,
      .layer_indexer_kv_cache_ptr_tensor = p.layer_kv_cache_ptr,
      .chunk_token_starts_host = p.chunk_token_starts_host,
      .chunk_kv_token_ranges_host = p.chunk_kv_token_ranges_host,
      .chunk_max_compressed_kv_lens_host = p.chunk_max_compressed_kv_lens_host,
      .total_q_tokens = p.q->shape[0],
      .total_k_tokens = p.k->shape[0],
      .batch_size = static_cast<size_t>(p.batch_size),
      .total_prefix_tokens = static_cast<size_t>(p.total_prefix_tokens),
      .total_tokens = static_cast<size_t>(p.total_prefix_tokens) + p.q->shape[0],
      .stream = context_->device->GetPrimaryComputeStream().Get(),
  };
  Tensor& topk_indices = *output;

  // When context_use_dsa is false, only the KV needs to be stored
  const bool context_use_dsa = (input_data.weights_ptr != nullptr);

  // Step 2: Prepare workspace (K buffers sized to max_chunked_kv_tokens_)
  FlashQuantWorkspace workspace;
  PrepareQuantWorkspace(input_data.total_q_tokens, max_chunked_kv_tokens_, max_seq_len_, workspace,
                        input_data.quant_workspace_ptr, workspace_buffer_->GetPtr<float>(), n_heads_, head_dim_);

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

  // Step 4: Store (compress_ratio==1 only) new K to cache
  if (compress_ratio_ == 1 && input_data.total_k_tokens > 0) {
    DISPATCH_STORE_BY_PAGE_BITS(
        page_bits_, llm_kernels::nvidia::fused_store_indexer_cache, input_data.k_ptr,
        input_data.layer_indexer_kv_cache_ptr_tensor->template GetPtr<void*>()[1 + layer_index_ * 2],
        input_data.store_out_loc_ptr, input_data.total_k_tokens, block_stride_, input_data.stream);
  }

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

    // Step 6: Per-chunk loop — gather KV, compute MQA, select TopK
    const int* token_starts = input_data.chunk_token_starts_host->template GetPtr<int>();
    const int* kv_token_ranges = input_data.chunk_kv_token_ranges_host->template GetPtr<int>();
    const int32_t* chunk_max_kv_lens = input_data.chunk_max_compressed_kv_lens_host->template GetPtr<int32_t>();
    const size_t num_chunks = input_data.chunk_token_starts_host->shape[0] - 1;

    void* cache = input_data.layer_indexer_kv_cache_ptr_tensor->template GetPtr<void*>()[1 + layer_index_ * 2];
    const int32_t max_num_blocks_per_seq =
        static_cast<int32_t>(input_data.block_table_tensor->GetElementNumber() / input_data.batch_size);

    for (size_t c = 0; c < num_chunks; ++c) {
      const size_t chunk_q_start = static_cast<size_t>(token_starts[c]);
      const size_t chunk_q_tokens = token_starts[c + 1] - chunk_q_start;
      const int64_t chunk_kv_tokens = static_cast<int64_t>(kv_token_ranges[2 * c + 1] - kv_token_ranges[2 * c]);

      // 6a. Gather KV for this chunk
      if (chunk_kv_tokens > 0) {
        // kv_token_ranges stores [start, end) pairs: [offset0, end0, offset1, end1, ...]
        const int32_t gather_offset = kv_token_ranges[2 * c];
        DISPATCH_STORE_BY_PAGE_BITS(
            page_bits_, llm_kernels::nvidia::fused_gather_indexer_cache, workspace.k_fp8_ptr, workspace.k_scale_ptr,
            cache, input_data.block_table_tensor->template GetPtr<void>(),
            input_data.seq_len_offset_tensor.template GetPtr<size_t>(), static_cast<uint32_t>(input_data.batch_size),
            chunk_kv_tokens, static_cast<int64_t>(block_stride_), max_num_blocks_per_seq,
            static_cast<int32_t>(compress_ratio_), gather_offset, input_data.stream);
      }

      // 6b. Compute MQA logits
      // Use per-chunk dynamic max compressed kv len (aligned to 256) instead of static max
      constexpr int64_t kBlockKV = 256;
      const int64_t chunk_max_kv = static_cast<int64_t>(chunk_max_kv_lens[c]);
      const int64_t dynamic_aligned_seq_len_kv = RoundUp(chunk_max_kv, kBlockKV);
      ComputeMqaLogits<SCALAR_T>(input_data, workspace, rank_, n_heads_, head_dim_, chunk_q_start, chunk_q_tokens,
                                 chunk_kv_tokens, dynamic_aligned_seq_len_kv);

      // 6c. TopK selection
      SelectTopK<SCALAR_T>(topk_layer_.get(), input_data, workspace, topk_indices, index_topk_, chunk_q_start,
                           chunk_q_tokens, dynamic_aligned_seq_len_kv);
    }
  }

  return Status();
}

size_t FlashSparseMlaIndexerLayer::GetWorkspaceSize() {
  // K buffers at front of layer workspace: [k_fp8 | k_scale | logits]
  size_t k_fp8_bytes = max_chunked_kv_tokens_ * head_dim_;
  size_t k_scale_bytes = max_chunked_kv_tokens_ * sizeof(float);

  // Logits buffer: [aligned_q_chunk, aligned_seq_len_kv] * sizeof(float)
  constexpr int kBlockQH = 128;
  const int block_q = kBlockQH / n_heads_;
  constexpr size_t kBlockKV = 256;
  // Compute max_chunked_q_tokens_ via shared utility (respects env override).
  max_chunked_q_tokens_ = ComputeIndexerMaxQChunkSize(static_cast<size_t>(max_seq_len_), max_step_token_num_);
  const size_t aligned_seq_len = RoundUp(max_chunked_q_tokens_, block_q);
  const size_t aligned_seq_len_kv = RoundUp(static_cast<size_t>(max_seq_len_), kBlockKV);
  const size_t logits_bytes = aligned_seq_len * aligned_seq_len_kv * sizeof(float);

  size_t total_bytes = k_fp8_bytes + k_scale_bytes + logits_bytes;

  KLLM_LOG_DEBUG << fmt::format("FlashSparseMlaIndexerLayer GetWorkspaceSize={}", total_bytes);

  return total_bytes;
}

Status FlashSparseMlaIndexerLayer::Forward(const FlashSparseMlaIndexerParams& p, Tensor* output,
                                           const LayerExecContext& exec_ctx) {
  return ForwardT<__nv_bfloat16>(p, output);
}

}  // namespace ksana_llm
