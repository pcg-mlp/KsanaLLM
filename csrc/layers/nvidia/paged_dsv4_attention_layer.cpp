/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/paged_dsv4_attention_layer.h"

#include "csrc/kernels/nvidia/flash_sparse_mla/flash_sparse_mla.h"
#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/rope.h"
#include "csrc/kernels/nvidia/others/vllm/main/deepseek_v4/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.h"
#include "device/nvidia/cuda_utils.h"
#include "utils/logger.h"

namespace ksana_llm {

Status PagedDsv4AttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                     std::shared_ptr<Context> context, int rank) {
  STATUS_CHECK_RETURN(AttentionLayer::Init(parameters, runtime_config, context, rank));

  KLLM_CHECK_WITH_INFO(num_kv_heads_ == 1, "PagedDsv4AttentionLayer expects num_kv_heads == 1");

  // Compute page_bits = ctz(block_token_num / compress_ratio)
  KLLM_CHECK(block_token_num_ > 0 && (block_token_num_ & (block_token_num_ - 1)) == 0);
  page_bits_ = __builtin_ctz(block_token_num_);
  KLLM_CHECK_WITH_INFO(page_bits_ == 1 || page_bits_ == 6 || page_bits_ == 8,
                       "PagedDsv4AttentionLayer expects page_bits in {1, 6, 8}");
  const uint32_t compress_ratio = std::any_cast<const uint32_t>(parameters.back());
  KLLM_CHECK(compress_ratio == 0 || compress_ratio == 4u || compress_ratio == 128u);
  compressed_block_token_num_ = static_cast<uint32_t>(block_token_num_) / (compress_ratio == 0 ? 1u : compress_ratio);
  KLLM_CHECK(compressed_block_token_num_ > 0 && (compressed_block_token_num_ & (compressed_block_token_num_ - 1)) == 0);

  return Status();
}

Status PagedDsv4AttentionLayer::Forward(const Dsv4AttentionInputs& inputs, Tensor* output) {
// Dispatch by inter_data_type_ and kv_cache_dtype_
#define DISPATCH_PAGED_DSV4_ATTENTION_BY_KVTYPE(dtype, kv_cache_dtype, func, ...)                             \
  switch (kv_cache_dtype) {                                                                                   \
    case TYPE_FP8_DS_MLA:                                                                                     \
      return func<dtype, uint8_t, llm_kernels::utils::KVCacheType::kFp8DsMla>(__VA_ARGS__);                   \
    case TYPE_FP8_E4M3:                                                                                       \
      return func<dtype, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3>(__VA_ARGS__);                    \
    default:                                                                                                  \
      KLLM_THROW(fmt::format("{}: Unsupported KVCacheDtype type: {}.", __PRETTY_FUNCTION__, kv_cache_dtype)); \
  }
  switch (inter_data_type_) {
    case DataType::TYPE_BF16:
      DISPATCH_PAGED_DSV4_ATTENTION_BY_KVTYPE(bfloat16, kv_cache_dtype_, ForwardT, inputs, output);
    default:
      KLLM_THROW(fmt::format("{}: Unsupported Dtype type: {}.", __PRETTY_FUNCTION__, inter_data_type_));
  }
#undef DISPATCH_PAGED_DSV4_ATTENTION_BY_KVTYPE
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status PagedDsv4AttentionLayer::ForwardT(const Dsv4AttentionInputs& inputs, Tensor* output) {
  int64_t* rotary_embedding_pos_ptr = inputs.rotary_embedding_pos->GetPtr<int64_t>();
  const Tensor& q_tensor = *inputs.q;
  void* q_ptr = q_tensor.GetPtr<void>();
  const Tensor& kv_tensor = *inputs.kv;
  void* kv_ptr = kv_tensor.GetPtr<void>();
  const Tensor& layer_swa_kv_cache_ptr_tensor = *inputs.layer_swa_kv_cache;
  const Tensor& tile_scheduler_metadata_tensor = *inputs.tile_scheduler_metadata;
  int* tile_scheduler_metadata_ptr = tile_scheduler_metadata_tensor.GetPtr<int>();
  const Tensor& num_splits_tensor = *inputs.num_splits;
  int* num_splits_ptr = num_splits_tensor.GetPtr<int>();
  float* attn_sink_ptr = inputs.attn_sink->GetPtr<float>();
  int* raw_out_loc_ptr = inputs.raw_out_loc->GetPtr<int>();
  const Tensor& indices_tensor = *inputs.swa_indices;
  int* indices_ptr = indices_tensor.GetPtr<int>();

  SCALAR_T* out_ptr = output->GetPtr<SCALAR_T>();

  const size_t token_num = q_tensor.shape[0];
  const size_t batch_size = num_splits_tensor.shape[0] - 1;
  const size_t q_seq_len = token_num / batch_size;
  // layer_swa_kv_cache_ptr_tensor layout:
  //   slot 0       : device_block_num (int64) — SWA pool 总 block 数
  //   slot 1       : swa_block_size   (int64) — SWA pool 单 block 字节数 (primary stride)
  //   slot 2..2+N-1: 每层 SWA base ptr (void*)
  CACHE_T* const kcache_ptr =
      reinterpret_cast<CACHE_T*>(layer_swa_kv_cache_ptr_tensor.GetPtr<void*>()[2 + this->layer_index_]);
  const int64_t num_blocks = layer_swa_kv_cache_ptr_tensor.GetPtr<int64_t>()[0];
  const int64_t swa_block_size = layer_swa_kv_cache_ptr_tensor.GetPtr<int64_t>()[1];
  const int topk = indices_tensor.shape[1];

  int* extra_indices_ptr = nullptr;
  CACHE_T* extra_kcache_ptr = nullptr;
  int64_t extra_num_blocks = 0;
  int extra_topk = 0;
  int* extra_topk_length_ptr = nullptr;
  // Extra parameters required only when compress_ratio > 0
  if (inputs.layer_extra_kv_cache != nullptr) {
    const Tensor& layer_extra_kv_cache_ptr_tensor = *inputs.layer_extra_kv_cache;
    const Tensor& extra_indices_tensor = *inputs.extra_indices;
    extra_indices_ptr = extra_indices_tensor.GetPtr<int>();
    // extra_topk_length is optional (provided for c128, not for c4)
    if (inputs.extra_topk_length != nullptr) {
      extra_topk_length_ptr = inputs.extra_topk_length->GetPtr<int>();
    }
    extra_kcache_ptr =
        reinterpret_cast<CACHE_T*>(layer_extra_kv_cache_ptr_tensor.GetPtr<void*>()[1 + this->layer_index_ * 2]);
    extra_num_blocks = *(layer_extra_kv_cache_ptr_tensor.GetPtr<int64_t>());
    extra_topk = extra_indices_tensor.shape[1];
  }

  auto stream = this->context_->device->GetPrimaryComputeStream().Get();

  // ── Fused Q-Norm + RoPE + KV Store + Q Expand ────────────────────────────
  constexpr int kMinNumHeads = 64;  // flashmla (fp8_ds_mla) only supports num_heads >= 64
  KLLM_CHECK_WITH_INFO(this->num_heads_ <= kMinNumHeads, "num_heads of dsv4 must be <= 64");
  int num_heads = this->num_heads_;

  uint8_t* workspace_ptr = workspace_buffer_->GetPtr<uint8_t>();
  void* expanded_q_ptr = q_ptr;
  SCALAR_T* expanded_out_ptr = out_ptr;
  if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
    expanded_q_ptr = workspace_ptr;
    const size_t expanded_q_bytes = token_num * kMinNumHeads * this->head_size_;  // fp8 = 1 byte/elem
    workspace_ptr += expanded_q_bytes;
  } else {
    if (this->num_heads_ < kMinNumHeads) {
      num_heads = kMinNumHeads;
      expanded_q_ptr = workspace_ptr;
      const size_t expanded_q_bytes = token_num * kMinNumHeads * this->head_size_ * sizeof(SCALAR_T);
      workspace_ptr += expanded_q_bytes;
      expanded_out_ptr = reinterpret_cast<SCALAR_T*>(workspace_ptr);
      workspace_ptr += expanded_q_bytes;  // expanded_out_bytes == expanded_q_bytes
    }
  }

  KLLM_CHECK_WITH_INFO(this->rotary_embedding_cuda_.has_value(), "rotary_embedding_cuda must be set!");
  llm_kernels::nvidia::InvokeFusedQNormRopeKVInsert<KV_DTYPE>(
      reinterpret_cast<const SCALAR_T*>(q_ptr), reinterpret_cast<const SCALAR_T*>(kv_ptr), expanded_q_ptr, kcache_ptr,
      rotary_embedding_pos_ptr, reinterpret_cast<const SCALAR_T*>(this->rotary_embedding_cuda_->cos_sin_cache_),
      raw_out_loc_ptr, this->layernorm_eps_, token_num, this->num_heads_, swa_block_size, this->page_bits_, stream);

  // Invoke sparse mla kernel to compute the output [token_num, num_heads_q, head_dim]
  // primary stride = swa_block_size (SWA pool); extra stride = main KV block_size_ (compressed KV in main pool)
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeFlashSparseMlaWithKVCache<SCALAR_T, CACHE_T, KV_DTYPE>(
      expanded_q_ptr, kcache_ptr, indices_ptr, tile_scheduler_metadata_ptr, num_splits_ptr,
      /*sched_meta_initialized, shape[2] is the init flag*/ tile_scheduler_metadata_tensor.shape[2] != 0, batch_size,
      q_seq_len, num_heads, this->head_size_, num_blocks, this->block_token_num_, /*num_heads_k*/ 1,
      /*kcache_batch_stride*/ static_cast<int>(swa_block_size), topk, this->head_size_, this->attn_scale_, stream,
      reinterpret_cast<float*>(workspace_ptr), expanded_out_ptr,
      /*topk_length_ptr*/ nullptr, attn_sink_ptr, extra_kcache_ptr, extra_indices_ptr, extra_topk_length_ptr,
      extra_num_blocks, this->compressed_block_token_num_, extra_topk,
      /*extra_kcache_batch_stride*/ this->block_size_));
  if constexpr (KV_DTYPE != llm_kernels::utils::KVCacheType::kFp8E4M3) {
    if (this->num_heads_ < kMinNumHeads) {
      const size_t dst_pitch = this->num_heads_ * this->head_size_ * sizeof(SCALAR_T);
      const size_t src_pitch = kMinNumHeads * this->head_size_ * sizeof(SCALAR_T);
      // Squeeze expanded_out_ptr to out_ptr
      Memcpy2DAsync(out_ptr, dst_pitch, expanded_out_ptr, src_pitch, dst_pitch, token_num, MEMCPY_DEVICE_TO_DEVICE,
                    this->context_->device->GetPrimaryComputeStream());
    }
  }

  // out_ptr [token_num, num_heads, head_dim], apply **inverse** RoPE on [..., -qk_rope_head_dim:]
  SCALAR_T* out_rope_ptr = out_ptr + this->qk_nope_head_dim_;
  llm_kernels::nvidia::fused_qk_rope(
      /*q=*/out_rope_ptr, /*k=*/nullptr,
      /*freqs_cis=*/this->rotary_embedding_cuda_->cos_sin_cache_,
      /*positions=*/rotary_embedding_pos_ptr,
      /*q_stride_batch=*/static_cast<int64_t>(this->num_heads_ * this->head_size_) * sizeof(SCALAR_T),
      /*k_stride_batch=*/0,
      /*q_stride_head=*/static_cast<int64_t>(this->head_size_) * sizeof(SCALAR_T),
      /*k_stride_head=*/0,
      /*num_q_heads=*/static_cast<uint32_t>(this->num_heads_),
      /*num_k_heads=*/0u,
      /*batch_size=*/static_cast<uint32_t>(token_num), stream, /*inverse=*/true);

  return Status();
}

}  // namespace ksana_llm
