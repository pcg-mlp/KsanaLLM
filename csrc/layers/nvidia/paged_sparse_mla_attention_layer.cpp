/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/paged_sparse_mla_attention_layer.h"

#include "csrc/kernels/nvidia/concat/concat.h"
#include "csrc/kernels/nvidia/flash_sparse_mla/flash_sparse_mla.h"
#include "csrc/kernels/nvidia/paged_attention/cache_copy_flash_attn_layout.h"
#include "csrc/kernels/nvidia/paged_attention/mla_cache_copy.h"
#include "device/nvidia/cuda_utils.h"
#include "kernels/nvidia/kernel_wrapper.h"
#include "utils/logger.h"

namespace ksana_llm {

Status PagedSparseMlaAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                          std::shared_ptr<Context> context, int rank) {
  return AttentionLayer::Init(parameters, runtime_config, context, rank);
}

Status PagedSparseMlaAttentionLayer::Forward(const std::vector<Tensor>& input_tensors,
                                             std::vector<Tensor>& output_tensors,
                         const LayerExecContext& exec_ctx) {
#define DISPATCH_PAGED_SPARSE_MLA_BY_KVTYPE(dtype, kv_cache_dtype, func, ...)                                 \
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
      DISPATCH_PAGED_SPARSE_MLA_BY_KVTYPE(bfloat16, kv_cache_dtype_, ForwardT, input_tensors, output_tensors);
    default:
      KLLM_THROW(fmt::format("{}: Unsupported Dtype type: {}.", __PRETTY_FUNCTION__, inter_data_type_));
  }
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status PagedSparseMlaAttentionLayer::ForwardT(const std::vector<Tensor>& input_tensors,
                                              std::vector<Tensor>& output_tensors) {
  auto input_iter = input_tensors.cbegin();
  int64_t* rotary_embedding_pos_ptr = (input_iter++)->GetPtr<int64_t>();
  int64_t* rotary_embedding_mask_ptr = (input_iter++)->GetPtr<int64_t>();
  const Tensor& q_pe_tensor = *input_iter++;
  void* q_pe_ptr = q_pe_tensor.GetPtr<void>();
  void* k_pe_ptr = (input_iter++)->GetPtr<void>();
  void* q_nope_ptr = (input_iter++)->GetPtr<void>();
  void* q_ptr = (input_iter++)->GetPtr<void>();
  SCALAR_T* compressed_kv_ptr = (input_iter++)->GetPtr<SCALAR_T>();
  const Tensor& kv_list_tensor = *input_iter++;
  int* seqlens_k_ptr = (input_iter++)->GetPtr<int>();
  const Tensor& block_offsets_tensor = *input_iter++;
  int* block_offsets_ptr = block_offsets_tensor.GetPtr<int>();
  const Tensor& layer_kv_cache_ptr_tensor = *input_iter++;
  const Tensor& block_table_tensor = *input_iter++;
  int* block_table_ptr = block_table_tensor.GetPtr<int>();
  const Tensor& tile_scheduler_metadata_tensor = *input_iter++;
  int* tile_scheduler_metadata_ptr = tile_scheduler_metadata_tensor.GetPtr<int>();
  int* num_splits_ptr = (input_iter++)->GetPtr<int>();
  const Tensor& indices_tensor = *input_iter++;
  int* indices_ptr = indices_tensor.GetPtr<int>();

  SCALAR_T* out_ptr = output_tensors[0].GetPtr<SCALAR_T>();

  const size_t token_num = q_pe_tensor.shape[0];
  const size_t batch_size = block_offsets_tensor.shape[0] - 1;
  const size_t q_seq_len = token_num / batch_size;
  void** const k_list_ptr = kv_list_tensor.GetPtr<void*>() + this->layer_index_ * kv_list_tensor.shape[1];
  CACHE_T* const kcache_ptr =
      reinterpret_cast<CACHE_T*>(layer_kv_cache_ptr_tensor.GetPtr<void*>()[1 + this->layer_index_ * 2]);
  const int max_num_blocks_per_seq = block_table_tensor.shape[1];
  const int64_t num_blocks = *(layer_kv_cache_ptr_tensor.GetPtr<int64_t>());
  const int topk = indices_tensor.shape[1];

  auto stream = this->context_->device->GetPrimaryComputeStream().Get();

  KLLM_CHECK_WITH_INFO(this->rotary_embedding_cuda_.has_value(), "rotary_embedding_cuda must be set!");
  // Apply rotary to q_pe [token_num, num_heads_q, qk_rope_head_dim] and k_pe [token_num, num_heads_k, qk_rope_head_dim]
  this->rotary_embedding_cuda_->SetInput(rotary_embedding_pos_ptr, rotary_embedding_mask_ptr, q_pe_ptr, k_pe_ptr,
                                         token_num, stream);
  CUDA_CHECK_LAST_ERROR(this->rotary_embedding_cuda_->Forward<SCALAR_T>());

  // Concat q_nope and q_pe to q [token_num, num_heads_q, qk_nope_head_dim + qk_rope_head_dim]
  constexpr size_t kInnerDimSize = 1;
  const size_t outer_q_dim_size = token_num * this->num_heads_;
  Concat<SCALAR_T>(q_nope_ptr, q_pe_ptr, this->kv_lora_rank_, this->qk_rope_head_dim_, outer_q_dim_size, kInnerDimSize,
                   q_ptr, stream);

  // Store compressed_kv [token_num, num_heads_k, kv_lora_rank] and k_pe [token_num, num_heads_k, qk_rope_head_dim] to
  // the kv cache
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaPagedKVCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
      compressed_kv_ptr, reinterpret_cast<SCALAR_T*>(k_pe_ptr), k_list_ptr, seqlens_k_ptr, block_offsets_ptr,
      this->block_token_num_, batch_size, q_seq_len, this->kv_lora_rank_, this->qk_rope_head_dim_, this->k_scale_,
      stream));

  // Convert indices [token_num, index_topk] from positions within sequence into positions within block table
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::PagedSparseMlaConvertBlockTable(indices_ptr, block_table_ptr, q_seq_len,
                                                                             topk, batch_size, max_num_blocks_per_seq,
                                                                             this->block_token_num_, stream));

  constexpr int kMinNumHeads = 64;  // flashmla (fp8_ds_mla) only supports num_heads >= 64
  void* expanded_q_ptr = q_ptr;
  SCALAR_T* expanded_out_ptr = out_ptr;
  int num_heads = this->num_heads_;

  uint8_t* workspace_ptr = workspace_buffer_->GetPtr<uint8_t>();
  const int head_size_q = this->kv_lora_rank_ + this->qk_rope_head_dim_;
  if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
    // Quant bf16 q_ptr into fp8 quant_q_ptr
    void* const quant_q_ptr = workspace_ptr;
    workspace_ptr += outer_q_dim_size * head_size_q * sizeof(CACHE_T);
    llm_kernels::nvidia::ConvertToCacheType<SCALAR_T, CACHE_T, KV_DTYPE>(
        /*q_src*/ reinterpret_cast<SCALAR_T*>(q_ptr), /*q_dst*/ reinterpret_cast<CACHE_T*>(quant_q_ptr), token_num,
        this->num_heads_, head_size_q, this->num_heads_ * head_size_q, /*q_scale*/ 1.f, stream);
    expanded_q_ptr = quant_q_ptr;
  } else if (this->num_heads_ < kMinNumHeads) {
    num_heads = kMinNumHeads;
    // Expand q_ptr to expanded_q_ptr
    using Q_DTYPE = std::conditional_t<KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8DsMla, SCALAR_T, CACHE_T>;
    const size_t dst_pitch = kMinNumHeads * head_size_q * sizeof(Q_DTYPE);
    const size_t src_pitch = this->num_heads_ * head_size_q * sizeof(Q_DTYPE);
    expanded_q_ptr = workspace_ptr;
    Memcpy2DAsync(expanded_q_ptr, dst_pitch, q_ptr, src_pitch, src_pitch, token_num, MEMCPY_DEVICE_TO_DEVICE,
                  this->context_->device->GetPrimaryComputeStream());
    workspace_ptr += token_num * dst_pitch;
    expanded_out_ptr = reinterpret_cast<SCALAR_T*>(workspace_ptr);
    workspace_ptr += token_num * dst_pitch;
  }
  // Invoke sparse mla kernel to compute the output [token_num, num_heads_q, kv_lora_rank]
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeFlashSparseMlaWithKVCache<SCALAR_T, CACHE_T, KV_DTYPE>(
      expanded_q_ptr, kcache_ptr, indices_ptr, tile_scheduler_metadata_ptr, num_splits_ptr,
      /*sched_meta_initialized, shape[2] is the init flag*/ tile_scheduler_metadata_tensor.shape[2] != 0, batch_size,
      q_seq_len, num_heads, head_size_q, num_blocks, this->block_token_num_,
      /*num_heads_k*/ 1,
      /*kcache_batch_stride*/ this->block_size_, topk, this->kv_lora_rank_, this->attn_scale_, stream,
      reinterpret_cast<float*>(workspace_ptr), expanded_out_ptr));
  if (KV_DTYPE != llm_kernels::utils::KVCacheType::kFp8E4M3 && this->num_heads_ < kMinNumHeads) {
    const size_t dst_pitch = this->num_heads_ * this->kv_lora_rank_ * sizeof(SCALAR_T);
    const size_t src_pitch = kMinNumHeads * this->kv_lora_rank_ * sizeof(SCALAR_T);
    // Squeeze expanded_out_ptr to out_ptr
    Memcpy2DAsync(out_ptr, dst_pitch, expanded_out_ptr, src_pitch, dst_pitch, token_num, MEMCPY_DEVICE_TO_DEVICE,
                  this->context_->device->GetPrimaryComputeStream());
  }

  return Status();
}

}  // namespace ksana_llm
