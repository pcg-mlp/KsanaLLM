/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/flash_dsv4_attention_layer.h"

#include "csrc/kernels/nvidia/flash_sparse_mla/flash_sparse_mla.h"
#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/rope.h"
#include "csrc/kernels/nvidia/others/vllm/main/deepseek_v4/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.h"
#include "device/nvidia/cuda_utils.h"
#include "utils/logger.h"

namespace ksana_llm {

Status FlashDsv4AttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                     std::shared_ptr<Context> context, int rank) {
  max_token_num_ = runtime_config.max_step_token_num;
  STATUS_CHECK_RETURN(AttentionLayer::Init(parameters, runtime_config, context, rank));

  KLLM_CHECK_WITH_INFO(num_kv_heads_ == 1, "FlashDsv4AttentionLayer expects num_kv_heads == 1");

  // Compute page_bits = ctz(block_token_num / compress_ratio)
  KLLM_CHECK(block_token_num_ > 0 && (block_token_num_ & (block_token_num_ - 1)) == 0);
  page_bits_ = __builtin_ctz(block_token_num_);
  KLLM_CHECK_WITH_INFO(page_bits_ == 1 || page_bits_ == 6 || page_bits_ == 8,
                       "FlashDsv4AttentionLayer expects page_bits in {1, 6, 8}");
  const uint32_t compress_ratio = std::any_cast<const uint32_t>(parameters.back());
  KLLM_CHECK(compress_ratio == 0 || compress_ratio == 4u || compress_ratio == 128u);
  compressed_block_token_num_ = static_cast<uint32_t>(block_token_num_) / (compress_ratio == 0 ? 1u : compress_ratio);
  KLLM_CHECK(compressed_block_token_num_ > 0 && (compressed_block_token_num_ & (compressed_block_token_num_ - 1)) == 0);

  return Status();
}

size_t FlashDsv4AttentionLayer::GetWorkspaceSize() {
  // V4 uses flash_mla_with_kvcache which needs:
  //   - lse buffer: [max_token_num, num_heads] * sizeof(float)
  //   - lse_accum: [num_sms, num_heads] * sizeof(float)
  //   - o_accum: [num_sms, num_heads, head_dim] * sizeof(float)
  constexpr int kMinNumHeads = 64;
  const size_t num_sms = llm_kernels::utils::GetSMCount();
  const size_t workspace_size = (/*lse*/ max_token_num_ * kMinNumHeads + /*lse_accum*/ num_sms * kMinNumHeads +
                                 /*o_accum*/ num_sms * kMinNumHeads * head_size_) *
                                sizeof(float);
  if (kv_cache_dtype_ == TYPE_FP8_DS_MLA && num_heads_ < kMinNumHeads) {
    // [expanded_q][expanded_out][workspace]
    const size_t expanded_q_size = max_token_num_ * kMinNumHeads * head_size_ * GetTypeSize(inter_data_type_);
    const size_t expanded_out_size = max_token_num_ * kMinNumHeads * head_size_ * GetTypeSize(inter_data_type_);
    return expanded_q_size + expanded_out_size + workspace_size;
  }
  return workspace_size;
}

Status FlashDsv4AttentionLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
// Dispatch by inter_data_type_ and kv_cache_dtype_
#define DISPATCH_FLASH_DSV4_ATTENTION_BY_KVTYPE(dtype, kv_cache_dtype, func, ...)                             \
  switch (kv_cache_dtype) {                                                                                   \
    case TYPE_FP8_DS_MLA:                                                                                     \
      return func<dtype, uint8_t, llm_kernels::utils::KVCacheType::kFp8DsMla>(__VA_ARGS__);                   \
    default:                                                                                                  \
      KLLM_THROW(fmt::format("{}: Unsupported KVCacheDtype type: {}.", __PRETTY_FUNCTION__, kv_cache_dtype)); \
  }
  switch (inter_data_type_) {
    case DataType::TYPE_BF16:
      DISPATCH_FLASH_DSV4_ATTENTION_BY_KVTYPE(bfloat16, kv_cache_dtype_, ForwardT, input_tensors, output_tensors);
    default:
      KLLM_THROW(fmt::format("{}: Unsupported Dtype type: {}.", __PRETTY_FUNCTION__, inter_data_type_));
  }
#undef DISPATCH_FLASH_DSV4_ATTENTION_BY_KVTYPE
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status FlashDsv4AttentionLayer::ForwardT(const std::vector<Tensor>& input_tensors,
                                         std::vector<Tensor>& output_tensors) {
  auto input_iter = input_tensors.cbegin();
  // Parse input tensors
  int64_t* rotary_embedding_pos_ptr = (input_iter++)->GetPtr<int64_t>();
  const Tensor& q_tensor = *input_iter++;
  void* q_ptr = q_tensor.GetPtr<void>();
  void* kv_ptr = (input_iter++)->GetPtr<void>();
  const Tensor& layer_swa_kv_cache_ptr_tensor = *input_iter++;
  const Tensor& tile_scheduler_metadata_tensor = *input_iter++;
  int* tile_scheduler_metadata_ptr = tile_scheduler_metadata_tensor.GetPtr<int>();
  int* num_splits_ptr = (input_iter++)->GetPtr<int>();
  float* attn_sink_ptr = (input_iter++)->GetPtr<float>();
  int* raw_out_loc_ptr = (input_iter++)->GetPtr<int>();
  const Tensor& indices_tensor = *input_iter++;
  int* indices_ptr = indices_tensor.GetPtr<int>();

  SCALAR_T* out_ptr = output_tensors[0].GetPtr<SCALAR_T>();

  const size_t token_num = q_tensor.shape[0];
  // layer_swa_kv_cache_ptr_tensor layout:
  //   slot 0       : device_block_num (int64) — SWA pool 总 block 数 (kernel indices 上限校验)
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
  if (input_iter != input_tensors.cend()) {
    const Tensor& layer_extra_kv_cache_ptr_tensor = *input_iter++;
    const Tensor& extra_indices_tensor = *input_iter++;
    extra_indices_ptr = extra_indices_tensor.GetPtr<int>();
    // extra_topk_length is optional (provided for c128, not for c4)
    if (input_iter != input_tensors.cend()) {
      extra_topk_length_ptr = (input_iter++)->GetPtr<int>();
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
  const int num_heads = kMinNumHeads;

  uint8_t* workspace_ptr = workspace_buffer_->GetPtr<uint8_t>();
  void* expanded_q_ptr = q_ptr;
  SCALAR_T* expanded_out_ptr = out_ptr;
  if (this->num_heads_ < kMinNumHeads) {
    expanded_q_ptr = workspace_ptr;
    const size_t expanded_q_bytes = token_num * kMinNumHeads * this->head_size_ * sizeof(SCALAR_T);
    workspace_ptr += expanded_q_bytes;
    expanded_out_ptr = reinterpret_cast<SCALAR_T*>(workspace_ptr);
    workspace_ptr += expanded_q_bytes;  // expanded_out_bytes == expanded_q_bytes
  }

  KLLM_CHECK_WITH_INFO(this->rotary_embedding_cuda_.has_value(), "rotary_embedding_cuda must be set!");
  llm_kernels::nvidia::InvokeFusedQNormRopeKVInsert(
      reinterpret_cast<const SCALAR_T*>(q_ptr), reinterpret_cast<const SCALAR_T*>(kv_ptr),
      reinterpret_cast<SCALAR_T*>(expanded_q_ptr), kcache_ptr, rotary_embedding_pos_ptr,
      reinterpret_cast<const SCALAR_T*>(this->rotary_embedding_cuda_->cos_sin_cache_), raw_out_loc_ptr,
      this->layernorm_eps_, token_num, this->num_heads_, swa_block_size, this->page_bits_, stream);

  // Invoke sparse mla kernel to compute the output [token_num, num_heads_q, head_dim]
  // primary stride = swa_block_size (SWA pool); extra stride = main KV block_size_ (compressed KV in main pool)
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeFlashSparseMlaWithKVCache<SCALAR_T, CACHE_T, KV_DTYPE>(
      expanded_q_ptr, kcache_ptr, indices_ptr, tile_scheduler_metadata_ptr, num_splits_ptr,
      /*sched_meta_initialized, shape[2] is the init flag*/ tile_scheduler_metadata_tensor.shape[2] != 0,
      /*batch_size*/ 1, token_num, num_heads, this->head_size_, num_blocks, this->block_token_num_,
      /*num_heads_k*/ 1,
      /*kcache_batch_stride*/ static_cast<int>(swa_block_size), topk, this->head_size_, this->attn_scale_, stream,
      reinterpret_cast<float*>(workspace_ptr), expanded_out_ptr,
      /*topk_length_ptr*/ nullptr, attn_sink_ptr, extra_kcache_ptr, extra_indices_ptr, extra_topk_length_ptr,
      extra_num_blocks, this->compressed_block_token_num_, extra_topk,
      /*extra_kcache_batch_stride*/ this->block_size_));
  if (this->num_heads_ < kMinNumHeads) {
    const size_t dst_pitch = this->num_heads_ * this->head_size_ * sizeof(SCALAR_T);
    const size_t src_pitch = kMinNumHeads * this->head_size_ * sizeof(SCALAR_T);
    // Squeeze expanded_out_ptr to out_ptr
    Memcpy2DAsync(out_ptr, dst_pitch, expanded_out_ptr, src_pitch, dst_pitch, token_num, MEMCPY_DEVICE_TO_DEVICE,
                  this->context_->device->GetPrimaryComputeStream());
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
