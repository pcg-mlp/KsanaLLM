/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include "kernels/nvidia/deepseek_deepgemm_wrapper.h"

#include <c10/cuda/CUDAStream.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "device/nvidia/cuda_utils.h"
#include "utils/memory_utils.h"
#include "utils/tensor.h"

namespace ksana_llm {
namespace nvidia {

// Major-ness stuffs
// Adapted from:
// https://github.com/deepseek-ai/DeepGEMM/blob/main/csrc/utils/layout.hpp
static void MajorCheck(const torch::Tensor& t) {
  const auto dim = t.dim();
  KLLM_CHECK(dim == 2 or dim == 3);
  if (dim == 3) {
    KLLM_CHECK(t.stride(0) == t.size(-2) * t.size(-1));
  }
  KLLM_CHECK(t.stride(-2) == 1 or t.stride(-1) == 1);
}

// Tensor utils
// Adapted from:
// https://github.com/deepseek-ai/DeepGEMM/blob/main/csrc/utils/layout.hpp
template <int N>
static auto GetShape(const torch::Tensor& t) {
  KLLM_CHECK(t.dim() == N);
  return [&t]<size_t... Is>(std::index_sequence<Is...>) {
    return std::make_tuple(static_cast<int>(t.sizes()[Is])...);
  }(std::make_index_sequence<N>());
}

DeepSeekDeepGEMMWrapper* DeepSeekDeepGEMMWrapper::GetInstance() {
#if ENABLE_DEEPSEEK_DEEPGEMM
  static DeepSeekDeepGEMMWrapper instance;
  return &instance;
#else
  KLLM_KERNEL_THROW("ENABLE_DEEPSEEK_DEEPGEMM=0, cannot create DeepSeekDeepGEMMWrapper instance.");
#endif
}

DeepSeekDeepGEMMWrapper::DeepSeekDeepGEMMWrapper() {
#if ENABLE_DEEPSEEK_DEEPGEMM
  num_device_sms_ = llm_kernels::utils::GetSMCount();

  int current_device = -1;
  cudaGetDevice(&current_device);
  KLLM_LOG_DEBUG << "Initializing DeepSeekDeepGEMMWrapper on device " << current_device << " with " << num_device_sms_
                 << " SMs";

  deepseek::InitializeDeepGemm(num_device_sms_);
  KLLM_LOG_DEBUG << "DeepSeekDeepGEMMWrapper initialized successfully on device " << current_device;
#else
  KLLM_KERNEL_THROW("ENABLE_DEEPSEEK_DEEPGEMM=0, skipping DeepSeek DeepGEMM kernel.");
#endif
}

void DeepSeekDeepGEMMWrapper::Fp8MqaLogits(const torch::Tensor& q_fp8,
                                           const std::pair<torch::Tensor, torch::Tensor>& kv_fp8,
                                           const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
                                           const torch::Tensor& cu_seq_len_k_end, torch::Tensor& logits,
                                           int max_seqlen_k, cudaStream_t stream) {
#if ENABLE_DEEPSEEK_DEEPGEMM
  // Save current torch CUDA stream and set to the passed-in stream
  auto original_stream = at::cuda::getCurrentCUDAStream();
  at::cuda::setCurrentCUDAStream(at::cuda::getStreamFromExternal(stream, original_stream.device_index()));

  KLLM_CHECK_WITH_INFO(q_fp8.dim() == 3, "Expect q tensor of shape [seq_len, num_heads, head_dim]");
  KLLM_CHECK_WITH_INFO(kv_fp8.first.dim() == 2, "Expect kv tensor of shape [seq_len_kv, head_dim]");
  KLLM_CHECK_WITH_INFO(kv_fp8.second.dim() == 1 || kv_fp8.second.dim() == 2, "Expect kv scales tensor of rank 1 or 2");
  KLLM_CHECK_WITH_INFO(weights.dim() == 2, "Expect weights tensor of shape [seq_len, num_heads]");
  KLLM_CHECK_WITH_INFO(cu_seq_len_k_start.dim() == 1, "Expect cu_seq_len_k_start rank 1 tensor");
  KLLM_CHECK_WITH_INFO(cu_seq_len_k_end.dim() == 1, "Expect cu_seq_len_k_end rank 1 tensor");

  const int seq_len = q_fp8.size(0);
  const int num_heads = q_fp8.size(1);
  const int head_dim = q_fp8.size(2);
  const int seq_len_kv = kv_fp8.first.size(0);
  const int head_dim_kv = kv_fp8.first.size(1);

  KLLM_CHECK_WITH_INFO(head_dim_kv == head_dim, "KV head dim must match query head dim");
  KLLM_CHECK_WITH_INFO(static_cast<int>(weights.size(0)) == seq_len, "Weights first dimension must equal seq_len");
  KLLM_CHECK_WITH_INFO(static_cast<int>(weights.size(1)) == num_heads, "Weights second dimension must equal num_heads");
  KLLM_CHECK_WITH_INFO(static_cast<int>(cu_seq_len_k_start.size(0)) == seq_len,
                       "cu_seq_len_k_start length must equal seq_len");
  KLLM_CHECK_WITH_INFO(static_cast<int>(cu_seq_len_k_end.size(0)) == seq_len,
                       "cu_seq_len_k_end length must equal seq_len");

  KLLM_CHECK_WITH_INFO(q_fp8.is_contiguous(), "q tensor must be contiguous");
  KLLM_CHECK_WITH_INFO(kv_fp8.first.is_contiguous(), "kv tensor must be contiguous");
  KLLM_CHECK_WITH_INFO(kv_fp8.second.is_contiguous(), "kv scales tensor must be contiguous");
  KLLM_CHECK_WITH_INFO(weights.is_contiguous(), "weights tensor must be contiguous");
  KLLM_CHECK_WITH_INFO(cu_seq_len_k_start.is_contiguous(), "cu_seq_len_k_start must be contiguous");
  KLLM_CHECK_WITH_INFO(cu_seq_len_k_end.is_contiguous(), "cu_seq_len_k_end must be contiguous");

  KLLM_CHECK_WITH_INFO(q_fp8.scalar_type() == torch::kFloat8_e4m3fn, "q tensor must be float8 e4m3");
  KLLM_CHECK_WITH_INFO(kv_fp8.first.scalar_type() == torch::kFloat8_e4m3fn, "kv tensor must be float8 e4m3");
  KLLM_CHECK_WITH_INFO(kv_fp8.second.scalar_type() == torch::kFloat, "kv scales tensor must be float32");
  KLLM_CHECK_WITH_INFO(weights.scalar_type() == torch::kFloat, "weights tensor must be float32");
  KLLM_CHECK_WITH_INFO(cu_seq_len_k_start.scalar_type() == torch::kInt, "cu_seq_len_k_start tensor must be int32");
  KLLM_CHECK_WITH_INFO(cu_seq_len_k_end.scalar_type() == torch::kInt, "cu_seq_len_k_end tensor must be int32");

  KLLM_CHECK_WITH_INFO(max_seqlen_k > 0, "max_seqlen_k must be positive for compressed logits mode");

  constexpr int seq_len_alignment = 4;
  constexpr int block_kv = 256;
  const int aligned_seq_len = RoundUp(seq_len, seq_len_alignment);
  const int stride_logits = RoundUp(max_seqlen_k, block_kv);

  KLLM_CHECK_WITH_INFO(logits.device() == q_fp8.device(), "logits tensor must be on the same device as q");
  KLLM_CHECK_WITH_INFO(logits.scalar_type() == torch::kFloat, "logits tensor must be float32");
  KLLM_CHECK_WITH_INFO(logits.dim() == 2, "logits tensor must have rank 2");
  KLLM_CHECK_WITH_INFO(static_cast<int>(logits.size(0)) >= aligned_seq_len,
                       "logits rows must be at least the aligned sequence length");
  KLLM_CHECK_WITH_INFO(static_cast<int>(logits.size(1)) >= stride_logits,
                       "logits cols must be at least the aligned kv length");
  KLLM_CHECK_WITH_INFO(logits.stride(1) == 1, "logits tensor must be contiguous on the last dimension");
  // Narrow to actual data range [seq_len, max_seqlen_k]; underlying stride remains stride_logits (aligned to block_kv).
  logits = logits.narrow(0, 0, seq_len).narrow(1, 0, max_seqlen_k);
  CUDA_CHECK_LAST_ERROR(deepseek::SmxxFp8MqaLogits(q_fp8, kv_fp8.first, kv_fp8.second, weights, cu_seq_len_k_start,
                                                   cu_seq_len_k_end, logits, seq_len, seq_len_kv, max_seqlen_k,
                                                   stride_logits, num_heads, head_dim, seq_len_alignment));

  // Restore original CUDA stream
  at::cuda::setCurrentCUDAStream(original_stream);
#else
  KLLM_KERNEL_THROW("ENABLE_DEEPSEEK_DEEPGEMM=0, skipping DeepSeek DeepGEMM kernel.");
#endif
}

void DeepSeekDeepGEMMWrapper::Fp8PagedMqaLogits(const torch::Tensor& q_fp8, const torch::Tensor& fused_kv_cache,
                                                const torch::Tensor& weights, const torch::Tensor& context_lens,
                                                const torch::Tensor& block_table, const torch::Tensor& schedule_meta,
                                                torch::Tensor& logits, int max_context_len, cudaStream_t stream,
                                                bool clean_logits) {
#if ENABLE_DEEPSEEK_DEEPGEMM
  // Save current torch CUDA stream and set to the passed-in stream
  auto original_stream = at::cuda::getCurrentCUDAStream();
  at::cuda::setCurrentCUDAStream(at::cuda::getStreamFromExternal(stream, original_stream.device_index()));

  KLLM_CHECK_WITH_INFO(q_fp8.dim() == 4, "Expect q tensor of shape [batch_size, next_n, num_heads, head_dim]");
  KLLM_CHECK_WITH_INFO(fused_kv_cache.dim() == 4, "Expect fused_kv_cache tensor of rank 4");
  KLLM_CHECK_WITH_INFO(weights.dim() == 2, "Expect weights tensor of rank 2");
  // NOTE(jinxcwu)
  // https://github.com/deepseek-ai/DeepGEMM/pull/231/files#diff-59e30829961e1b429bc12115673562f6f15d2ed347cac8d27a879bf101e977cbL160
  const bool is_context_lens_2d = context_lens.dim() == 2;
  KLLM_CHECK_WITH_INFO(block_table.dim() == 2, "Expect block_table tensor of rank 2");
  KLLM_CHECK_WITH_INFO(schedule_meta.dim() == 2, "Expect schedule_meta tensor of rank 2");

  const int batch_size = static_cast<int>(q_fp8.size(0));
  const int next_n = static_cast<int>(q_fp8.size(1));
  const int num_heads = static_cast<int>(q_fp8.size(2));
  const int head_dim = static_cast<int>(q_fp8.size(3));

  const int num_kv_blocks = static_cast<int>(fused_kv_cache.size(0));
  const int block_kv = static_cast<int>(fused_kv_cache.size(1));
  const int num_heads_kv = static_cast<int>(fused_kv_cache.size(2));
  const int head_dim_with_sf = static_cast<int>(fused_kv_cache.size(3));

  KLLM_CHECK_WITH_INFO(static_cast<int>(weights.size(0)) == batch_size * next_n,
                       "weights first dimension must equal batch_size * next_n");
  KLLM_CHECK_WITH_INFO(static_cast<int>(weights.size(1)) == num_heads, "weights second dimension must equal num_heads");
  KLLM_CHECK_WITH_INFO(static_cast<int>(context_lens.size(0)) == batch_size,
                       "context_lens length must equal batch_size");
  KLLM_CHECK_WITH_INFO(static_cast<int>(block_table.size(0)) == batch_size,
                       "block_table first dimension must equal batch_size");
  KLLM_CHECK_WITH_INFO(num_heads_kv == 1, "Expect num_heads_kv == 1 in fused_kv_cache");
  KLLM_CHECK_WITH_INFO(head_dim_with_sf == head_dim + static_cast<int>(sizeof(float)),
                       "Expect fused_kv_cache last dim to equal head_dim + sizeof(float)");

  const int num_sms = num_device_sms_;
  KLLM_CHECK_WITH_INFO(static_cast<int>(schedule_meta.size(0)) == num_sms + 1,
                       "schedule_meta rows must equal num_sms + 1");
  KLLM_CHECK_WITH_INFO(static_cast<int>(schedule_meta.size(1)) == 2, "schedule_meta cols must equal 2");

  KLLM_CHECK_WITH_INFO(q_fp8.is_contiguous(), "q tensor must be contiguous");
  KLLM_CHECK_WITH_INFO(weights.is_contiguous(), "weights tensor must be contiguous");
  KLLM_CHECK_WITH_INFO(context_lens.is_contiguous(), "context_lens tensor must be contiguous");
  KLLM_CHECK_WITH_INFO(schedule_meta.is_contiguous(), "schedule_meta tensor must be contiguous");

  KLLM_CHECK_WITH_INFO(q_fp8.scalar_type() == torch::kFloat8_e4m3fn, "q tensor must be float8 e4m3");
  KLLM_CHECK_WITH_INFO(fused_kv_cache.scalar_type() == torch::kByte, "fused_kv_cache tensor must be uint8");
  KLLM_CHECK_WITH_INFO(weights.scalar_type() == torch::kFloat, "weights tensor must be float32");
  KLLM_CHECK_WITH_INFO(context_lens.scalar_type() == torch::kInt, "context_lens tensor must be int32");
  KLLM_CHECK_WITH_INFO(block_table.scalar_type() == torch::kInt, "block_table tensor must be int32");
  KLLM_CHECK_WITH_INFO(schedule_meta.scalar_type() == torch::kInt, "schedule_meta tensor must be int32");

  const int kv_cache_stride_bytes = static_cast<int>(fused_kv_cache.stride(0));
  const int block_table_stride = static_cast<int>(block_table.stride(0));

  KLLM_CHECK_WITH_INFO(fused_kv_cache.stride(1) == head_dim_with_sf,
                       "fused_kv_cache stride(1) must equal head_dim + sizeof(float)");
  KLLM_CHECK_WITH_INFO(fused_kv_cache.stride(2) == head_dim_with_sf,
                       "fused_kv_cache stride(2) must equal head_dim + sizeof(float)");
  KLLM_CHECK_WITH_INFO(fused_kv_cache.stride(3) == 1, "fused_kv_cache stride(3) must equal 1");
  KLLM_CHECK_WITH_INFO(block_table.stride(1) == 1, "block_table stride(1) must equal 1");
  KLLM_CHECK_WITH_INFO(kv_cache_stride_bytes % static_cast<int>(sizeof(float)) == 0,
                       "kv cache stride must be divisible by sizeof(float), kv_cache_stride_bytes");

  const auto device = fused_kv_cache.device();
  const auto fp8_options = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device);
  const auto f32_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

  auto kv_cache =
      torch::from_blob(fused_kv_cache.data_ptr(), {num_kv_blocks, block_kv, head_dim},
                       {static_cast<int64_t>(kv_cache_stride_bytes), static_cast<int64_t>(head_dim), 1}, fp8_options);

  auto kv_cache_scales =
      torch::from_blob(fused_kv_cache.data_ptr<uint8_t>() + block_kv * head_dim, {num_kv_blocks, block_kv},
                       {static_cast<int64_t>(kv_cache_stride_bytes / static_cast<int>(sizeof(float))), 1}, f32_options);

  // NOTE(jinxcwu)
  // https://github.com/deepseek-ai/DeepGEMM/pull/280/files#diff-59e30829961e1b429bc12115673562f6f15d2ed347cac8d27a879bf101e977cbR233
  constexpr int split_kv = 256;
  const int aligned_max_context_len = RoundUp(max_context_len, split_kv);

  KLLM_CHECK_WITH_INFO(logits.device() == q_fp8.device(), "logits tensor must be on the same device as q");
  KLLM_CHECK_WITH_INFO(logits.scalar_type() == torch::kFloat, "logits tensor must be float32");
  KLLM_CHECK_WITH_INFO(logits.dim() == 2, "logits tensor must have rank 2");
  KLLM_CHECK_WITH_INFO(static_cast<int>(logits.size(0)) >= batch_size * next_n,
                       "logits rows must be at least batch_size * next_n");
  KLLM_CHECK_WITH_INFO(static_cast<int>(logits.size(1)) >= aligned_max_context_len,
                       "logits cols must be at least the aligned max context length");
  KLLM_CHECK_WITH_INFO(logits.stride(1) == 1, "logits tensor must be contiguous on the last dimension");
  logits = logits.narrow(-1, 0, max_context_len);
  CUDA_CHECK_LAST_ERROR(deepseek::SmxxFp8PagedMqaLogits(
      q_fp8, kv_cache, kv_cache_scales, weights, context_lens, logits, block_table, schedule_meta, batch_size, next_n,
      num_heads, head_dim, num_kv_blocks, block_kv, is_context_lens_2d, kv_cache_stride_bytes, aligned_max_context_len,
      block_table_stride, num_sms, split_kv, clean_logits, max_context_len));

  // Restore original CUDA stream
  at::cuda::setCurrentCUDAStream(original_stream);
#else
  KLLM_KERNEL_THROW("ENABLE_DEEPSEEK_DEEPGEMM=0, skipping DeepSeek DeepGEMM kernel.");
#endif
}

void DeepSeekDeepGEMMWrapper::PagedMqaLogitsMetadata(const torch::Tensor& context_lens,
                                                     torch::Tensor& schedule_metadata, int batch_size, int block_kv,
                                                     cudaStream_t stream) {
#if ENABLE_DEEPSEEK_DEEPGEMM
  // Save current torch CUDA stream and set to the passed-in stream
  auto original_stream = at::cuda::getCurrentCUDAStream();
  at::cuda::setCurrentCUDAStream(at::cuda::getStreamFromExternal(stream, original_stream.device_index()));

  // NOTE(jinxcwu)
  // https://github.com/deepseek-ai/DeepGEMM/pull/231/files#diff-59e30829961e1b429bc12115673562f6f15d2ed347cac8d27a879bf101e977cbL125
  const bool is_context_lens_2d = context_lens.dim() == 2;
  int batch_size_ = 0, next_n = 0;
  if (is_context_lens_2d) {
    batch_size_ = context_lens.size(0);
    next_n = context_lens.size(1);
  } else {
    KLLM_CHECK_WITH_INFO(context_lens.dim() == 1, "context_lens must be rank 1");
    batch_size_ = context_lens.size(0);
  }
  KLLM_CHECK_WITH_INFO(context_lens.scalar_type() == torch::kInt, "context_lens tensor must be int32");
  KLLM_CHECK_WITH_INFO(context_lens.is_contiguous(), "context_lens must be contiguous");
  KLLM_CHECK_WITH_INFO(batch_size_ == batch_size, "context_lens size(0) must equal batch_size");

  KLLM_CHECK_WITH_INFO(schedule_metadata.scalar_type() == torch::kInt, "schedule_metadata tensor must be int32");
  KLLM_CHECK_WITH_INFO(schedule_metadata.dim() == 2, "schedule_metadata must be rank 2");
  KLLM_CHECK_WITH_INFO(static_cast<int>(schedule_metadata.size(0)) == num_device_sms_ + 1,
                       "schedule_metadata rows must equal num_sms + 1");
  KLLM_CHECK_WITH_INFO(static_cast<int>(schedule_metadata.size(1)) == 2, "schedule_metadata cols must equal 2");

  CUDA_CHECK_LAST_ERROR(deepseek::SmxxPagedMqaLogitsMetadata(context_lens, schedule_metadata, batch_size, next_n,
                                                             block_kv, num_device_sms_, is_context_lens_2d));

  // Restore original CUDA stream
  at::cuda::setCurrentCUDAStream(original_stream);
#else
  KLLM_KERNEL_THROW("ENABLE_DEEPSEEK_DEEPGEMM=0, skipping DeepSeek DeepGEMM kernel.");
#endif
}

void DeepSeekDeepGEMMWrapper::Tf32HcPreNormGemm(const Tensor& a_tensor, const Tensor& b_tensor, Tensor& d_tensor,
                                                Tensor& sqr_sum_tensor, const int num_splits, const int rank,
                                                cudaStream_t stream) {
#if ENABLE_DEEPSEEK_DEEPGEMM
  // Save current torch CUDA stream and set to the passed-in stream
  auto original_stream = at::cuda::getCurrentCUDAStream();
  at::cuda::setCurrentCUDAStream(at::cuda::getStreamFromExternal(stream, original_stream.device_index()));

  // Convert Ksana tensor to torch tensor
  auto device = torch::Device(torch::kCUDA, rank);
  auto a = torch::from_blob(a_tensor.GetPtr<bfloat16*>(),
                            {static_cast<int64_t>(a_tensor.shape[0]), static_cast<int64_t>(a_tensor.shape[1])},
                            torch::TensorOptions().dtype(torch::kBFloat16).device(device));
  auto b = torch::from_blob(b_tensor.GetPtr<float*>(),
                            {static_cast<int64_t>(b_tensor.shape[0]), static_cast<int64_t>(b_tensor.shape[1])},
                            torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto d_shape = num_splits > 1 ? std::vector<int64_t>{num_splits, static_cast<int64_t>(d_tensor.shape[1]),
                                                       static_cast<int64_t>(d_tensor.shape[2])}
                                : std::vector<int64_t>{static_cast<int64_t>(d_tensor.shape[1]),
                                                       static_cast<int64_t>(d_tensor.shape[2])};
  auto d = torch::from_blob(d_tensor.GetPtr<float*>(), d_shape,
                            torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto sqr_sum_shape = num_splits > 1 ? std::vector<int64_t>{num_splits, static_cast<int64_t>(sqr_sum_tensor.shape[1])}
                                      : std::vector<int64_t>{static_cast<int64_t>(sqr_sum_tensor.shape[1])};
  auto sqr_sum = torch::from_blob(sqr_sum_tensor.GetPtr<float*>(), sqr_sum_shape,
                                  torch::TensorOptions().dtype(torch::kFloat32).device(device));

  // A and B must be K-major, D must be N-major
  MajorCheck(a);
  MajorCheck(b);
  MajorCheck(d);
  KLLM_CHECK_WITH_INFO(a.stride(-1) == 1, "Tensor a must be row-major");
  KLLM_CHECK_WITH_INFO(b.stride(-1) == 1, "Tensor b must be row-major");
  KLLM_CHECK_WITH_INFO(d.stride(-1) == 1, "Tensor d must be row-major");

  // S must be contiguous
  KLLM_CHECK_WITH_INFO(sqr_sum.is_contiguous(), "Tensor sqr_sum must be contiguous");

  // Type and shape checks
  const auto& [m, k] = GetShape<2>(a);
  const auto& [n, k_] = GetShape<2>(b);
  if (num_splits > 1) {
    const auto& [num_splits_, m_, n_] = GetShape<3>(d);
    const auto& [num_splits__, m__] = GetShape<2>(sqr_sum);
    KLLM_CHECK(num_splits == num_splits_ and num_splits == num_splits__);
    KLLM_CHECK_WITH_INFO(m == m_ and m == m__ and n == n_ and k == k_, "Shape mismatch for num_splits > 0");
  } else {
    const auto& [m_, n_] = GetShape<2>(d);
    const auto& [m__] = GetShape<1>(sqr_sum);
    KLLM_CHECK_WITH_INFO(m == m_ and m == m__ and n == n_ and k == k_, "Shape mismatch for num_splits = 1");
  }
  KLLM_CHECK(n > 0 and k > 0);
  KLLM_CHECK_WITH_INFO(a.scalar_type() == torch::kBFloat16, "Tensor a must be bfloat16");
  KLLM_CHECK_WITH_INFO(b.scalar_type() == torch::kFloat, "Tensor b must be float");
  KLLM_CHECK_WITH_INFO(d.scalar_type() == torch::kFloat, "Tensor d must be float");
  KLLM_CHECK_WITH_INFO(sqr_sum.scalar_type() == torch::kFloat, "Tensor sqr_sum must be float");

  // Do nothing if the problem is empty
  if (m == 0) {
    // Restore original CUDA stream before early return
    at::cuda::setCurrentCUDAStream(original_stream);
    return;
  }

  // Dispatch into different implements
  CUDA_CHECK_LAST_ERROR(deepseek::Sm90Tf32HcPreNormGemm(a, b, d, sqr_sum, m, n, k, num_splits));

  // Restore original CUDA stream
  at::cuda::setCurrentCUDAStream(original_stream);
#else
  KLLM_KERNEL_THROW("ENABLE_DEEPSEEK_DEEPGEMM=0, skipping DeepSeek DeepGEMM kernel.");
#endif
}

}  // namespace nvidia
}  // namespace ksana_llm
