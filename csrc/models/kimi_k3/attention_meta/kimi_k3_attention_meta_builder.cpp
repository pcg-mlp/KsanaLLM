/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/kimi_k3/attention_meta/kimi_k3_attention_meta_builder.h"

#include <fmt/core.h>

#ifdef ENABLE_CUDA
#  include "kernels/nvidia/kda_kernel_wrapper.h"
#else
constexpr int kKdaChunkSize = 64;
#endif

#include "models/deepseek_v4/attention_meta/deepseek_v4_compressor_slot_resource.h"
#include "utils/logger.h"

namespace ksana_llm {

namespace {

// Resolve layer_mamba_kv_cache_ptr_tensor at first Prepare: slot 0 = total_block_num,
// per-layer conv/recurrent offsets → absolute pointers.
void InitMambaBasePointersIfNeeded(KimiK3SharedAttentionMeta& kimi_shared) {
  if (kimi_shared.layer_mamba_kv_cache_ptr_initialized) {
    return;
  }

  KLLM_CHECK_WITH_INFO(kimi_shared.mamba_state_buffer.GetPtr<void>() != nullptr,
                       "KimiK3SharedAttentionMeta::mamba_state_buffer is not allocated (Init was not called).");

  kimi_shared.layer_mamba_kv_cache_ptr_initialized = true;

  // Total slots = capacity * decode_token_num, derived from the buffer shape.
  const int64_t mamba_pool_block_num =
      static_cast<int64_t>(kimi_shared.mamba_state_buffer.shape[0] * kimi_shared.mamba_state_buffer.shape[1]);

  int64_t* mamba_meta = kimi_shared.layer_mamba_kv_cache_ptr_tensor.GetPtr<int64_t>();
  mamba_meta[0] = mamba_pool_block_num;

  int64_t* layer_mamba_ptr = mamba_meta + 1;
  const uintptr_t cache_base = reinterpret_cast<uintptr_t>(kimi_shared.mamba_state_buffer.GetPtr<uint8_t>());

  KLLM_LOG_INFO << fmt::format("init KimiK3 mamba pool base_ptr={} total_block_num={}",
                               reinterpret_cast<const void*>(cache_base), mamba_pool_block_num);

  for (int i = 0; i < kimi_shared.layer_num_on_node; ++i) {
    layer_mamba_ptr[0] = static_cast<int64_t>(cache_base + static_cast<uintptr_t>(layer_mamba_ptr[0]));  // conv
    layer_mamba_ptr[1] = static_cast<int64_t>(cache_base + static_cast<uintptr_t>(layer_mamba_ptr[1]));  // recurrent
    layer_mamba_ptr += 2;
  }
}

}  // namespace

// ── Flash builder ──────────────────────────────────────────────────────────

void KimiK3FlashAttentionMetaBuilder::Build(FlashAttentionMeta& meta, SharedAttentionMeta& shared_meta,
                                            size_t dp_total_prefix_len) {
  // Inherit full DeepSeekV3 flash build: MLA + DSA + Indexer prepare steps.
  DeepSeekV3FlashAttentionMetaBuilder::Build(meta, shared_meta, dp_total_prefix_len);

  auto& kimi_shared = static_cast<KimiK3SharedAttentionMeta&>(shared_meta);
  InitMambaBasePointersIfNeeded(kimi_shared);

  auto& kda_meta = static_cast<KimiK3FlashAttentionMeta&>(meta);
  PrepareStateIndices(kda_meta, kimi_shared);
  PrepareConv1D(kda_meta);
  PrepareChunkKDA(kda_meta);
}

void KimiK3FlashAttentionMetaBuilder::PrepareStateIndices(KimiK3FlashAttentionMeta& meta,
                                                          KimiK3SharedAttentionMeta& kimi_shared) {
  auto* slot_resource = kimi_shared.slot_resource_;
  KLLM_CHECK_WITH_INFO(slot_resource != nullptr,
                       "PrepareStateIndices: slot_resource_ is null "
                       "(WireInflightResources not called).");
  const size_t batch_size = meta.dp_reqs.size();

  auto& state_indices_host = meta.state_indices_host;
  state_indices_host.resize(batch_size);

  for (size_t i = 0; i < batch_size; ++i) {
    const int slot_id = slot_resource->Get(meta.dp_reqs[i]->req_id);
    state_indices_host[i] =
        slot_id * kimi_shared.mamba_state_buffer.shape[1];  // [s * decode_token_num, (s + 1) * decode_token_num)
  }

  // Slice view from the shared tensor, then upload host data.
  meta.state_indices_tensor =
      kimi_shared.state_indices_tensor.GetView({state_indices_host.size()}, kimi_shared.state_indices_tensor.shape[0]);
  kimi_shared.state_indices_tensor.shape[0] += meta.state_indices_tensor.GetElementNumber();
  MemcpyAsync(meta.state_indices_tensor.GetPtr<void>(), state_indices_host.data(),
              state_indices_host.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
              config_.context->device->GetH2DStream());
}

void KimiK3FlashAttentionMetaBuilder::PrepareConv1D(KimiK3FlashAttentionMeta& meta) {
  const size_t batch_size = meta.dp_reqs.size();

  auto& has_initial_state_host = meta.has_initial_state_host;
  has_initial_state_host.resize(batch_size);

  int max_extend_len = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    const auto* req = meta.dp_reqs[i];
    const int extend_len = static_cast<int>(req->GetInputIdsLength());
    if (extend_len > max_extend_len) {
      max_extend_len = extend_len;
    }
    has_initial_state_host[i] = (req->prefix_cache_len > 0) ? 1 : 0;
  }
  meta.max_extend_len_host = max_extend_len;

  meta.has_initial_state_tensor.shape = {has_initial_state_host.size()};
  MemcpyAsync(meta.has_initial_state_tensor.GetPtr<void>(), has_initial_state_host.data(),
              has_initial_state_host.size() * sizeof(uint8_t), MEMCPY_HOST_TO_DEVICE,
              config_.context->device->GetH2DStream());
}

void KimiK3FlashAttentionMetaBuilder::PrepareChunkKDA(KimiK3FlashAttentionMeta& meta) {
  constexpr int kChunkSize = kKdaChunkSize;
  const size_t batch_size = meta.dp_reqs.size();

  auto& chunk_offsets_host = meta.chunk_offsets_host;
  auto& chunk_indices_host = meta.chunk_indices_host;
  chunk_offsets_host.resize(batch_size + 1);
  chunk_indices_host.clear();

  chunk_offsets_host[0] = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    const int extend_len = static_cast<int>(meta.dp_reqs[i]->GetInputIdsLength());
    const int num_chunks = (extend_len + kChunkSize - 1) / kChunkSize;
    chunk_offsets_host[i + 1] = chunk_offsets_host[i] + num_chunks;
    for (int c = 0; c < num_chunks; ++c) {
      chunk_indices_host.push_back(static_cast<int32_t>(i));  // seq_idx
      chunk_indices_host.push_back(c);                        // local chunk_idx
    }
  }
  meta.num_chunks_host = chunk_offsets_host.back();

  auto stream = config_.context->device->GetH2DStream();
  meta.chunk_offsets_tensor.shape = {chunk_offsets_host.size()};
  MemcpyAsync(meta.chunk_offsets_tensor.GetPtr<void>(), chunk_offsets_host.data(),
              chunk_offsets_host.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE, stream);
  meta.chunk_indices_tensor.shape = {static_cast<size_t>(meta.num_chunks_host), 2};
  MemcpyAsync(meta.chunk_indices_tensor.GetPtr<void>(), chunk_indices_host.data(),
              chunk_indices_host.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE, stream);
}

// ── Paged builder ──────────────────────────────────────────────────────────

void KimiK3PagedAttentionMetaBuilder::Build(PagedAttentionMeta& meta, SharedAttentionMeta& shared_meta) {
  // Inherit full DeepSeekV3 paged build: MLA + DSA + Indexer prepare steps.
  DeepSeekV3PagedAttentionMetaBuilder::Build(meta, shared_meta);

  auto& kimi_shared = static_cast<KimiK3SharedAttentionMeta&>(shared_meta);
  InitMambaBasePointersIfNeeded(kimi_shared);

  auto& kimi_meta = static_cast<KimiK3PagedAttentionMeta&>(meta);
  PrepareStateIndices(kimi_meta, kimi_shared);
}

void KimiK3PagedAttentionMetaBuilder::PrepareStateIndices(KimiK3PagedAttentionMeta& meta,
                                                          KimiK3SharedAttentionMeta& kimi_shared) {
  auto* slot_resource = kimi_shared.slot_resource_;
  KLLM_CHECK_WITH_INFO(slot_resource != nullptr,
                       "PrepareStateIndices: slot_resource_ is null "
                       "(WireInflightResources not called).");
  const size_t batch_size = meta.dp_reqs.size();

  auto& state_indices_host = meta.state_indices_host;
  state_indices_host.resize(batch_size);

  for (size_t i = 0; i < batch_size; ++i) {
    const int slot_id = slot_resource->Get(meta.dp_reqs[i]->req_id);
    state_indices_host[i] =
        slot_id * kimi_shared.mamba_state_buffer.shape[1];  // [s * decode_token_num, (s + 1) * decode_token_num)
  }

  // Slice view from the shared tensor, then upload host data.
  meta.state_indices_tensor =
      kimi_shared.state_indices_tensor.GetView({state_indices_host.size()}, kimi_shared.state_indices_tensor.shape[0]);
  kimi_shared.state_indices_tensor.shape[0] += meta.state_indices_tensor.GetElementNumber();
  MemcpyAsync(meta.state_indices_tensor.GetPtr<void>(), state_indices_host.data(),
              state_indices_host.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
              config_.context->device->GetH2DStream());
}

}  // namespace ksana_llm
