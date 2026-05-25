/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "models/base/forward_meta/mla_meta.h"

#include "device/device_utils.h"
#include "models/base/forward_meta/common.h"
#include "utils/logger.h"

namespace ksana_llm {

MlaMeta::MlaMeta(const RuntimeConfig& runtime_config, int rank, std::shared_ptr<Context> context) : context_(context) {
  attn_dp_group_id_ = rank / runtime_config.parallel_basic_config.attn_tensor_parallel_size;
  max_chunked_kv_tokens_ = runtime_config.max_chunked_kv_tokens;

  const size_t max_batch_size = runtime_config.max_batch_size;
  chunk_kv_batch_starts_host = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT32, {max_batch_size + 1});
  chunk_kv_cu_extend_lens_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {2 * max_batch_size});
  chunk_kv_cu_seq_lens_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {2 * max_batch_size});
}

void MlaMeta::Prepare(const std::vector<ForwardRequest>& forward_reqs) {
  // Reset meta
  flash_input.Reset();
  dp_context_tokens = 0;

  // Categorize requests into flash (prefill) vs page (decode)
  for (const auto& req : forward_reqs) {
    if (req.attn_dp_group_id == attn_dp_group_id_) {
      const size_t extend_len = req.GetInputIdsLength();
      if (GetForwardRequestType(req) == ForwardRequestType::kFlash) {
        flash_input.dp_reqs.emplace_back(&req);
        dp_context_tokens += extend_len;
      } else {  // ForwardRequestType::kPage
      }
    }
  }

  if (!flash_input.dp_reqs.empty()) {
    PrepareChunkKV();
  }
}

void MlaMeta::PrepareChunkKV() {
  const auto& reqs = flash_input.dp_reqs;
  const int batch_size = reqs.size();

  size_t num_chunks = 0;
  int* batch_starts_ptr = chunk_kv_batch_starts_host.GetPtr<int>();
  chunk_kv_cu_extend_lens_list_.clear();
  chunk_kv_cu_seq_lens_list_.clear();

  // Greedily pack consecutive requests into the same chunk while the summed input length fits in
  // `max_chunked_kv_tokens`; when it would overflow, start a new chunk.
  for (int i = 0; i < batch_size;) {
    batch_starts_ptr[num_chunks++] = i;

    int cur_extend_len_sum = 0;
    int cur_seq_len_sum = 0;
    chunk_kv_cu_extend_lens_list_.push_back(0);
    chunk_kv_cu_seq_lens_list_.push_back(0);

    do {
      cur_extend_len_sum += reqs[i]->GetInputIdsLength();
      cur_seq_len_sum += reqs[i]->forwarding_tokens->size();
      chunk_kv_cu_extend_lens_list_.push_back(cur_extend_len_sum);
      chunk_kv_cu_seq_lens_list_.push_back(cur_seq_len_sum);
      ++i;
    } while (i < batch_size &&
             cur_seq_len_sum + static_cast<int>(reqs[i]->forwarding_tokens->size()) <= max_chunked_kv_tokens_);
  }
  // Append a sentinel so consumers can compute the last chunk's size via consecutive differences.
  batch_starts_ptr[num_chunks++] = batch_size;
  chunk_kv_batch_starts_host.shape[0] = num_chunks;

  HOST_TO_DEVICE(chunk_kv_cu_extend_lens_list_, chunk_kv_cu_extend_lens_tensor);
  HOST_TO_DEVICE(chunk_kv_cu_seq_lens_list_, chunk_kv_cu_seq_lens_tensor);
}

}  // namespace ksana_llm
