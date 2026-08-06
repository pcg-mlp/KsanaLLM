/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "runtime/speculative/mtp/mtp_gather_index_builder.h"

#include "configure/context.h"
#include "profiler/profile_event.h"
#include "runtime/forward_request.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"
#include "utils/tensor.h"

namespace ksana_llm {

MtpGatherIndexBuilder::MtpGatherIndexBuilder(const size_t max_token_num, std::shared_ptr<Context> context, int rank)
    : max_token_num_(max_token_num), context_(context), rank_(rank) {
  nextn_hidden_idx_uint64_tensor_ = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_token_num_});
}

void MtpGatherIndexBuilder::UpdateMainForward(const std::vector<ForwardRequest>& forward_reqs) {
  PROFILE_EVENT_SCOPE(MtpGatherIndexBuilder_UpdateMainForward, "MtpGatherIndexBuilder_UpdateMainForward", rank_);
  std::unordered_map<int64_t, size_t> updated_mtp_req_id_to_pos;
  updated_mtp_req_id_to_pos.reserve(forward_reqs.size());

  size_t total_len = 0;
  for (const auto& req : forward_reqs) {
    const size_t input_ids_len = req.GetInputIdsLength();
    updated_mtp_req_id_to_pos[req.req_id] = total_len;
    total_len += input_ids_len;
  }

  mtp_req_id_to_pos_.swap(updated_mtp_req_id_to_pos);
}

void MtpGatherIndexBuilder::BuildGatherIndices(const std::vector<ForwardRequest>& forward_reqs) {
  PROFILE_EVENT_SCOPE(MtpGatherIndexBuilder_BuildGatherIndices, "MtpGatherIndexBuilder_BuildGatherIndices", rank_);
  std::unordered_map<int64_t, size_t> updated_mtp_req_id_to_pos;
  updated_mtp_req_id_to_pos.reserve(forward_reqs.size());

  std::vector<size_t> mtp_hidden_gather_idx;
  mtp_hidden_gather_idx.reserve(forward_reqs.size());
  size_t total_len = 0;
  for (const auto& req : forward_reqs) {
    const size_t input_ids_len = req.GetInputIdsLength();
    updated_mtp_req_id_to_pos[req.req_id] = total_len + input_ids_len - 1;
    total_len += input_ids_len;

    const size_t begin_idx = mtp_req_id_to_pos_[req.req_id];
    for (size_t idx = begin_idx; idx < begin_idx + input_ids_len; ++idx) {
      mtp_hidden_gather_idx.emplace_back(idx);
    }
  }

  mtp_req_id_to_pos_.swap(updated_mtp_req_id_to_pos);

  nextn_hidden_idx_uint64_tensor_.shape = {mtp_hidden_gather_idx.size()};
  MemcpyAsync(nextn_hidden_idx_uint64_tensor_.GetPtr<void>(), mtp_hidden_gather_idx.data(),
              mtp_hidden_gather_idx.size() * sizeof(decltype(mtp_hidden_gather_idx)::value_type), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetH2DStream());
  KLLM_LOG_DEBUG << "mtp_hidden_gather_idx: " << mtp_hidden_gather_idx;
}

Tensor* MtpGatherIndexBuilder::GetGatherTensor() { return &nextn_hidden_idx_uint64_tensor_; }

}  // namespace ksana_llm
