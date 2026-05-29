/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/base/attention_meta/paged_attention_meta_builder.h"

#include <numeric>

#include "profiler/profile_event.h"

namespace ksana_llm {

using namespace attention_meta_builder_detail;

StandardPagedAttentionMetaBuilder::StandardPagedAttentionMetaBuilder(const AttentionBuilderConfig& config)
    : config_(config) {}

void StandardPagedAttentionMetaBuilder::Build(PagedAttentionMeta& meta, SharedAttentionMeta& shared_meta) {
  auto& standard_shared = *shared_meta.GetStandard();
  auto& standard_meta = static_cast<StandardAttentionMeta&>(meta);
  PrepareInputLength(meta, shared_meta);
  PrepareKVCacheBlocks(standard_meta, standard_shared, config_);
  OnPrepareKVCacheBlockTable(standard_meta, standard_shared);
  PrepareDecodeRotary(meta, shared_meta);
}

void StandardPagedAttentionMetaBuilder::OnPrepareKVCacheBlockTable(StandardAttentionMeta& meta,
                                                                   StandardSharedAttentionMeta& shared_meta) {
  PrepareKVCacheBlockTable(meta, shared_meta, config_, config_.layer_num_on_node, 2);
}

void StandardPagedAttentionMetaBuilder::OnPrepareExpandedInputLength(PagedAttentionMeta& /*meta*/,
                                                                     SharedAttentionMeta& /*shared_meta*/) {}

void StandardPagedAttentionMetaBuilder::PrepareInputLength(PagedAttentionMeta& meta, SharedAttentionMeta& shared_meta) {
  PROFILE_EVENT_SCOPE(PrepareInputLength, "PrepareInputLength", config_.rank);
  const auto& reqs = meta.dp_reqs;
  auto& standard_shared = *shared_meta.GetStandard();

  standard_shared.input_length_host.resize(reqs.size());
  for (size_t i = 0; i < reqs.size(); ++i) {
    standard_shared.input_length_host[i] = reqs[i]->forwarding_tokens->size();
  }

  KLLM_LOG_DEBUG << "input_length_host " << standard_shared.input_length_host;
  meta.input_length = standard_shared.input_length.GetView({standard_shared.input_length_host.size()},
                                                           standard_shared.input_length.shape[0]);
  standard_shared.input_length.shape[0] += meta.input_length.GetElementNumber();
  MemcpyAsync(
      meta.input_length.GetPtr<void>(), standard_shared.input_length_host.data(),
      standard_shared.input_length_host.size() * sizeof(decltype(standard_shared.input_length_host)::value_type),
      MEMCPY_HOST_TO_DEVICE, config_.context->device->GetH2DStream());

  OnPrepareExpandedInputLength(meta, shared_meta);
}

void StandardPagedAttentionMetaBuilder::PrepareDecodeRotary(PagedAttentionMeta& meta,
                                                            SharedAttentionMeta& shared_meta) {
  auto& standard_shared = static_cast<StandardSharedAttentionMeta&>(shared_meta);
  PROFILE_EVENT_SCOPE(PrepareDecodeRotary, "PrepareDecodeRotary", config_.rank);
  size_t total_input_len = 0;
  for (const auto& req : meta.dp_reqs) {
    total_input_len += req->forwarding_tokens->size() - req->kv_cached_token_num;
  }
  meta.total_dp_input_ids_len = total_input_len;

  // prepare mask
  std::vector<int64_t> rotary_mask_host(total_input_len, 1);
  meta.rotary_embedding_mask = standard_shared.rotary_embedding_mask.GetView(
      {rotary_mask_host.size()}, standard_shared.rotary_embedding_mask.shape[0]);
  standard_shared.rotary_embedding_mask.shape[0] += meta.rotary_embedding_mask.GetElementNumber();
  MemcpyAsync(meta.rotary_embedding_mask.GetPtr<void>(), rotary_mask_host.data(),
              sizeof(decltype(rotary_mask_host)::value_type) * rotary_mask_host.size(), MEMCPY_HOST_TO_DEVICE,
              config_.context->device->GetD2HStream());

  // prepare pos
  std::vector<int64_t> rotary_pos_host(total_input_len, 1);
  size_t rotary_data_offset = 0;
  for (const auto& req : meta.dp_reqs) {
    const size_t input_len = req->forwarding_tokens->size() - req->kv_cached_token_num;
    auto pos_offset =
        !config_.model_config.rope_scaling_factor_config.mrope_section.empty() ? *req->mrotary_embedding_pos_offset : 0;
    pos_offset = config_.model_config.type == "arc_hunyuan_video" ? *req->xdrotary_embedding_pos_offset : pos_offset;
    std::iota(rotary_pos_host.begin() + rotary_data_offset, rotary_pos_host.begin() + rotary_data_offset + input_len,
              req->kv_cached_token_num + pos_offset);
    rotary_data_offset += input_len;
  }
  KLLM_LOG_DEBUG << "rotary_pos_host " << rotary_pos_host;
  meta.rotary_embedding_pos = standard_shared.rotary_embedding_pos.GetView(
      {rotary_pos_host.size()}, standard_shared.rotary_embedding_pos.shape[0]);
  standard_shared.rotary_embedding_pos.shape[0] += meta.rotary_embedding_pos.GetElementNumber();
  MemcpyAsync(meta.rotary_embedding_pos.GetPtr<void>(), rotary_pos_host.data(),
              sizeof(decltype(rotary_pos_host)::value_type) * rotary_pos_host.size(), MEMCPY_HOST_TO_DEVICE,
              config_.context->device->GetD2HStream());

  EventRecord(config_.rotary_embedding_event, config_.context->device->GetD2HStream());
#ifdef ENABLE_ACL
  StreamSynchronize(config_.context->device->GetD2HStream());
#endif
}

}  // namespace ksana_llm
