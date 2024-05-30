/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/samplers/beam_search/beam_search_sampling.h"

namespace ksana_llm {

BeamSearchSampling::BeamSearchSampling(std::shared_ptr<Context> context) : context_(context) {}

void BeamSearchSampling::Update(std::vector<std::shared_ptr<InferRequest>>& req_group, int dst_idx, int src_idx,
                                int token_idx, float cumulative_score) {
  auto& dst_req = req_group[dst_idx];
  auto& src_req = req_group[src_idx];
  if (dst_idx != src_idx) {
    dst_req->output_tokens = src_req->output_tokens;
    dst_req->logprobs = src_req->logprobs;

    // 复制kv cache
    int block_token_num = GetBlockManager()->GetBlockTokenNum();
    size_t block_nums = (src_req->output_tokens.size() + block_token_num - 1) / block_token_num;

    int origin_device_id = GetBlockManager()->GetDeviceId();
    size_t block_size = GetBlockManager()->GetBlockSize();
    for (int device_id = 0; device_id < context_->GetTensorParallelSize(); device_id++) {
      GetBlockManager()->SetDeviceId(device_id);
      std::vector<void*> src_addrs, dst_addrs;
      GetBlockManager()->GetBlockPtrs(src_req->kv_cache_blocks[device_id], src_addrs);
      GetBlockManager()->GetBlockPtrs(dst_req->kv_cache_blocks[device_id], dst_addrs);
      NLLM_LOG_DEBUG << " device_id " << device_id << " src_addrs size " << src_addrs.size() << " dst_addrs size "
                     << dst_addrs.size() << " src step " << src_req->step;
      // TODO(zakwang): Exception handling
      for (size_t block_idx = 0; block_idx < block_nums && block_idx < src_addrs.size() && block_idx < dst_addrs.size();
           block_idx++) {
        MemcpyAsync(dst_addrs[block_idx], src_addrs[block_idx], block_size, MEMCPY_DEVICE_TO_DEVICE,
                    context_->GetD2DStreams()[device_id]);
      }
    }
    DeviceSynchronize();
    GetBlockManager()->SetDeviceId(origin_device_id);
  }
  dst_req->output_tokens.back() = dst_req->logprobs.back()[token_idx].first;
  dst_req->cumulative_score = cumulative_score;
}

Status BeamSearchSampling::Sampling(SamplingRequest& sampling_req) {
  int num_beams = sampling_req.sampling_config->num_beams;
  std::vector<std::shared_ptr<InferRequest>>& req_group = *sampling_req.req_group;
  if (num_beams > 1) {
    for (auto& beam_req : req_group) {
      if (sampling_req.output_tokens->size() > beam_req->output_tokens.size()) {
        NLLM_LOG_DEBUG << "CheckBeamSearch false";
        return Status();
      }
    }
    // context decode
    if (sampling_req.output_tokens->size() - sampling_req.input_tokens->size() == 1) {
      for (int i = 0; i < num_beams; i++) {
        auto& beam_req = req_group[i];
        beam_req->output_tokens.back() = (*sampling_req.logprobs)[0][i].first;
        beam_req->cumulative_score = (*sampling_req.logprobs)[0][i].second;
        beam_req->logprobs = (*sampling_req.logprobs);
      }
      return Status();
    }
    // beam search
    std::vector<std::pair<float, int>> output_source;
    int i = 0;
    for (auto& beam_req : req_group) {
      for (int top_index = 0; top_index < num_beams; top_index++, i++) {
        int beam_search_token = beam_req->logprobs.back()[top_index].first;
        float beam_search_token_logprob = beam_req->logprobs.back()[top_index].second;
        float now_cumulative_score = beam_req->cumulative_score + beam_search_token_logprob;
        std::vector<int>& stop_token_ids = beam_req->sampling_config.stop_token_ids;
        bool req_finish =
          beam_search_token == beam_req->end_id ||
          std::find(stop_token_ids.begin(), stop_token_ids.end(), beam_search_token) != stop_token_ids.end();
        if (req_finish) {
          size_t len = beam_req->output_tokens.size() - beam_req->input_tokens.size();
          auto cumulative_score = now_cumulative_score / pow(len, beam_req->sampling_config.length_penalty);

          std::vector<int> output_tokens = beam_req->output_tokens;
          output_tokens.back() = beam_search_token;
          beam_req->beam_search_group.push_back({output_tokens, beam_req->logprobs, cumulative_score});
          continue;
        }
        output_source.push_back({now_cumulative_score, i});
      }
    }
    std::sort(output_source.rbegin(), output_source.rend());
    // Processing beam-req that does not require copying.
    std::set<int> finished_set;
    for (i = 0; i < num_beams && i < (int)output_source.size(); i++) {
      int req_idx = output_source[i].second / num_beams;
      int token_idx = output_source[i].second % num_beams;
      if (token_idx == 0) {
        finished_set.insert(req_idx);
        Update(req_group, req_idx, req_idx, token_idx, output_source[i].first);
      }
    }
    // Processing that needs to be replicated.
    int dst_idx = 0;
    for (i = 0; i < num_beams && i < (int)output_source.size(); i++) {
      int req_idx = output_source[i].second / num_beams;
      int token_idx = output_source[i].second % num_beams;
      if (token_idx != 0) {
        while (finished_set.count(dst_idx)) dst_idx++;
        Update(req_group, dst_idx, req_idx, token_idx, output_source[i].first);
        dst_idx++;
      }
    }
  }

  return Status();
}

}  // namespace ksana_llm
