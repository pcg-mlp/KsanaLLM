/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "runtime/draft_generator/draft_tokens.h"
#include "runtime/profile_metrics.h"
#include "runtime/sampling_request.h"
#include "runtime/schedule_output.h"
#include "utils/logger.h"

namespace ksana_llm {

class GenerationResult {
 public:
  std::vector<int> accepted_tokens;
  std::vector<int> generated_tokens;
  DraftTokens draft_tokens;

  void SerializeToVector(std::vector<int>& encoded_tokens) const {
    encoded_tokens.clear();
    SerializeField(accepted_tokens, encoded_tokens);
    SerializeField(generated_tokens, encoded_tokens);
    SerializeField(draft_tokens.mtp, encoded_tokens);
    SerializeField(draft_tokens.spec, encoded_tokens);
    SerializeField(draft_tokens.ptp, encoded_tokens);
  }

  void DeserializeFromVector(const std::vector<int>& encoded_tokens) {
    size_t idx = 0;
    idx = DeserializeField(encoded_tokens, idx, accepted_tokens);
    idx = DeserializeField(encoded_tokens, idx, generated_tokens);
    idx = DeserializeField(encoded_tokens, idx, draft_tokens.mtp);
    idx = DeserializeField(encoded_tokens, idx, draft_tokens.spec);
    idx = DeserializeField(encoded_tokens, idx, draft_tokens.ptp);
    KLLM_CHECK(idx == encoded_tokens.size());
  }

 private:
  static void SerializeField(const std::vector<int>& field, std::vector<int>& out) {
    out.push_back(static_cast<int>(field.size()));
    out.insert(out.end(), field.begin(), field.end());
  }

  static size_t DeserializeField(const std::vector<int>& in, size_t idx, std::vector<int>& field) {
    KLLM_CHECK_WITH_INFO(idx < in.size(), "DeserializeField: idx out of range");
    KLLM_CHECK_WITH_INFO(in[idx] >= 0, "DeserializeField: negative field size");
    size_t len = static_cast<size_t>(in[idx++]);
    KLLM_CHECK_WITH_INFO(idx + len <= in.size(), "DeserializeField: field data out of range");
    field.assign(in.begin() + idx, in.begin() + idx + len);
    return idx + len;
  }
};

class SamplingOutput {
 public:
  size_t schedule_id = DEFAULT_SCHEDULE_ID;

  // The schedule output.
  std::shared_ptr<ScheduleOutput> schedule_output;

  // The sampling request.
  std::vector<SamplingRequest> sampling_reqs;

  // The sampling result.
  std::unordered_map<int64_t, std::vector<int>> sampling_result;

  // The samling top logprobs.
  std::unordered_map<int64_t, std::vector<std::vector<std::pair<int, float>>>> target_logprobs;

  // The target responses.
  std::unordered_map<int64_t, std::unordered_map<std::string, PythonTensor>> target_responses;

  // Batch-level performance metrics (only populated when enable_profile_metrics=true).
  bool has_profile_metrics = false;
  ProfileMetrics profile_metrics;
};

class SamplingOutputParser {
 public:
  // We just assume the data memory is large enough, and do not check it.
  static Status SerializeSamplingOutput(const SamplingOutput* sampling_output, void* ptr, size_t* bytes);
  static Status DeserializeSamplingOutput(const void* ptr, SamplingOutput* sampling_output);

  // Get the serialized byte of a SamplingOutput object.
  static size_t GetSerializedSize(const SamplingOutput* sampling_output);
};

// An object pool for schedule output.
class SamplingOutputPool {
 public:
  // Get a schedule output object.
  SamplingOutput* GetFreeSamplingOutput();

  // Free the schedule output to object pool.
  Status FreeSamplingOutput(SamplingOutput* sampling_output);

  // Put to and get from received buffer.
  void PutRecvSamplingOutput(SamplingOutput* sampling_output);
  SamplingOutput* GetRecvSamplingOutput();

  // Put to and get from send buffer.
  void PutSendSamplingOutput(SamplingOutput* sampling_output);
  Packet* GetSendSerializedPacket();

  // All blocked queue will be returned immediately.
  Status Stop();

 private:
  BlockingQueue<SamplingOutput*> free_queue_;
  BlockingQueue<SamplingOutput*> recv_queue_;

  // Send buffer.
  BlockingQueue<Packet*> serialized_send_queue_;
};

}  // namespace ksana_llm
