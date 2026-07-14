/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "runtime/sampling_request.h"
#include "runtime/schedule_output.h"
#include "utils/logger.h"

namespace ksana_llm {

class GenerationResult {
 public:
  std::vector<int> accepted_tokens;
  std::vector<int> generated_tokens;
  std::vector<int> draft_tokens;
  void SerializeToVector(std::vector<int>& encoded_tokens) const {
    encoded_tokens.clear();
    encoded_tokens.push_back(accepted_tokens.size());
    encoded_tokens.insert(encoded_tokens.end(), accepted_tokens.begin(), accepted_tokens.end());
    encoded_tokens.push_back(generated_tokens.size());
    encoded_tokens.insert(encoded_tokens.end(), generated_tokens.begin(), generated_tokens.end());
    encoded_tokens.push_back(draft_tokens.size());
    encoded_tokens.insert(encoded_tokens.end(), draft_tokens.begin(), draft_tokens.end());
  }

  void DeserializeFromVector(const std::vector<int>& encoded_tokens) {
    size_t idx = 0;
    size_t accepted_size = encoded_tokens[idx++];
    accepted_tokens.resize(accepted_size);
    std::copy(encoded_tokens.begin() + idx, encoded_tokens.begin() + idx + accepted_size, accepted_tokens.begin());
    idx += accepted_size;
    size_t generated_size = encoded_tokens[idx++];
    generated_tokens.resize(generated_size);
    std::copy(encoded_tokens.begin() + idx, encoded_tokens.begin() + idx + generated_size, generated_tokens.begin());
    idx += generated_size;
    size_t draft_size = encoded_tokens[idx++];
    draft_tokens.resize(draft_size);
    std::copy(encoded_tokens.begin() + idx, encoded_tokens.begin() + idx + draft_size, draft_tokens.begin());
    KLLM_CHECK((accepted_size + generated_size + draft_size + 3) == encoded_tokens.size());
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
