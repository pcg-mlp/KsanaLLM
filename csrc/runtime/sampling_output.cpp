/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/sampling_output.h"

#include "distributed/packet_util.h"
#include "ipc/core/ipc_func_serialization.h"
#include "profiler/profile_event.h"

namespace ksana_llm {

Status SamplingOutputParser::SerializeSamplingOutput(const SamplingOutput* sampling_output, void* ptr, size_t* bytes) {
  PROFILE_EVENT_SCOPE(serialize_sampling_output_total, "Executor.SerializeSamplingOutput.Total");
  SerializeInit(*bytes);
  // schedule_id
  SerializePod<size_t>(sampling_output->schedule_id, ptr, *bytes);

  // sampling_result
  SerializeKeyToVector<int64_t, int>(sampling_output->sampling_result, ptr, *bytes);

  // target_responses
  size_t target_response_size = 0;
  for (const auto& sampling_req : sampling_output->sampling_reqs) {
    if (sampling_req.response != nullptr) {
      ++target_response_size;
    }
  }
  SerializePod<size_t>(target_response_size, ptr, *bytes);
  for (const auto& sampling_req : sampling_output->sampling_reqs) {
    if (sampling_req.response != nullptr) {
      SerializePod<int64_t>(sampling_req.req_id, ptr, *bytes);
      // Serialize std::unordered_map<std::string, PythonTensor>
      SerializePod<size_t>(sampling_req.response->size(), ptr, *bytes);
      for (const auto& [target, response_tensor] : *sampling_req.response) {
        SerializeString(target, ptr, *bytes);
        SerializeVector<uint8_t>(response_tensor.data, ptr, *bytes);
        SerializeVector<size_t>(response_tensor.shape, ptr, *bytes);
        SerializeString(response_tensor.dtype, ptr, *bytes);
      }
    }
  }

  // target_logprobs
  size_t target_logprobs_size = 0;
  for (const auto& sampling_req : sampling_output->sampling_reqs) {
      if (!sampling_req.logprobs->empty()) {
        ++target_logprobs_size;
      }
    }
  SerializePod<size_t>(target_logprobs_size, ptr, *bytes);
  for (const auto& [req_id, req_logprobs] : sampling_output->target_logprobs) {
    if (!req_logprobs.empty()) {
      SerializePod<int64_t>(req_id, ptr, *bytes);
      // Serialize std::vector<std::vector<std::pair<int, float>>>
      SerializePod<size_t>(req_logprobs.size(), ptr, *bytes);
      for (const auto& topk_logprobs : req_logprobs) {
        SerializePod<size_t>(topk_logprobs.size(), ptr, *bytes);
        for (const auto& [token_id, logprob] : topk_logprobs) {
          SerializePod<int>(token_id, ptr, *bytes);
          SerializePod<float>(logprob, ptr, *bytes);
        }
      }
    }
  }

  // has_profile_metrics + profile_metrics（batch 级别，enable_profile_metrics=true 时填充）
  SerializePod<bool>(sampling_output->has_profile_metrics, ptr, *bytes);
  if (sampling_output->has_profile_metrics) {
    std::vector<int> pm_buf;
    sampling_output->profile_metrics.SerializeTo(pm_buf);
    SerializePod<size_t>(pm_buf.size(), ptr, *bytes);
    for (int v : pm_buf) {
      SerializePod<int>(v, ptr, *bytes);
    }
  }

  return Status();
}

Status SamplingOutputParser::DeserializeSamplingOutput(const void* ptr, SamplingOutput* sampling_output) {
  PROFILE_EVENT_SCOPE(deserialize_sampling_output_total, "Engine.DeserializeSamplingOutput.Total");
  size_t offset = 0;

  // schedule_id
  sampling_output->schedule_id = DeserializePod<size_t>(ptr, offset);

  // sampling_result
  sampling_output->sampling_result.clear();
  DeserializeKeyToVector<int64_t, int>(sampling_output->sampling_result, ptr, offset);

  // target_responses
  sampling_output->target_responses.clear();
  const size_t target_response_size = DeserializePod<size_t>(ptr, offset);
  for (size_t i = 0; i < target_response_size; i++) {
    const int64_t req_id = DeserializePod<size_t>(ptr, offset);
    const size_t response_size = DeserializePod<size_t>(ptr, offset);
    std::unordered_map<std::string, PythonTensor> response;
    // Deserialize std::unordered_map<std::string, PythonTensor>
    for (size_t j = 0; j < response_size; j++) {
      std::string target = DeserializeString(ptr, offset);
      PythonTensor response_tensor;
      DeserializeVector<uint8_t>(response_tensor.data, ptr, offset);
      DeserializeVector<size_t>(response_tensor.shape, ptr, offset);
      response_tensor.dtype = DeserializeString(ptr, offset);
      response.emplace(std::move(target), std::move(response_tensor));
    }
    sampling_output->target_responses.emplace(req_id, std::move(response));
  }

  // target_logprobs
  sampling_output->target_logprobs.clear();
  const size_t target_logprobs_size = DeserializePod<size_t>(ptr, offset);
  for (size_t i = 0; i < target_logprobs_size; i++) {
    const int64_t req_id = DeserializePod<int64_t>(ptr, offset);
    const size_t req_logprobs_size = DeserializePod<size_t>(ptr, offset);
    std::vector<std::vector<std::pair<int, float>>> req_logprobs;
    // Deserialize std::vector<std::vector<std::pair<int, float>>>
    for (size_t j = 0; j < req_logprobs_size; j++) {
      const size_t topk_logprobs_size = DeserializePod<size_t>(ptr, offset);
      std::vector<std::pair<int, float>> topk_logprobs;
      for (size_t k = 0; k < topk_logprobs_size; k++) {
        const int token_id = DeserializePod<int>(ptr, offset);
        const float logprob = DeserializePod<float>(ptr, offset);
        topk_logprobs.emplace_back(token_id, logprob);
      }
      req_logprobs.emplace_back(std::move(topk_logprobs));
    }
    sampling_output->target_logprobs.emplace(req_id, std::move(req_logprobs));
  }

  // has_profile_metrics + profile_metrics
  sampling_output->has_profile_metrics = DeserializePod<bool>(ptr, offset);
  sampling_output->profile_metrics = ProfileMetrics{};
  if (sampling_output->has_profile_metrics) {
    const size_t pm_size = DeserializePod<size_t>(ptr, offset);
    std::vector<int> pm_buf(pm_size);
    for (size_t i = 0; i < pm_size; i++) {
      pm_buf[i] = DeserializePod<int>(ptr, offset);
    }
    ProfileMetrics::DeserializeFrom(pm_buf, 0, sampling_output->profile_metrics);
  }

  return Status();
}

size_t SamplingOutputParser::GetSerializedSize(const SamplingOutput* sampling_output) {
  // Use a large enough memory temporarily, change to really size layer.
  return 64 * 1024 * 1024;
}

SamplingOutput* SamplingOutputPool::GetFreeSamplingOutput() {
  if (free_queue_.Empty()) {
    SamplingOutput* sampling_output = new SamplingOutput();
    return sampling_output;
  }

  return free_queue_.Get();
}

Status SamplingOutputPool::FreeSamplingOutput(SamplingOutput* sampling_output) {
  free_queue_.Put(sampling_output);

  return Status();
}

void SamplingOutputPool::PutRecvSamplingOutput(SamplingOutput* sampling_output) { recv_queue_.Put(sampling_output); }

SamplingOutput* SamplingOutputPool::GetRecvSamplingOutput() { return recv_queue_.Get(); }

void SamplingOutputPool::PutSendSamplingOutput(SamplingOutput* sampling_output) {
  size_t sampling_output_size = SamplingOutputParser::GetSerializedSize(sampling_output);

  Packet* packet = GetRawPacket(sampling_output_size);
  if (packet == nullptr) {
    throw std::runtime_error("SamplingOutputPool::PutToSendQueue allocate memory error.");
  }

  size_t serialized_bytes;
  packet->type = PacketType::CONTROL_REQ_SAMPLING_OUTPUT;
  SamplingOutputParser::SerializeSamplingOutput(sampling_output, packet->body, &serialized_bytes);
  packet->size = serialized_bytes;

  serialized_send_queue_.Put(packet);
}

Packet* SamplingOutputPool::GetSendSerializedPacket() { return serialized_send_queue_.Get(); }

Status SamplingOutputPool::Stop() {
  free_queue_.Stop();
  serialized_send_queue_.Stop();
  recv_queue_.Stop();

  return Status();
}

}  // namespace ksana_llm
