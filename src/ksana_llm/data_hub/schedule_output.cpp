/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/data_hub/schedule_output.h"

#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/distributed/packet_util.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

std::string ScheduleOutput::ToString() {
  std::string result;
  result += "{\n";
  result += "  schedule_id: " + std::to_string(schedule_id) + "\n";
  result += "  finish_req_ids: " + Vector2Str(finish_req_ids) + "\n";
  result += "  merged_swapout_req_ids: " + Vector2Str(merged_swapout_req_ids) + "\n";
  result += "  merged_swapin_req_ids: " + Vector2Str(merged_swapin_req_ids) + "\n";
  result += "  merged_swapin_req_ids: " + Vector2Str(merged_swapin_req_ids) + "\n";

  result += "  swapout_req_block_ids:\n";
  for (auto pair : swapout_req_block_ids) {
    result += "    " + std::to_string(pair.first) + ": " + Vector2Str(pair.second) + "\n";
  }

  result += "  swapin_req_block_ids:\n";
  for (auto pair : swapin_req_block_ids) {
    result += "    " + std::to_string(pair.first) + ": " + Vector2Str(pair.second) + "\n";
  }

  result += "  running_reqs:\n";
  result += InferRequestToString(running_reqs);

  result += "  worker_running_reqs:\n";
  result += InferRequestToString(worker_running_reqs);

  result += "}";

  return result;
}

size_t ScheduleOutputParser::GetSerializedSize(const ScheduleOutput* schedule_output) {
  size_t serialized_bytes = 0;

  // schedule_id
  serialized_bytes += sizeof(size_t);

  // finish_req_ids
  serialized_bytes += sizeof(int);
  serialized_bytes += schedule_output->finish_req_ids.size() * sizeof(int64_t);

  // merged swapout reqs
  serialized_bytes += sizeof(int);
  serialized_bytes += schedule_output->merged_swapout_req_ids.size() * sizeof(int64_t);

  // merged swapin reqs
  serialized_bytes += sizeof(int);
  serialized_bytes += schedule_output->merged_swapin_req_ids.size() * sizeof(int64_t);

  // swapout req with blocks.
  serialized_bytes += sizeof(int);
  for (auto& [k, v] : schedule_output->swapout_req_block_ids) {
    // key
    serialized_bytes += sizeof(int64_t);

    // value
    serialized_bytes += sizeof(int);
    serialized_bytes += v.size() * sizeof(int);
  }

  // swapin req with blocks.
  serialized_bytes += sizeof(int);
  for (auto& [k, v] : schedule_output->swapin_req_block_ids) {
    // key
    serialized_bytes += sizeof(int64_t);

    // value
    serialized_bytes += sizeof(int);
    serialized_bytes += v.size() * sizeof(int);
  }

  // running reqs.
  serialized_bytes += sizeof(int);

  for (auto req : schedule_output->running_reqs) {
    // req_id
    serialized_bytes += sizeof(int64_t);

    // model_name
    serialized_bytes += sizeof(int);
    serialized_bytes += req->model_name.size();

    // output_tokens
    serialized_bytes += sizeof(int);
    serialized_bytes += req->output_tokens.size() * sizeof(int);

    // infer_stage
    serialized_bytes += sizeof(InferStage);

    // step
    serialized_bytes += sizeof(int);

    // kv_cache_blocks
    serialized_bytes += sizeof(int);
    for (auto& v : req->kv_cache_blocks) {
      serialized_bytes += sizeof(int);
      serialized_bytes += v.size() * sizeof(int);
    }

    // is_use_prefix_cache
    serialized_bytes += sizeof(bool);

    // prefix_cache_len
    serialized_bytes += sizeof(int);

    // prefix_cache_blocks_number
    serialized_bytes += sizeof(int);

    // kv_cached_token_num
    serialized_bytes += sizeof(int);

    // mrotary_embedding_pos_offset
    serialized_bytes += sizeof(int64_t);
  }

  return serialized_bytes;
}

Status ScheduleOutputParser::SerializeAsWorkerInferRequest(const std::vector<std::shared_ptr<InferRequest>>& infer_reqs,
                                                   void* data, size_t& bytes) {
  size_t offset = 0;

  int req_num = infer_reqs.size();
  std::memcpy(data + offset, &req_num, sizeof(int));
  offset += sizeof(int);

  size_t inner_bytes;

  for (auto req : infer_reqs) {
    // req_id
    std::memcpy(data + offset, &req->req_id, sizeof(int64_t));
    offset += sizeof(int64_t);

    // model_name
    int model_name_size = req->model_name.size();
    std::memcpy(data + offset, &model_name_size, sizeof(int));
    offset += sizeof(int);

    std::memcpy(data + offset, req->model_name.data(), req->model_name.size());
    offset += req->model_name.size();

    // output_tokens
    SerializeVector(req->output_tokens, data + offset, inner_bytes);
    offset += inner_bytes;

    // infer_stage
    std::memcpy(data + offset, &req->infer_stage, sizeof(InferStage));
    offset += sizeof(InferStage);

    // step
    std::memcpy(data + offset, &req->step, sizeof(int));
    offset += sizeof(int);

    // kv_cache_blocks
    SerializeVectorOfVector(req->kv_cache_blocks, data + offset, inner_bytes);
    offset += inner_bytes;

    // is_use_prefix_cache
    std::memcpy(data + offset, &req->is_use_prefix_cache, sizeof(bool));
    offset += sizeof(bool);

    // prefix_cache_len
    std::memcpy(data + offset, &req->prefix_cache_len, sizeof(int));
    offset += sizeof(int);

    // prefix_cache_blocks_number
    std::memcpy(data + offset, &req->prefix_cache_blocks_number, sizeof(int));
    offset += sizeof(int);

    // kv_cached_token_num
    std::memcpy(data + offset, &req->kv_cached_token_num, sizeof(int));
    offset += sizeof(int);

    // mrotary_embedding_pos_offset
    std::memcpy(data + offset, &req->mrotary_embedding_pos_offset, sizeof(int64_t));
    offset += sizeof(int64_t);
  }
  bytes = offset;

  return Status();
}

Status ScheduleOutputParser::DeserializeWorkerInferRequest(
    void* data, std::vector<std::shared_ptr<WorkerInferRequest>>& worker_infer_reqs, size_t& bytes) {
  size_t offset = 0;

  int req_num = *reinterpret_cast<int*>(data + offset);
  offset += sizeof(int);

  size_t inner_bytes;

  for (size_t i = 0; i < req_num; ++i) {
    std::shared_ptr<WorkerInferRequest> req = std::make_shared<WorkerInferRequest>();

    // req_id
    req->req_id = *reinterpret_cast<int64_t*>(data + offset);
    offset += sizeof(int64_t);

    // model_name
    int model_name_size = *reinterpret_cast<int*>(data + offset);
    offset += sizeof(int);

    std::string model_name;
    model_name.assign(reinterpret_cast<char*>(data + offset), model_name_size);
    req->model_name = model_name;
    offset += model_name_size;

    // output_tokens
    std::vector<int> output_tokens;
    DeserializeVector(data + offset, output_tokens, inner_bytes);
    req->output_tokens = output_tokens;
    offset += inner_bytes;

    // infer_stage
    req->infer_stage = *reinterpret_cast<InferStage*>(data + offset);
    offset += sizeof(InferStage);

    // step
    req->step = *reinterpret_cast<int*>(data + offset);
    offset += sizeof(int);

    // kv_cache_blocks
    std::vector<std::vector<int>> kv_cache_blocks;
    DeserializeVectorOfVector(data + offset, kv_cache_blocks, inner_bytes);
    req->kv_cache_blocks = kv_cache_blocks;
    offset += inner_bytes;

    // is_use_prefix_cache
    req->is_use_prefix_cache = *reinterpret_cast<bool*>(data + offset);
    offset += sizeof(bool);

    // prefix_cache_len
    req->prefix_cache_len = *reinterpret_cast<int*>(data + offset);
    offset += sizeof(int);

    // prefix_cache_blocks_number
    req->prefix_cache_blocks_number = *reinterpret_cast<int*>(data + offset);
    offset += sizeof(int);

    // kv_cached_token_num
    req->kv_cached_token_num = *reinterpret_cast<int*>(data + offset);
    offset += sizeof(int);

    // mrotary_embedding_pos_offset.
    req->mrotary_embedding_pos_offset = *reinterpret_cast<int64_t*>(data + offset);
    offset += sizeof(int64_t);

    // Get model instance from data hub.
    req->model_instance = GetModelInstance(req->model_name);

    worker_infer_reqs.push_back(req);
  }
  bytes = offset;

  return Status();
}

Status ScheduleOutputParser::SerializeScheduleOutput(const ScheduleOutput* schedule_output, void* data) {
  size_t offset = 0;

  // schedule_id
  std::memcpy(data + offset, &schedule_output->schedule_id, sizeof(size_t));
  offset += sizeof(size_t);

  size_t bytes;

  // finished reqs.
  SerializeVector(schedule_output->finish_req_ids, data + offset, bytes);
  offset += bytes;

  // merged swapout reqs
  SerializeVector(schedule_output->merged_swapout_req_ids, data + offset, bytes);
  offset += bytes;

  // merged swapin reqs
  SerializeVector(schedule_output->merged_swapin_req_ids, data + offset, bytes);
  offset += bytes;

  // swapout req with blocks.
  SerializeKeyToVector(schedule_output->swapout_req_block_ids, data + offset, bytes);
  offset += bytes;

  // swapin req with blocks.
  SerializeKeyToVector(schedule_output->swapin_req_block_ids, data + offset, bytes);
  offset += bytes;

  // running reqs.
  SerializeAsWorkerInferRequest(schedule_output->running_reqs, data + offset, bytes);
  offset += bytes;

  return Status();
}

Status ScheduleOutputParser::DeserializeScheduleOutput(void* data, ScheduleOutput* schedule_output) {
  size_t offset = 0;

  // schedule_id
  schedule_output->schedule_id = *reinterpret_cast<size_t*>(data + offset);
  offset += sizeof(size_t);

  size_t bytes;

  // finished reqs
  std::vector<int64_t> finish_req_ids;
  DeserializeVector(data + offset, finish_req_ids, bytes);
  schedule_output->finish_req_ids = finish_req_ids;
  offset += bytes;

  // merged swapout reqs
  std::vector<int64_t> merged_swapout_req_ids;
  DeserializeVector(data + offset, merged_swapout_req_ids, bytes);
  schedule_output->merged_swapout_req_ids = merged_swapout_req_ids;
  offset += bytes;

  // merged swapin reqs
  std::vector<int64_t> merged_swapin_req_ids;
  DeserializeVector(data + offset, merged_swapin_req_ids, bytes);
  schedule_output->merged_swapin_req_ids = merged_swapin_req_ids;
  offset += bytes;

  // swapout req with blocks.
  std::unordered_map<int64_t, std::vector<int>> swapout_req_block_ids;
  DeserializeKeyToVector(data + offset, swapout_req_block_ids, bytes);
  schedule_output->swapout_req_block_ids = swapout_req_block_ids;
  offset += bytes;

  // swapin req with blocks.
  std::unordered_map<int64_t, std::vector<int>> swapin_req_block_ids;
  DeserializeKeyToVector(data + offset, swapin_req_block_ids, bytes);
  schedule_output->swapin_req_block_ids = swapin_req_block_ids;
  offset += bytes;

  // running reqs.
  std::vector<std::shared_ptr<WorkerInferRequest>> worker_running_reqs;
  DeserializeWorkerInferRequest(data + offset, worker_running_reqs, bytes);
  schedule_output->worker_running_reqs = worker_running_reqs;
  offset += bytes;

  return Status();
}

ScheduleOutput* ScheduleOutputPool::GetScheduleOutput() {
  if (schedule_output_free_buffers_.Empty()) {
    ScheduleOutput* schedule_output = new ScheduleOutput();
    return schedule_output;
  }

  return schedule_output_free_buffers_.Get();
}

Status ScheduleOutputPool::FreeScheduleOutput(ScheduleOutput* schedule_output) {
  schedule_output->Clear();
  schedule_output_free_buffers_.Put(schedule_output);

  return Status();
}

Status ScheduleOutputPool::PutToRecvQueue(ScheduleOutput* schedule_output) {
  schedule_output_recv_buffers_.Put(schedule_output);
  return Status();
}

ScheduleOutput* ScheduleOutputPool::GetFromRecvQueue() { return schedule_output_recv_buffers_.Get(); }

Status ScheduleOutputPool::PutToSendQueue(ScheduleOutput* schedule_output) {
  size_t schedule_output_size = ScheduleOutputParser::GetSerializedSize(schedule_output);

  Packet* packet = GetRawPacket(schedule_output_size);
  if (packet == nullptr) {
    throw std::runtime_error("ControlChannel::ProcessSendScheduleLoop allocate memory error.");
  }

  packet->type = PacketType::CONTROL_REQ_SCHEDULE;
  ScheduleOutputParser::SerializeScheduleOutput(schedule_output, packet->body);

  schedule_output_send_buffers_.Put(packet);
  return Status();
}

Packet* ScheduleOutputPool::GetFromSendQueue() { return schedule_output_send_buffers_.Get(); }

Status ScheduleOutputPool::Stop() {
  schedule_output_free_buffers_.Stop();
  schedule_output_send_buffers_.Stop();
  schedule_output_recv_buffers_.Stop();

  return Status();
}

}  // namespace ksana_llm
