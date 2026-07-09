/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "runtime/schedule_output.h"

#include <torch/torch.h>

#include "batch_scheduler/schedule_processor_interface.h"
#include "distributed/packet_util.h"
#include "ipc/core/ipc_func_serialization.h"
#include "profiler/profile_event.h"
#include "runtime/executor_infer_request.h"
#include "utils/status.h"
#include "utils/string_utils.h"

namespace ksana_llm {

// The global schedule processor, used to make mutex of serialization and schedule.
std::shared_ptr<ScheduleProcessorInterface> ScheduleOutputParser::schedule_processor_ = nullptr;

void SerializeTensor(const Tensor& tensor, void* ptr, size_t& offset) {
  KLLM_CHECK_WITH_INFO(tensor.location == MemoryLocation::LOCATION_HOST, "Only supports serializing host tensor.");

  SerializeVector<size_t>(tensor.shape, ptr, offset);

  int32_t dtype_value = static_cast<int32_t>(tensor.dtype);
  SerializePod<int32_t>(dtype_value, ptr, offset);

  size_t data_size = tensor.GetTotalBytes();
  SerializePod<size_t>(data_size, ptr, offset);

  uint8_t* dst = reinterpret_cast<uint8_t*>(ptr);
  std::memcpy(dst + offset, tensor.GetPtr<void>(), data_size);
  offset += data_size;
}

void DeserializeTensor(Tensor& tensor, const void* ptr, size_t& offset) {
  std::vector<size_t> shape;
  DeserializeVector<size_t>(shape, ptr, offset);

  int32_t dtype_value = DeserializePod<int32_t>(ptr, offset);
  DataType dtype = static_cast<DataType>(dtype_value);

  size_t data_size = DeserializePod<size_t>(ptr, offset);

  const uint8_t* src = reinterpret_cast<const uint8_t*>(ptr);

  tensor = Tensor(MemoryLocation::LOCATION_HOST, dtype, shape);
  std::memcpy(tensor.GetPtr<void>(), src + offset, data_size);
  offset += data_size;
}

void SerializeVectorOfTensor(const std::vector<Tensor>& tensors, void* ptr, size_t& offset) {
  SerializePod<size_t>(tensors.size(), ptr, offset);
  for (size_t i = 0; i < tensors.size(); i++) {
    SerializeTensor(tensors[i], ptr, offset);
  }
}

void DeserializeVectorOfTensor(std::vector<Tensor>& tensors, const void* ptr, size_t& offset) {
  size_t nums = DeserializePod<size_t>(ptr, offset);
  tensors.resize(nums);
  for (size_t i = 0; i < nums; ++i) {
    DeserializeTensor(tensors[i], ptr, offset);
  }
}

static Status SerializeEmbeddingSlice(const EmbeddingSlice& embedding_slice, void* ptr, size_t& offset) {
  // pos
  SerializeVector<int>(embedding_slice.pos, ptr, offset);

  // embeddings
  SerializeVectorOfVector<float>(embedding_slice.embeddings, ptr, offset);

  // embedding_tensors
  SerializeVectorOfTensor(embedding_slice.embedding_tensors, ptr, offset);

  // additional_tensors
  SerializeVectorOfTensor(embedding_slice.additional_tensors, ptr, offset);

  return Status();
}

static Status DeserializeEmbeddingSlice(EmbeddingSlice& embedding_slice, const void* ptr, size_t& offset) {
  // pos
  DeserializeVector<int>(embedding_slice.pos, ptr, offset);

  // embeddings
  DeserializeVectorOfVector<float>(embedding_slice.embeddings, ptr, offset);

  // embedding_tensors
  DeserializeVectorOfTensor(embedding_slice.embedding_tensors, ptr, offset);

  // additional_tensors
  DeserializeVectorOfTensor(embedding_slice.additional_tensors, ptr, offset);

  return Status();
}

static Status SerializeSamplingConfig(const SamplingConfig& sampling_config, void* ptr, size_t& offset) {
  // topk
  SerializePod<int>(sampling_config.topk, ptr, offset);

  // num_beams
  SerializePod<int>(sampling_config.num_beams, ptr, offset);

  // num_return_sequences
  SerializePod<int>(sampling_config.num_return_sequences, ptr, offset);

  // topp
  SerializePod<float>(sampling_config.topp, ptr, offset);

  // temperature
  SerializePod<float>(sampling_config.temperature, ptr, offset);

  // repetition_penalty
  SerializePod<float>(sampling_config.repetition_penalty, ptr, offset);

  // length_penalty
  SerializePod<float>(sampling_config.length_penalty, ptr, offset);

  // stop_token_ids
  SerializeVector<int>(sampling_config.stop_token_ids, ptr, offset);

  // ignore_eos
  SerializePod<bool>(sampling_config.ignore_eos, ptr, offset);

  // max_new_tokens
  SerializePod<int>(sampling_config.max_new_tokens, ptr, offset);

  // logprobs_num
  SerializePod<int>(sampling_config.logprobs_num, ptr, offset);

  // no_repeat_ngram_size
  SerializePod<int>(sampling_config.no_repeat_ngram_size, ptr, offset);

  // encoder_no_repeat_ngram_size
  SerializePod<int>(sampling_config.encoder_no_repeat_ngram_size, ptr, offset);

  // decoder_no_repeat_ngram_size
  SerializePod<int>(sampling_config.decoder_no_repeat_ngram_size, ptr, offset);

  // stop_strings
  SerializeVector<std::string>(sampling_config.stop_strings, ptr, offset);

  // json_schema
  SerializeString(sampling_config.json_schema, ptr, offset);

  // enable_structured_output
  SerializePod<bool>(sampling_config.enable_structured_output, ptr, offset);

  // enable_thinking
  SerializePod<bool>(sampling_config.enable_thinking, ptr, offset);

  // ptp_token_num
  SerializePod<int>(sampling_config.ptp_token_num, ptr, offset);

  return Status();
}

static Status SerializeStructuredGeneratorConfig(const StructuredGeneratorConfig& config, void* ptr, size_t& offset) {
  // constraint_type
  SerializeEnum<StructuredConstraintType>(config.constraint_type, ptr, offset);

  // constraint_spec
  SerializeString(config.constraint_spec, ptr, offset);

  return Status();
}

static Status DeserializeSamplingConfig(SamplingConfig& sampling_config, const void* ptr, size_t& offset) {
  // topk
  sampling_config.topk = DeserializePod<int>(ptr, offset);

  // num_beams
  sampling_config.num_beams = DeserializePod<int>(ptr, offset);

  // num_return_sequences
  sampling_config.num_return_sequences = DeserializePod<int>(ptr, offset);

  // topp
  sampling_config.topp = DeserializePod<float>(ptr, offset);

  // temperature
  sampling_config.temperature = DeserializePod<float>(ptr, offset);

  // repetition_penalty
  sampling_config.repetition_penalty = DeserializePod<float>(ptr, offset);

  // length_penalty
  sampling_config.length_penalty = DeserializePod<float>(ptr, offset);

  // stop_token_ids
  DeserializeVector<int>(sampling_config.stop_token_ids, ptr, offset);

  // ignore_eos
  sampling_config.ignore_eos = DeserializePod<bool>(ptr, offset);

  // max_new_tokens
  sampling_config.max_new_tokens = DeserializePod<int>(ptr, offset);

  // logprobs_num
  sampling_config.logprobs_num = DeserializePod<int>(ptr, offset);

  // no_repeat_ngram_size
  sampling_config.no_repeat_ngram_size = DeserializePod<int>(ptr, offset);

  // encoder_no_repeat_ngram_size
  sampling_config.encoder_no_repeat_ngram_size = DeserializePod<int>(ptr, offset);

  // decoder_no_repeat_ngram_size
  sampling_config.decoder_no_repeat_ngram_size = DeserializePod<int>(ptr, offset);

  // top_strings
  DeserializeVector<std::string>(sampling_config.stop_strings, ptr, offset);

  // json_schema
  sampling_config.json_schema = DeserializeString(ptr, offset);

  // enable_structured_output
  sampling_config.enable_structured_output = DeserializePod<bool>(ptr, offset);

  // enable_thinking
  sampling_config.enable_thinking = DeserializePod<bool>(ptr, offset);

  // ptp_token_num
  sampling_config.ptp_token_num = DeserializePod<int>(ptr, offset);

  return Status();
}

static Status DeserializeStructuredGeneratorConfig(StructuredGeneratorConfig& config, const void* ptr, size_t& offset) {
  // constraint_type
  config.constraint_type = DeserializeEnum<StructuredConstraintType>(ptr, offset);

  // constraint_spec
  config.constraint_spec = DeserializeString(ptr, offset);

  KLLM_LOG_INFO << "Deserialized StructuredGeneratorConfig: "
                << "constraint_type=" << static_cast<int>(config.constraint_type)
                << ", constraint_spec=" << config.constraint_spec;

  return Status();
}

// full transfer 序列化时优先读 Launch 时刻快照中的易竞态字段 (forwarding_tokens / kv_cached_token_num 等),
// 避免异步序列化线程读到后续 LaunchPlanningTask 已修改的 req->*.
static const InferRequestSerializationSnapshot* FindReqSnapshot(
    const std::unordered_map<int64_t, InferRequestSerializationSnapshot>* req_snapshots, int64_t req_id) {
  if (req_snapshots == nullptr) {
    return nullptr;
  }
  auto it = req_snapshots->find(req_id);
  if (it == req_snapshots->end()) {
    return nullptr;
  }
  return &it->second;
}

static Status SerializeInferRequests(
    const std::vector<std::shared_ptr<InferRequest>>& infer_reqs,
    const std::unordered_map<int64_t, InferRequestSerializationSnapshot>* req_snapshots, void* ptr, size_t& offset) {
  // vec size
  SerializePod<size_t>(infer_reqs.size(), ptr, offset);

  for (auto req : infer_reqs) {
    const InferRequestSerializationSnapshot* snap = FindReqSnapshot(req_snapshots, req->req_id);
    // req_id
    SerializePod<int64_t>(req->req_id, ptr, offset);

    // logits_custom_length
    SerializePod<size_t>(req->logits_custom_length, ptr, offset);

    // sampling_token_num
    SerializePod<size_t>(snap ? snap->sampling_token_num : req->sampling_token_num, ptr, offset);

    // last_step_token_num
    SerializePod<size_t>(req->last_step_token_num, ptr, offset);

    // input_tokens
    SerializeVector<int>(req->input_tokens, ptr, offset);

    // mrotary_embedding_pos_offset
    SerializePod<int64_t>(req->mrotary_embedding_pos_offset, ptr, offset);

    // xdrotary_embedding_pos_offset
    SerializePod<int64_t>(req->xdrotary_embedding_pos_offset, ptr, offset);

    // output_tokens
    SerializeVector<int>(req->output_tokens, ptr, offset);

    // draft_tokens
    SerializeVector<int>(req->draft_tokens.mtp, ptr, offset);
    SerializeVector<int>(req->draft_tokens.spec, ptr, offset);
    SerializeVector<int>(req->draft_tokens.ptp, ptr, offset);

    // accepted_hidden_states, not supported.

    // suggested_draft_num, not supported.

    // accepted_tokens
    SerializeVector<int>(req->accepted_tokens, ptr, offset);

    // forwarding_tokens_draft_num
    SerializePod<int64_t>(snap ? snap->forwarding_tokens_draft_num : req->forwarding_tokens_draft_num, ptr, offset);

    // generated_tokens
    SerializeVector<int>(req->generated_tokens, ptr, offset);

    // logprobs, not supported

    // request_target
    // Serialize std::unordered_map<std::string, TargetDescribe>
    SerializePod<size_t>(req->request_target.size(), ptr, offset);
    for (const auto& [target, request_describe] : req->request_target) {
      SerializeString(target, ptr, offset);
      SerializeVector<int>(request_describe.cutoff_layer, ptr, offset);
      SerializeVector<int>(request_describe.token_id, ptr, offset);
      SerializeVector<std::pair<int, int>>(request_describe.slice_pos, ptr, offset);
      SerializeEnum<TokenReduceMode>(request_describe.token_reduce_mode, ptr, offset);
      SerializePod<int>(request_describe.input_top_logprobs_num, ptr, offset);
    }

    // response doesn't need to be transferred

    // sampling_config
    SerializeSamplingConfig(req->sampling_config, ptr, offset);

    // attn_dp_group_id
    SerializePod<uint32_t>(req->attn_dp_group_id, ptr, offset);

    // kv_comm_group_key
    SerializeString(req->kv_comm_group_key, ptr, offset);

    // forwarding_tokens
    if (snap) {
      SerializeVector<int>(snap->forwarding_tokens, ptr, offset);
    } else {
      SerializeVector<int>(req->forwarding_tokens, ptr, offset);
    }

    // sampling_result_tokens
    SerializeVector<int>(req->sampling_result_tokens, ptr, offset);

    // infer_stage
    SerializeEnum<InferStage>(req->infer_stage, ptr, offset);

    // step
    SerializePod<int>(snap ? snap->step : req->step, ptr, offset);

    // kv_cached_token_num
    SerializePod<int>(snap ? snap->kv_cached_token_num : req->kv_cached_token_num, ptr, offset);

    // kv_from_remote (pd_v2). Mirrors the patch path; see SerializeInferRequestPatches.
    // Without this, the FIRST-time send of a PD request lands on the
    // Executor with kv_from_remote=false, which routes ForwardRequest::GetType
    // to kFlash (prefill kernel) — Decode then re-prefills the prompt and
    // overwrites the bytes peer Prefill RDMA-wrote, producing degenerate
    // output.
    SerializePod<bool>(req->pd_v2.kv_from_remote, ptr, offset);

    // kv_cache_blocks
    if (snap) {
      SerializeVector<int>(snap->kv_cache_blocks, ptr, offset);
    } else {
      SerializeVector<int>(req->kv_cache_blocks, ptr, offset);
    }

    // swa_kv_cache_blocks
    if (snap) {
      SerializeVector<int>(snap->swa_kv_cache_blocks, ptr, offset);
    } else {
      SerializeVector<int>(req->swa_kv_cache_blocks, ptr, offset);
    }

    // swa_kv_cache_block_idx_offsets
    if (snap) {
      SerializeVector<size_t>(snap->swa_kv_cache_block_idx_offsets, ptr, offset);
    } else {
      SerializeVector<size_t>(req->swa_kv_cache_block_idx_offsets, ptr, offset);
    }

    // block_token_num
    SerializePod<size_t>(req->block_token_num, ptr, offset);

    // logits_offset
    SerializePod<size_t>(req->logits_offset, ptr, offset);

    // prefix_cache_len
    SerializePod<int>(snap ? snap->prefix_cache_len : req->prefix_cache_len, ptr, offset);

    // flexible_cached_copy_tasks, not supported

    // ngram_dict, not supported.

    // timestamp_in_us, not supported.

    // req_ctx, not supported.

    // incremental_decoded_str
    SerializeString(req->incremental_decoded_str, ptr, offset);

    // input_refit_embedding
    // NOTE(jinxcwu) 只在context需要多模态数据，需要放在infer_stage后面
    // TODO(jinxcwu) 这里好像只调用一次，可以不用做判断
    if (req->infer_stage == InferStage::kContext) {
      SerializeEmbeddingSlice(req->input_refit_embedding, ptr, offset);
    }

    // need_recreate_generator (serialize first to decide whether to serialize config)
    SerializePod<bool>(req->need_recreate_generator, ptr, offset);

    // structured_generator_config (only serialize if need_recreate_generator is true)
    if (req->need_recreate_generator) {
      SerializeStructuredGeneratorConfig(req->structured_generator_config, ptr, offset);
    }

    // pd_v2: Decode-side KV pool routing. Travels with the **first**
    // (full) ExecutorInferRequest only — same convention as
    // structured_generator_config above. Subsequent patches don't
    // re-transmit. Empty vector = non-pd_v2, no overhead.
    SerializePod<size_t>(req->pd_v2.decode_targets.size(), ptr, offset);
    for (const auto& t : req->pd_v2.decode_targets) {
      SerializePod<uint32_t>(t.decode_attn_tp_rank_in_group, ptr, offset);
      SerializePod<uint64_t>(t.kv_pool_base_va, ptr, offset);
      SerializeVector<int>(t.kv_block_ids, ptr, offset);
    }
    SerializeString(req->pd_v2.decode_engine_inference_addr, ptr, offset);
    SerializePod<uint32_t>(req->pd_v2.decode_attn_dp_group_id, ptr, offset);
  }

  return Status();
}

static Status DeserializeInferRequests(std::vector<std::shared_ptr<ExecutorInferRequest>>& infer_reqs, const void* ptr,
                                       size_t& offset) {
  // vec size
  size_t vec_size = DeserializePod<size_t>(ptr, offset);

  for (size_t i = 0; i < vec_size; ++i) {
    std::shared_ptr<ExecutorInferRequest> req = std::make_shared<ExecutorInferRequest>();

    // req_id
    req->req_id = DeserializePod<int64_t>(ptr, offset);

    // logits_custom_length
    req->logits_custom_length = DeserializePod<size_t>(ptr, offset);

    // sampling_token_num
    req->sampling_token_num = DeserializePod<size_t>(ptr, offset);

    // last_step_token_num
    req->last_step_token_num = DeserializePod<size_t>(ptr, offset);

    // input_tokens
    DeserializeVector<int>(req->input_tokens, ptr, offset);

    // mrotary_embedding_pos_offset
    req->mrotary_embedding_pos_offset = DeserializePod<int64_t>(ptr, offset);

    // xdrotary_embedding_pos_offset
    req->xdrotary_embedding_pos_offset = DeserializePod<int64_t>(ptr, offset);

    // output_tokens
    DeserializeVector<int>(req->output_tokens, ptr, offset);

    // draft_tokens
    DeserializeVector<int>(req->draft_tokens.mtp, ptr, offset);
    DeserializeVector<int>(req->draft_tokens.spec, ptr, offset);
    DeserializeVector<int>(req->draft_tokens.ptp, ptr, offset);

    // accepted_hidden_states, not supported.

    // suggested_draft_num, not supported.

    // accepted_tokens
    DeserializeVector<int>(req->accepted_tokens, ptr, offset);

    // forwarding_tokens_draft_num
    req->forwarding_tokens_draft_num = DeserializePod<int64_t>(ptr, offset);

    // generated_tokens
    DeserializeVector<int>(req->generated_tokens, ptr, offset);

    // logprobs, not supported

    // request_target
    // Deserialize std::unordered_map<std::string, TargetDescribe>
    const size_t request_target_size = DeserializePod<size_t>(ptr, offset);
    for (size_t i = 0; i < request_target_size; i++) {
      const std::string target = DeserializeString(ptr, offset);
      TargetDescribe request_describe;
      DeserializeVector<int>(request_describe.cutoff_layer, ptr, offset);
      DeserializeVector<int>(request_describe.token_id, ptr, offset);
      DeserializeVector<std::pair<int, int>>(request_describe.slice_pos, ptr, offset);
      request_describe.token_reduce_mode = DeserializeEnum<TokenReduceMode>(ptr, offset);
      request_describe.input_top_logprobs_num = DeserializePod<int>(ptr, offset);
      req->request_target.emplace(target, request_describe);
    }

    // response doesn't need to be transferred

    // sampling_config
    DeserializeSamplingConfig(req->sampling_config, ptr, offset);

    // attn_dp_group_id
    req->attn_dp_group_id = DeserializePod<uint32_t>(ptr, offset);

    // kv_comm_group_key
    req->kv_comm_group_key = DeserializeString(ptr, offset);

    // forwarding_tokens
    DeserializeVector<int>(req->forwarding_tokens, ptr, offset);

    // sampling_result_tokens
    DeserializeVector<int>(req->sampling_result_tokens, ptr, offset);

    // infer_stage
    req->infer_stage = DeserializeEnum<InferStage>(ptr, offset);

    // step
    req->step = DeserializePod<int>(ptr, offset);

    // kv_cached_token_num
    req->kv_cached_token_num = DeserializePod<int>(ptr, offset);

    // kv_from_remote (pd_v2). See SerializeInferRequests for rationale.
    req->kv_from_remote = DeserializePod<bool>(ptr, offset);

    // kv_cache_blocks
    DeserializeVector<int>(req->kv_cache_blocks, ptr, offset);

    // swa_kv_cache_blocks
    DeserializeVector<int>(req->swa_kv_cache_blocks, ptr, offset);

    // swa_kv_cache_block_idx_offsets
    DeserializeVector<size_t>(req->swa_kv_cache_block_idx_offsets, ptr, offset);

    // block_token_num
    req->block_token_num = DeserializePod<size_t>(ptr, offset);

    // logits_offset
    req->logits_offset = DeserializePod<size_t>(ptr, offset);

    // prefix_cache_len
    req->prefix_cache_len = DeserializePod<int>(ptr, offset);

    // flexible_cached_copy_tasks, not supported

    // ngram_dict, not supported.

    // timestamp_in_us, not supported.

    // req_ctx, not supported.

    // incremental_decoded_str
    req->incremental_decoded_str = DeserializeString(ptr, offset);

    // input_refit_embedding
    if (req->infer_stage == InferStage::kContext) {
      DeserializeEmbeddingSlice(req->input_refit_embedding, ptr, offset);
    }

    // need_create_generator (deserialize first to decide whether to deserialize config)
    req->need_create_generator = DeserializePod<bool>(ptr, offset);

    // structured_generator_config (only deserialize if need_create_generator is true)
    if (req->need_create_generator) {
      DeserializeStructuredGeneratorConfig(req->structured_generator_config, ptr, offset);
    }

    // pd_v2: Decode-side KV pool routing (mirror of serialize side above).
    {
      const size_t target_count = DeserializePod<size_t>(ptr, offset);
      req->pd_v2_decode_targets.resize(target_count);
      for (size_t i = 0; i < target_count; ++i) {
        req->pd_v2_decode_targets[i].decode_attn_tp_rank_in_group = DeserializePod<uint32_t>(ptr, offset);
        req->pd_v2_decode_targets[i].kv_pool_base_va = DeserializePod<uint64_t>(ptr, offset);
        DeserializeVector<int>(req->pd_v2_decode_targets[i].kv_block_ids, ptr, offset);
      }
      req->pd_v2_decode_engine_inference_addr = DeserializeString(ptr, offset);
      req->pd_v2_decode_attn_dp_group_id = DeserializePod<uint32_t>(ptr, offset);
    }

    infer_reqs.push_back(req);
  }

  return Status();
}

static Status SerializeInferRequestPatches(const std::vector<std::shared_ptr<InferRequestPatch>>& infer_req_patches,
                                           void* ptr, size_t& offset) {
  // vec size
  SerializePod<size_t>(infer_req_patches.size(), ptr, offset);

  for (auto req : infer_req_patches) {
    // req_id
    SerializePod<int64_t>(req->req_id, ptr, offset);

    // delta_forwarding_tokens
    SerializeVector<int>(req->delta_forwarding_tokens, ptr, offset);

    // forwarding_tokens_offset
    SerializePod<size_t>(req->forwarding_tokens_offset, ptr, offset);

    // forwarding_tokens_draft_num
    SerializePod<size_t>(req->forwarding_tokens_draft_num, ptr, offset);

    // kv_cached_token_num
    SerializePod<size_t>(req->kv_cached_token_num, ptr, offset);

    // kv_from_remote (pd_v2)
    SerializePod<bool>(req->kv_from_remote, ptr, offset);

    // prefix_cache_len
    SerializePod<int>(req->prefix_cache_len, ptr, offset);

    // delta_kv_cache_blocks
    SerializeVector<int>(req->delta_kv_cache_blocks, ptr, offset);

    // kv_cache_blocks_offset
    SerializePod<size_t>(req->kv_cache_blocks_offset, ptr, offset);

    // swa_kv_cache_blocks (full list, not incremental)
    SerializeVector<int>(req->swa_kv_cache_blocks, ptr, offset);

    // swa_kv_cache_block_idx_offsets
    SerializeVector<size_t>(req->swa_kv_cache_block_idx_offsets, ptr, offset);

    // infer_stage
    SerializeEnum<InferStage>(req->infer_stage, ptr, offset);

    // step
    SerializePod<int>(req->step, ptr, offset);

    // sampling_token_num
    SerializePod<size_t>(req->sampling_token_num, ptr, offset);
  }

  return Status();
}

static Status DeserializeInferRequestPatches(std::vector<std::shared_ptr<InferRequestPatch>>& infer_req_patches,
                                             const void* ptr, size_t& offset) {
  // vec size
  size_t vec_size = DeserializePod<size_t>(ptr, offset);

  for (size_t i = 0; i < vec_size; ++i) {
    std::shared_ptr<InferRequestPatch> req_patch = std::make_shared<InferRequestPatch>();

    // req_id
    req_patch->req_id = DeserializePod<int64_t>(ptr, offset);

    // delta_forwarding_tokens
    DeserializeVector<int>(req_patch->delta_forwarding_tokens, ptr, offset);

    // forwarding_tokens_offset
    req_patch->forwarding_tokens_offset = DeserializePod<size_t>(ptr, offset);

    // forwarding_tokens_draft_num
    req_patch->forwarding_tokens_draft_num = DeserializePod<size_t>(ptr, offset);

    // kv_cached_token_num
    req_patch->kv_cached_token_num = DeserializePod<size_t>(ptr, offset);

    // kv_from_remote (pd_v2)
    req_patch->kv_from_remote = DeserializePod<bool>(ptr, offset);

    // prefix_cache_len
    req_patch->prefix_cache_len = DeserializePod<int>(ptr, offset);

    // delta_kv_cache_blocks
    DeserializeVector<int>(req_patch->delta_kv_cache_blocks, ptr, offset);

    // kv_cache_blocks_offset
    req_patch->kv_cache_blocks_offset = DeserializePod<size_t>(ptr, offset);

    // swa_kv_cache_blocks (full list, not incremental)
    DeserializeVector<int>(req_patch->swa_kv_cache_blocks, ptr, offset);

    // swa_kv_cache_block_idx_offsets
    DeserializeVector<size_t>(req_patch->swa_kv_cache_block_idx_offsets, ptr, offset);

    // infer_stage
    req_patch->infer_stage = DeserializeEnum<InferStage>(ptr, offset);

    // step
    req_patch->step = DeserializePod<int>(ptr, offset);

    // sampling_token_num
    req_patch->sampling_token_num = DeserializePod<size_t>(ptr, offset);

    infer_req_patches.push_back(req_patch);
  }

  return Status();
}

size_t ScheduleOutputParser::GetSerializedSize(const ScheduleOutput* schedule_output) {
  // Use a large enough memory temporarily, change to really size layer.
  return 64 * 1024 * 1024;
}

void ScheduleOutputParser::SetScheduleProcessor(const std::shared_ptr<ScheduleProcessorInterface> schedule_processor) {
  schedule_processor_ = schedule_processor;
}

Status ScheduleOutputParser::SerializeScheduleOutput(const ScheduleOutput* schedule_output, void* ptr, size_t* bytes) {
  PROFILE_EVENT_SCOPE(serialize_schedule_output_total, "Engine.SerializeScheduleOutput.Total");
  KLLM_CHECK_WITH_INFO(schedule_processor_ != nullptr,
                       "ScheduleOutputParser::schedule_processor_ is nullptr when serializing. "
                       "Please call ScheduleOutputParser::SetScheduleProcessor(...) before SerializeScheduleOutput.");
  {
    PROFILE_EVENT_SCOPE(serialize_lock_wait, "Engine.SerializeScheduleOutput.LockWait");
    schedule_processor_->Lock();
  }
  // 按 patch_transfer 分流: full transfer 走 InferRequest, 增量走 InferRequestPatch.
  // patch_transfer 与 forwarding_tokens 等字段均优先读 req_serialization_snapshots (见 schedule_output.h).
  std::vector<std::shared_ptr<InferRequest>> full_running_reqs;
  std::vector<std::shared_ptr<InferRequestPatch>> patch_running_reqs;
  {
    PROFILE_EVENT_SCOPE(serialize_under_lock, "Engine.SerializeScheduleOutput.UnderLock");
    for (const auto& req : schedule_output->running_reqs) {
      bool patch_transfer = req->patch_transfer;
      auto snap_it_pt = schedule_output->req_serialization_snapshots.find(req->req_id);
      if (snap_it_pt != schedule_output->req_serialization_snapshots.end() &&
          snap_it_pt->second.patch_transfer_populated) {
        patch_transfer = snap_it_pt->second.patch_transfer;
      }
      if (!patch_transfer) {
        full_running_reqs.push_back(req);
        continue;
      }

      std::shared_ptr<InferRequestPatch> patch_req = std::make_shared<InferRequestPatch>();

      patch_req->step = req->step;
      patch_req->req_id = req->req_id;
      patch_req->infer_stage = req->infer_stage;
      patch_req->prefix_cache_len = req->prefix_cache_len;
      // pd_v2: kv_from_remote is per-request (not snapshot-tracked), so it is set
      // directly here; the remaining fields are populated from the snapshot below.
      patch_req->kv_from_remote = req->pd_v2.kv_from_remote;
      patch_req->forwarding_tokens_offset = 0;
      patch_req->kv_cache_blocks_offset = 0;

      auto snap_it = schedule_output->req_serialization_snapshots.find(req->req_id);
      if (snap_it != schedule_output->req_serialization_snapshots.end()) {
        const auto& snap = snap_it->second;
        patch_req->kv_cached_token_num = snap.kv_cached_token_num;
        patch_req->forwarding_tokens_draft_num = snap.forwarding_tokens_draft_num;
        patch_req->sampling_token_num = snap.sampling_token_num;
        patch_req->delta_forwarding_tokens = snap.forwarding_tokens;
        patch_req->delta_kv_cache_blocks = snap.kv_cache_blocks;
        patch_req->swa_kv_cache_blocks = snap.swa_kv_cache_blocks;
        patch_req->swa_kv_cache_block_idx_offsets = snap.swa_kv_cache_block_idx_offsets;
        patch_req->prefix_cache_len = snap.prefix_cache_len;
        patch_req->step = snap.step;
      } else {
        patch_req->kv_cached_token_num = req->kv_cached_token_num;
        patch_req->forwarding_tokens_draft_num = req->forwarding_tokens_draft_num;
        patch_req->sampling_token_num = req->sampling_token_num;
        // TODO(yancyliu): transfer full forwarding_tokens & kv_cache_blocks now, change to incremental later.
        patch_req->delta_forwarding_tokens = req->forwarding_tokens;
        patch_req->delta_kv_cache_blocks = req->kv_cache_blocks;
        patch_req->swa_kv_cache_blocks = req->swa_kv_cache_blocks;
        patch_req->swa_kv_cache_block_idx_offsets = req->swa_kv_cache_block_idx_offsets;
      }

      patch_running_reqs.push_back(patch_req);
    }

    SerializeInit(*bytes);

    // schedule_id
    SerializePod<size_t>(schedule_output->schedule_id, ptr, *bytes);

    // enable_profile_metrics + schedule_dispatch_time_ns
    SerializePod<bool>(schedule_output->enable_profile_metrics, ptr, *bytes);
    SerializePod<int64_t>(schedule_output->schedule_dispatch_time_ns, ptr, *bytes);

    // finish_req_ids
    SerializeVectorOfVector(schedule_output->finish_req_ids, ptr, *bytes);

    // recompute_req_ids（必须在 finish_req_ids 之后，反序列化顺序需保持一致）
    SerializeVectorOfVector<int64_t>(schedule_output->recompute_req_ids, ptr, *bytes);

    // running reqs or executor running reqs.
    // NOTE: Use lock & unlock here, because running_reqs maybe modified in scheduler thread.
    // And this will cause wrong serialized bytes.
    {
      PROFILE_EVENT_SCOPE(serialize_full_reqs, "Engine.SerializeScheduleOutput.FullReqs");
      SerializeInferRequests(full_running_reqs, &schedule_output->req_serialization_snapshots, ptr, *bytes);
    }
    {
      PROFILE_EVENT_SCOPE(serialize_patch_reqs, "Engine.SerializeScheduleOutput.PatchReqs");
      SerializeInferRequestPatches(patch_running_reqs, ptr, *bytes);
    }
  }
  schedule_processor_->Unlock();

  // NOTE: structured_generator_configs and need_create_generator_req_ids are now
  // embedded in each ExecutorInferRequest, so we don't need to serialize them separately

  return Status();
}

Status ScheduleOutputParser::DeserializeScheduleOutput(const void* ptr, ScheduleOutput* schedule_output) {
  PROFILE_EVENT_SCOPE(deserialize_schedule_output_total, "Executor.DeserializeScheduleOutput.Total");
  size_t offset = 0;

  // schedule_id
  schedule_output->schedule_id = DeserializePod<size_t>(ptr, offset);

  // enable_profile_metrics + schedule_dispatch_time_ns
  schedule_output->enable_profile_metrics = DeserializePod<bool>(ptr, offset);
  schedule_output->schedule_dispatch_time_ns = DeserializePod<int64_t>(ptr, offset);

  // finish_req_ids
  DeserializeVectorOfVector(schedule_output->finish_req_ids, ptr, offset);

  // recompute_req_ids（与序列化顺序保持一致）
  DeserializeVectorOfVector<int64_t>(schedule_output->recompute_req_ids, ptr, offset);

  // running reqs.
  {
    PROFILE_EVENT_SCOPE(deserialize_full_reqs, "Executor.DeserializeScheduleOutput.FullReqs");
    DeserializeInferRequests(schedule_output->executor_running_reqs, ptr, offset);
  }
  {
    PROFILE_EVENT_SCOPE(deserialize_patch_reqs, "Executor.DeserializeScheduleOutput.PatchReqs");
    DeserializeInferRequestPatches(schedule_output->running_req_patches, ptr, offset);
  }

  // NOTE: structured_generator_configs and need_create_generator_req_ids are now
  // embedded in each ExecutorInferRequest, so we extract them from the requests
  schedule_output->structured_generator_configs.clear();
  schedule_output->need_create_generator_req_ids.clear();
  for (const auto& req : schedule_output->executor_running_reqs) {
    if (req->structured_generator_config.HasConstraint()) {
      schedule_output->structured_generator_configs[req->req_id] = req->structured_generator_config;
      if (req->need_create_generator) {
        schedule_output->need_create_generator_req_ids.insert(req->req_id);
      }
    }
  }

  return Status();
}

ScheduleOutput* ScheduleOutputPool::GetFreeScheduleOutput() {
  if (free_queue_.Empty()) {
    ScheduleOutput* schedule_output = new ScheduleOutput();
    return schedule_output;
  }

  return free_queue_.Get();
}

Status ScheduleOutputPool::FreeScheduleOutput(ScheduleOutput* schedule_output) {
  schedule_output->Clear();
  free_queue_.Put(schedule_output);

  return Status();
}

void ScheduleOutputPool::PutSendScheduleOutput(ScheduleOutput* schedule_output) {
  size_t schedule_output_size = ScheduleOutputParser::GetSerializedSize(schedule_output);

  Packet* packet = GetRawPacket(schedule_output_size);
  if (packet == nullptr) {
    throw std::runtime_error("ScheduleOutputPool::PutToSendQueue allocate memory error.");
  }

  size_t serialized_bytes;
  packet->type = PacketType::CONTROL_REQ_SCHEDULE;
  ScheduleOutputParser::SerializeScheduleOutput(schedule_output, packet->body, &serialized_bytes);
  packet->size = serialized_bytes;

  serialized_send_queue_.Put(packet);
}

Packet* ScheduleOutputPool::GetSendSerializedPacket() { return serialized_send_queue_.Get(); }

void ScheduleOutputPool::PutRecvSerializedPacket(Packet* packet) { serialized_recv_queue_.Put(packet); }

Packet* ScheduleOutputPool::GetRecvSerializedPacket() { return serialized_recv_queue_.Get(); }

Status ScheduleOutputPool::Stop() {
  free_queue_.Stop();

  serialized_send_queue_.Stop();
  serialized_recv_queue_.Stop();

  return Status();
}

}  // namespace ksana_llm
