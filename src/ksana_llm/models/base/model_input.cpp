/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/model_input.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

ModelInput::ModelInput(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context)
    : model_config_(model_config), rank_(rank), context_(context) {
  block_size_ = GetBlockManager()->GetBlockSize();
  max_batch_size_ = model_config_.max_batch_size;
  max_token_num_ = model_config.max_scheduler_token_num;
  num_layer_ = model_config.num_layer;

  int head_num = model_config.head_num;
  int tensor_para_size = model_config.tensor_para_size;
  int head_num_per_tp = head_num / tensor_para_size;
  int size_per_head = model_config.size_per_head;

  int max_seq_len_;
  max_seq_len_ = model_config.max_token_num;
  size_t max_block_num =
      (max_seq_len_ * max_batch_size_ + model_config.block_token_num - 1) / model_config.block_token_num;

  BlockManagerConfig block_manager_config;
  STATUS_CHECK_FAILURE(Singleton<Environment>::GetInstance()->GetBlockManagerConfig(block_manager_config));

  size_t device_total, device_free;
  Status status =
      GetDeviceMemoryInfo(MemoryDevice::MEMORY_DEVICE, &device_free, &device_total);
  if (status.OK()) {
    size_t reserved_memory_size = device_total * block_manager_config.reserved_device_memory_ratio;
    max_block_num = std::min(max_block_num, (device_free - reserved_memory_size) / GetBlockManager()->GetBlockSize());
  }
  NLLM_LOG_INFO << "max_block_num " << max_block_num;

  size_t extra_token_number = 0;
  int prefix_cache_tokens_number =
      block_manager_config.prefix_cache_len > 0 ? block_manager_config.prefix_cache_len : 0;
  if (prefix_cache_tokens_number > 0) {
    extra_token_number = prefix_cache_tokens_number * max_batch_size_;
  }

  STATUS_CHECK_FAILURE(CreateTensor(kv_cache_offset_tensor, {max_batch_size_ + 1}, TYPE_INT32, rank_, MEMORY_DEVICE));
  STATUS_CHECK_FAILURE(
      CreateTensor(input_ids, {max_token_num_ + extra_token_number}, TYPE_INT32, rank_, MEMORY_DEVICE));

  STATUS_CHECK_FAILURE(CreateTensor(kv_list, {static_cast<unsigned long>(num_layer_), max_block_num, 2}, TYPE_POINTER,
                                    rank_, MEMORY_DEVICE));

  STATUS_CHECK_FAILURE(
      CreateTensor(kv_cache_buffer,
                   {static_cast<unsigned long>(max_batch_size_), static_cast<unsigned long>((max_seq_len_ + 511) / 512),
                    static_cast<unsigned long>(head_num_per_tp), static_cast<unsigned long>(size_per_head) + 2},
                    TYPE_FP32, rank_, MEMORY_DEVICE));

  STATUS_CHECK_FAILURE(
      CreateTensor(input_offset_uint64_tensor, {max_batch_size_ + 1}, TYPE_UINT64, rank_, MEMORY_DEVICE));

  STATUS_CHECK_FAILURE(
      CreateTensor(input_tokens_int32_tensor, {max_batch_size_ + 1}, TYPE_INT32, rank_, MEMORY_DEVICE));
  STATUS_CHECK_FAILURE(
      CreateTensor(rotary_embedding_pos, {max_token_num_ + extra_token_number}, TYPE_INT64, rank_, MEMORY_DEVICE));

  EventCreateWithFlags(&kvcache_offset_event, EVENT_DISABLE_TIMING);
  EventCreateWithFlags(&rotary_embedding_event, EVENT_DISABLE_TIMING);
  EventCreateWithFlags(&input_ids_event, EVENT_DISABLE_TIMING);
}

ModelInput::~ModelInput() {
  STATUS_CHECK_FAILURE(DestroyTensor(input_ids, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(input_offset_uint64_tensor, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(input_tokens_int32_tensor, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(rotary_embedding_pos, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(kv_cache_buffer, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(kv_cache_offset_tensor, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(kv_list, rank_));

  EventDestroy(kvcache_offset_event);
  EventDestroy(rotary_embedding_event);
  EventDestroy(input_ids_event);
}

void ModelInput::ParseFromRequests(const std::vector<ForwardRequest>& forward_reqs, bool is_context_stage) {
  batch_size = forward_reqs.size();
  NLLM_LOG_DEBUG << (is_context_stage ? "ContextDecode" : "Decode") << " With Batch Size " << batch_size;
  if (batch_size == 0) {
    std::invalid_argument(fmt::format("ModelInput empty forward requests."));
  } else if (batch_size > (size_t)model_config_.max_batch_size) {
    std::invalid_argument(
        fmt::format("ModelInput bs exceed max bs. {} > {}", batch_size, model_config_.max_batch_size));
  }

  max_tokens = 0;
  total_seq_len = 0;
  total_block_num = 0;
  kv_cache_offset_list = {0};
  for (size_t idx = 0; idx < batch_size; ++idx) {
    total_seq_len += forward_reqs[idx].output_tokens->size();
    total_block_num += forward_reqs[idx].kv_cache_ptrs[rank_].size();
    kv_cache_offset_list.push_back(total_block_num);
  }

  kv_cache_offset_tensor.shape = {kv_cache_offset_list.size()};
  MemcpyAsync(kv_cache_offset_tensor.GetPtr<void>(), kv_cache_offset_list.data(),
              kv_cache_offset_list.size() * sizeof(int), MEMCPY_HOST_TO_DEVICE, context_->GetD2HStreams()[rank_]);
  NLLM_LOG_DEBUG << " Total Block Num " << total_block_num;

  if (is_context_stage) {
    PrepareKVCacheBlocks(forward_reqs);
    PreparePrefillPositionIds(forward_reqs);
    PreparePrefillInputIds(forward_reqs);
  } else {
    PrepareKVCacheBlocks(forward_reqs);
    PrepareDecodePositionIds(forward_reqs);
    PrepareDecodeInputIds(forward_reqs);
  }
}

void ModelInput::PrepareKVCacheBlocks(const std::vector<ForwardRequest>& forward_reqs) {
  kv_list.shape = {model_config_.num_layer, total_block_num * 2};
  std::vector<void*> cpu_kv_list(model_config_.num_layer * total_block_num * 2);
  for (size_t layer_idx = 0; layer_idx < model_config_.num_layer; ++layer_idx) {
    int kv_list_index = 0;
    // 处理k
    for (size_t idx = 0; idx < batch_size; ++idx) {
      size_t block_num = forward_reqs[idx].kv_cache_ptrs[rank_].size();
      for (size_t block_idx = 0; block_idx < block_num; block_idx++) {
        void* kv_cache_ptr = forward_reqs[idx].kv_cache_ptrs[rank_][block_idx];
        kv_cache_ptr += layer_idx * block_size_ / model_config_.num_layer;
        cpu_kv_list[layer_idx * total_block_num * 2 + kv_list_index] = kv_cache_ptr;
        kv_list_index++;
      }
    }
    // 处理v
    for (size_t idx = 0; idx < batch_size; ++idx) {
      size_t block_num = forward_reqs[idx].kv_cache_ptrs[rank_].size();
      for (size_t block_idx = 0; block_idx < block_num; block_idx++) {
        void* kv_cache_ptr = forward_reqs[idx].kv_cache_ptrs[rank_][block_idx];
        kv_cache_ptr += layer_idx * block_size_ / model_config_.num_layer + block_size_ / model_config_.num_layer / 2;
        cpu_kv_list[layer_idx * total_block_num * 2 + kv_list_index] = kv_cache_ptr;
        kv_list_index++;
      }
    }
  }
  MemcpyAsync(kv_list.GetPtr<void>(), cpu_kv_list.data(), cpu_kv_list.size() * sizeof(void*), MEMCPY_HOST_TO_DEVICE,
              context_->GetD2HStreams()[rank_]);
  EventRecord(kvcache_offset_event, context_->GetD2HStreams()[rank_]);
}

void ModelInput::PreparePrefillPositionIds(const std::vector<ForwardRequest>& forward_reqs) {
  std::vector<int64_t> cpu_rotary_pos(total_seq_len);
  int cpu_rotary_pos_idx = 0;
  for (size_t idx = 0; idx < batch_size; ++idx) {
    for (size_t pos = 0; pos < forward_reqs[idx].output_tokens->size(); ++pos) {
      cpu_rotary_pos[cpu_rotary_pos_idx++] = pos;
    }
  }
  MemcpyAsync(rotary_embedding_pos.GetPtr<void>(), cpu_rotary_pos.data(), sizeof(int64_t) * total_seq_len,
              MEMCPY_HOST_TO_DEVICE, context_->GetD2HStreams()[rank_]);
  EventRecord(rotary_embedding_event, context_->GetD2HStreams()[rank_]);
}

void ModelInput::PrepareDecodePositionIds(const std::vector<ForwardRequest>& forward_reqs) {
  std::vector<int64_t> cpu_rotary_pos(batch_size);
  for (size_t idx = 0; idx < batch_size; ++idx) {
    cpu_rotary_pos[idx] = forward_reqs[idx].output_tokens->size() - 1;
  }
  MemcpyAsync(rotary_embedding_pos.GetPtr<void>(), cpu_rotary_pos.data(), sizeof(int64_t) * batch_size,
              MEMCPY_HOST_TO_DEVICE, context_->GetD2HStreams()[rank_]);
  EventRecord(kvcache_offset_event, context_->GetD2HStreams()[rank_]);
}

void ModelInput::PreparePrefillInputIds(const std::vector<ForwardRequest>& forward_reqs) {
  input_ids.shape = {total_seq_len};
  size_t input_offset = 0;
  std::vector<int> input_offset_list_int32(batch_size + 1, 0);
  std::vector<size_t> input_offset_list_uint64(batch_size + 1, 0ul);
  std::vector<int> input_ids_cpu(0);
  for (size_t idx = 0; idx < batch_size; ++idx) {
    std::vector<int>* req_input = forward_reqs[idx].output_tokens;
    size_t length = req_input->size();
    input_ids_cpu.insert(input_ids_cpu.end(), req_input->begin(), req_input->end());
    input_offset += length;
    input_offset_list_int32[idx + 1] = static_cast<int>(input_offset);
    input_offset_list_uint64[idx + 1] = input_offset;
    max_tokens = std::max(max_tokens, length);
  }
  MemcpyAsync(input_ids.GetPtr<int>(), input_ids_cpu.data(), input_ids_cpu.size() * sizeof(int), MEMCPY_HOST_TO_DEVICE,
              context_->GetH2DStreams()[rank_]);

  input_offset_uint64_tensor.shape = {batch_size + 1};
  input_offset_uint64_tensor.dtype = TYPE_UINT64;
  MemcpyAsync(input_offset_uint64_tensor.GetPtr<void>(), input_offset_list_uint64.data(),
              (batch_size + 1) * sizeof(size_t), MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);
  EventRecord(input_ids_event, context_->GetH2DStreams()[rank_]);

#ifdef ENABLE_ACL
  // Event wait between streams seems not work, force sync here.
  StreamSynchronize(context_->GetH2DStreams()[rank_]);
#endif
}

void ModelInput::PrepareDecodeInputIds(const std::vector<ForwardRequest>& forward_reqs) {
  input_ids.shape = {batch_size};
  std::vector<int> input_ids_cpu(batch_size);
  size_t input_offset = 0;
  std::vector<int> input_offset_list_int32(batch_size + 1, 0);
  std::vector<int> input_tokens_list_int32(batch_size, 0);
  std::vector<size_t> input_offset_list_uint64(batch_size + 1, 0ul);
  for (size_t idx = 0; idx < batch_size; ++idx) {
    std::vector<int>* req_input = forward_reqs[idx].output_tokens;
    size_t length = req_input->size();
    input_ids_cpu[idx] = req_input->at(length - 1);
    max_tokens = std::max(max_tokens, length);
    input_offset++;
    input_tokens_list_int32[idx] = length;
    input_offset_list_int32[idx + 1] = static_cast<int>(input_offset);
    input_offset_list_uint64[idx + 1] = input_offset;
  }

  MemcpyAsync(input_ids.GetPtr<void>(), input_ids_cpu.data(), batch_size * sizeof(int), MEMCPY_HOST_TO_DEVICE,
              context_->GetH2DStreams()[rank_]);

  // create input offset tensor int32 and uint64
  input_tokens_int32_tensor.shape = {static_cast<unsigned long>(batch_size)};
  input_offset_uint64_tensor.shape = {static_cast<unsigned long>(batch_size) + 1};
  void* input_tokens_int32_ptr = input_tokens_int32_tensor.GetPtr<void>();
  MemcpyAsync(input_tokens_int32_ptr, input_tokens_list_int32.data(), (batch_size) * sizeof(int), MEMCPY_HOST_TO_DEVICE,
              context_->GetH2DStreams()[rank_]);
  MemcpyAsync(input_offset_uint64_tensor.GetPtr<void>(), input_offset_list_uint64.data(),
              (batch_size + 1) * sizeof(size_t), MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);
  EventRecord(input_ids_event, context_->GetH2DStreams()[rank_]);

#ifdef ENABLE_ACL
  StreamSynchronize(context_->GetH2DStreams()[rank_]);
#endif
}

}  // namespace ksana_llm
