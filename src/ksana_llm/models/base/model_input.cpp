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
  Status status = GetDeviceMemoryInfo(MemoryDevice::MEMORY_DEVICE, &device_free, &device_total);
  if (status.OK()) {
    size_t reserved_memory_size = device_total * block_manager_config.reserved_device_memory_ratio;
    max_block_num = std::min(max_block_num, (device_free - reserved_memory_size) / GetBlockManager()->GetBlockSize());
  }
  KLLM_LOG_INFO << "max_block_num " << max_block_num;

  // For prefix caching, the token will be used multiple times, reset it to max possible value.
  if (Singleton<Environment>::GetInstance()->IsPrefixCachingEnabled()) {
    max_block_num = (max_token_num_ * max_batch_size_) / GetBlockManager()->GetBlockTokenNum();
  }

  STATUS_CHECK_FAILURE(CreateTensor(kv_cache_offset_tensor, {max_batch_size_ + 1}, TYPE_INT32, rank_, MEMORY_DEVICE));
  STATUS_CHECK_FAILURE(CreateTensor(input_ids, {max_token_num_}, TYPE_INT32, rank_, MEMORY_DEVICE));
  STATUS_CHECK_FAILURE(
      CreateTensor(kv_list, {static_cast<uint64_t>(num_layer_), max_block_num, 2}, TYPE_POINTER, rank_, MEMORY_DEVICE));

  STATUS_CHECK_FAILURE(
      CreateTensor(kv_cache_buffer,
                   {static_cast<uint64_t>(max_batch_size_), static_cast<uint64_t>((max_seq_len_ + 511) / 512),
                    static_cast<uint64_t>(head_num_per_tp), static_cast<uint64_t>(size_per_head) + 2},
                   TYPE_FP32, rank_, MEMORY_DEVICE));

  STATUS_CHECK_FAILURE(
      CreateTensor(input_offset_uint64_tensor, {max_batch_size_ + 1}, TYPE_UINT64, rank_, MEMORY_DEVICE));

  STATUS_CHECK_FAILURE(
      CreateTensor(logits_custom_length_uint64_tensor, {max_batch_size_ + 1}, TYPE_UINT64, rank_, MEMORY_DEVICE));

  STATUS_CHECK_FAILURE(
      CreateTensor(input_tokens_int32_tensor, {max_batch_size_ + 1}, TYPE_INT32, rank_, MEMORY_DEVICE));
  STATUS_CHECK_FAILURE(CreateTensor(rotary_embedding_pos, {max_token_num_}, TYPE_INT64, rank_, MEMORY_DEVICE));
  STATUS_CHECK_FAILURE(CreateTensor(rotary_embedding_mask, {max_token_num_}, TYPE_INT64, rank_, MEMORY_DEVICE));
  STATUS_CHECK_FAILURE(
      CreateTensor(input_prefix_uint64_tensor, {max_batch_size_ + 1}, TYPE_UINT64, rank_, MEMORY_DEVICE));
  STATUS_CHECK_FAILURE(
      CreateTensor(logits_length_prefix_uint64_tensor, {max_batch_size_ + 1}, TYPE_UINT64, rank_, MEMORY_DEVICE));

  STATUS_CHECK_FAILURE(CreateTensor(cpu_input_refit_tensor.pos_pair_tensor, {input_ids.shape[0], 2}, TYPE_INT64, rank_,
                                    MemoryDevice::MEMORY_HOST));
  STATUS_CHECK_FAILURE(CreateTensor(cpu_input_refit_tensor.emb_fp32_ptr_tensor, input_ids.shape, TYPE_POINTER, rank_,
                                    MemoryDevice::MEMORY_HOST));

  EventCreateWithFlags(&kvcache_offset_event, EVENT_DISABLE_TIMING);
  EventCreateWithFlags(&rotary_embedding_event, EVENT_DISABLE_TIMING);
  EventCreateWithFlags(&input_ids_event, EVENT_DISABLE_TIMING);
#ifdef ENABLE_ACL_ATB
  // NOTE(karlluo): for ATb, all device blocks locate on a flatten plane memory space.
  // The Ksana kv cache consists of blocks, each of which is an independent storage space. The blocks are not
  // guaranteed to be contiguous in memory. Each block has a shape of [2, layer_num, block_token_num, head_num,
  // head_dim], where 2 represents key and value. The Ascend ATB kv cache consists of kcache and vcache, which are
  // independent contiguous storage spaces. The shapes of kcache and vcache are [num_blocks * layer_num,
  // block_token_num, head_num, head_dim]. Each block has a size of [block_token_num, head_num, head_dim]. To
  // interface with the NPU, Ascend ATB (hereinafter referred to as ATB) needs to be used. In order for the NPU's
  // self/paged attention to utilize Ksana's kv cache and share the underlying memory/GPU memory management
  // capabilities, the Ksana kv cache needs to be converted to the Ascend ATB kv cache format.
  // 1. Change the block allocation method so that the blocks are contiguous in physical memory, while the upper-level
  // pointers point to different storage spaces. Originally, each block in the Ksana kv cache called malloc once. This
  // should be changed to pre-allocate a contiguous storage space of size [num_blocks, 2, layer_num, block_token_num,
  // head_num, head_dim]. The pointers of each block should then point to cache_base_ptr + (block index * 2 *
  // layer_num * block_token_num * head_num * head_dim * sizeof(DTYPE)).
  // 2. During each inference process, each prompt will carry an array of block IDs, which can be used to obtain the
  // pointers to the storage space. For ATB, conversion is required to use these pointers. The conversion process is
  // as follows:
  //    - Given a block ID array [b0, b1, b2, b3, b4] and the base address pointer of the Ksana kv cache after the
  //    modification in step 1, cache_base_ptr.
  //    - For ATB: The Ksana kv cache has a total of num_blocks * 2 * layer_num blocks.
  //    - Therefore, the block ID array for ATB is [b0 * layer_num * 2, b1 * layer_num * 2, b2 * layer_num * 2, b3 *
  //    layer_num * 2, b4 * layer_num * 2].
  //    - Ksana's kv cache swaps memory/GPU memory at the block level, so to reuse Ksana's kv cache's underlying
  //    memory/GPU memory management capabilities, ATB's kcache and vcache share the same Ksana kv cache.
  //    - Since each block in Ksana is divided into K and V parts, each part having a size of [layer_num,
  //    block_token_num, head_num, head_dim].
  //    - To allow ATB's kcache and vcache to share the same block ID array, the kcache pointer is cache_base_ptr, and
  //    the vcache pointer is cache_base_ptr + (layer_num * block_token_num * head_num * head_dim * sizeof(DTYPE)).
  //    - Therefore, the block ID array for kcache/vcache is [b0 * layer_num * 2 + layer_idx, b1 * layer_num * 2 +
  //    layer_idx, b2 * layer_num * 2 + layer_idx, b3 * layer_num * 2 + layer_idx, b4 * layer_num * 2 + layer_idx].
  STATUS_CHECK_FAILURE(
      CreateTensor(seq_len_host, {static_cast<uint64_t>(max_batch_size_)}, TYPE_INT32, rank_, MEMORY_HOST));
  STATUS_CHECK_FAILURE(CreateTensor(
      layers_slot_mapping, {static_cast<uint64_t>(model_config.num_layer), static_cast<uint64_t>(max_token_num_)},
      TYPE_INT32, rank_, MEMORY_DEVICE));
  STATUS_CHECK_FAILURE(CreateTensor(
      layers_block_table,
      {static_cast<uint64_t>(model_config.num_layer), static_cast<uint64_t>(max_batch_size_ * max_block_num)},
      TYPE_INT32, rank_, MEMORY_DEVICE));
  GetBlockManager()->SetDeviceId(rank);
  void* cur_rank_block_base_ptr = GetBlockManager()->GetBlockBasePtr();
  void* k_cache_base_ptr = cur_rank_block_base_ptr;
  void* v_cache_base_ptr = cur_rank_block_base_ptr + (block_size_ / 2);
  // https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/acce/ascendtb/ascendtb_01_0070.html
  // k/v_cache_blocks_base only support float16
  STATUS_CHECK_FAILURE(CreateTensor(
      k_cache_blocks_base,
      {GetBlockManager()->GetBlockManagerConfig().device_allocator_config.blocks_num * 2 * model_config.num_layer,
       model_config.block_token_num, model_config.head_num, model_config.size_per_head},
      TYPE_FP16, rank, MEMORY_DEVICE, k_cache_base_ptr));
  STATUS_CHECK_FAILURE(CreateTensor(
      v_cache_blocks_base,
      {GetBlockManager()->GetBlockManagerConfig().device_allocator_config.blocks_num * 2 * model_config.num_layer,
       model_config.block_token_num, model_config.head_num, model_config.size_per_head},
      TYPE_FP16, rank, MEMORY_DEVICE, v_cache_base_ptr));
  // 0: layers_slot_mapping_dim_1, 1: max_num_blocks_per_query
  STATUS_CHECK_FAILURE(CreateTensor(atb_attention_attr, {2}, TYPE_UINT64, rank, MEMORY_HOST));
#endif
}

ModelInput::~ModelInput() {
  STATUS_CHECK_FAILURE(DestroyTensor(input_ids, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(input_offset_uint64_tensor, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(logits_custom_length_uint64_tensor, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(cpu_input_refit_tensor.pos_pair_tensor, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(cpu_input_refit_tensor.emb_fp32_ptr_tensor, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(input_tokens_int32_tensor, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(input_prefix_uint64_tensor, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(logits_length_prefix_uint64_tensor, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(rotary_embedding_pos, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(rotary_embedding_mask, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(kv_cache_buffer, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(kv_cache_offset_tensor, rank_));
  STATUS_CHECK_FAILURE(DestroyTensor(kv_list, rank_));

  EventDestroy(kvcache_offset_event);
  EventDestroy(rotary_embedding_event);
  EventDestroy(input_ids_event);
}

void ModelInput::ParseFromRequests(const std::vector<ForwardRequest>& forward_reqs, bool is_context_stage) {
  // NOTE(karlluo): check batch size
  batch_size = forward_reqs.size();
  KLLM_LOG_DEBUG << (is_context_stage ? "ContextDecode" : "Decode") << " With Batch Size " << batch_size;
  if (batch_size == 0) {
    std::invalid_argument(fmt::format("ModelInput empty forward requests."));
  } else if (batch_size > (size_t)model_config_.max_batch_size) {
    std::invalid_argument(
        fmt::format("ModelInput bs exceed max bs. {} > {}", batch_size, model_config_.max_batch_size));
  }

  max_tokens = 0;
  total_seq_len = 0;
  total_prefix_len = 0;
  total_block_num = 0;
  kv_cache_offset_list = {0};
  for (size_t idx = 0; idx < batch_size; ++idx) {
    total_seq_len += forward_reqs[idx].output_tokens->size();
    total_prefix_len += forward_reqs[idx].prefix_cache_len;
    total_block_num += forward_reqs[idx].kv_cache_ptrs[rank_].size();
    kv_cache_offset_list.push_back(total_block_num);
  }

  kv_cache_offset_tensor.shape = {kv_cache_offset_list.size()};
  MemcpyAsync(kv_cache_offset_tensor.GetPtr<void>(), kv_cache_offset_list.data(),
              kv_cache_offset_list.size() * sizeof(int), MEMCPY_HOST_TO_DEVICE, context_->GetD2HStreams()[rank_]);
  KLLM_LOG_DEBUG << " Total Block Num " << total_block_num;

#ifdef ENABLE_ACL_ATB
  PrepareATBKVCache(forward_reqs, is_context_stage);
#endif

  if (is_context_stage) {
    PrepareKVCacheBlocks(forward_reqs);
    PreparePrefillPositionIds(forward_reqs);
    PreparePrefillInputIds(forward_reqs);
    PrepareInputRefit(forward_reqs);
  } else {
    PrepareKVCacheBlocks(forward_reqs);
    PrepareDecodePositionIds(forward_reqs);
    PrepareDecodeInputIds(forward_reqs);
  }
}

void ModelInput::PrepareKVCacheBlocks(const std::vector<ForwardRequest>& forward_reqs) {
  kv_list.shape = {model_config_.num_layer, total_block_num * 2};
  cpu_kv_list.resize(model_config_.num_layer * total_block_num * 2);
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
#ifdef ENABLE_ACL
  StreamSynchronize(context_->GetD2HStreams()[rank_]);
#endif
  EventRecord(kvcache_offset_event, context_->GetD2HStreams()[rank_]);
}

void ModelInput::PreparePrefillPositionIds(const std::vector<ForwardRequest>& forward_reqs) {
  std::vector<int64_t> cpu_rotary_pos(total_seq_len);
  std::vector<int64_t> cpu_rotary_mask(total_seq_len, 1);
  int cpu_rotary_pos_idx = 0;
  for (size_t idx = 0; idx < batch_size; ++idx) {
    if (forward_reqs[idx].prefix_cache_len > 0) {
      std::fill(cpu_rotary_mask.begin() + cpu_rotary_pos_idx,
                cpu_rotary_mask.begin() + cpu_rotary_pos_idx + forward_reqs[idx].prefix_cache_len, 0);
    }
    for (size_t pos = 0; pos < forward_reqs[idx].output_tokens->size(); ++pos) {
      cpu_rotary_pos[cpu_rotary_pos_idx++] = pos;
    }
  }
  MemcpyAsync(rotary_embedding_pos.GetPtr<void>(), cpu_rotary_pos.data(), sizeof(int64_t) * total_seq_len,
              MEMCPY_HOST_TO_DEVICE, context_->GetD2HStreams()[rank_]);
#ifdef ENABLE_ACL
  StreamSynchronize(context_->GetD2HStreams()[rank_]);
#endif
  MemcpyAsync(rotary_embedding_mask.GetPtr<void>(), cpu_rotary_mask.data(), sizeof(int64_t) * total_seq_len,
              MEMCPY_HOST_TO_DEVICE, context_->GetD2HStreams()[rank_]);
  EventRecord(rotary_embedding_event, context_->GetD2HStreams()[rank_]);
}

void ModelInput::PrepareInputRefit(const std::vector<ForwardRequest>& forward_reqs) {
  size_t pos = 0;
  size_t cpu_input_refit_pos_pair_idx = 0;
  // Get pointers to the CPU input_refit position pair and CPU input_refit embedding float32 tensors
  int64_t* cpu_input_refit_pos_pair = reinterpret_cast<int64_t*>(cpu_input_refit_tensor.pos_pair_tensor.GetPtr<void>());
  void** cpu_input_refit_emb_fp32_ptr =
      reinterpret_cast<void**>(cpu_input_refit_tensor.emb_fp32_ptr_tensor.GetPtr<void>());

  for (size_t bs_idx = 0; bs_idx < batch_size; ++bs_idx) {
    const ForwardRequest& forward_req = forward_reqs[bs_idx];
    std::vector<int>& input_refit_pos = (*forward_req.input_refit_embedding).pos;
    std::vector<std::vector<float>>& input_refit_embedding = (*forward_req.input_refit_embedding).embeddings;
    // Iterate over the input_refit positions and embeddings
    for (size_t input_refit_idx = 0;
         input_refit_idx < input_refit_pos.size() && input_refit_idx < input_refit_embedding.size();
         input_refit_idx++) {
      cpu_input_refit_emb_fp32_ptr[cpu_input_refit_pos_pair_idx >> 1] = input_refit_embedding[input_refit_idx].data();
      cpu_input_refit_pos_pair[cpu_input_refit_pos_pair_idx++] = input_refit_pos[input_refit_idx] + pos;
      cpu_input_refit_pos_pair[cpu_input_refit_pos_pair_idx++] = input_refit_embedding[input_refit_idx].size();
    }
    pos += forward_req.output_tokens->size();
  }
  cpu_input_refit_tensor.emb_fp32_ptr_tensor.shape = {cpu_input_refit_pos_pair_idx / 2};
  cpu_input_refit_tensor.pos_pair_tensor.shape = {cpu_input_refit_pos_pair_idx / 2, 2};
}

#ifdef ENABLE_ACL_ATB
void ModelInput::PrepareATBKVCache(const std::vector<ForwardRequest>& forward_reqs, bool is_context_stage) {
  // NOTE(karlluo): block manager will change the block number in
  // ResetPreAllocatedBlocks, block_managr's allocator's blocks_num is difference from the allocator's member config, so
  // we need get it from allocator instance.
  size_t total_block_num = GetBlockManager()->GetAllocatorConfig().blocks_num * 2 * model_config_.num_layer;
  if (total_block_num != k_cache_blocks_base.shape[0]) {
    k_cache_blocks_base.shape[0] = total_block_num;
    v_cache_blocks_base.shape[0] = total_block_num;
  }

  uint32_t batch_size = forward_reqs.size();
  layers_slot_mapping_host.clear();
  layers_block_table_host.clear();
  size_t max_num_blocks_per_query = 0;
  // for prefill stage: slot_mapping shape is [num_layers, all_reqs_tokens]
  // for decode stage: slot_mapping shape is [num_layers, batch_size]
  size_t slot_mapping_dim_1 = is_context_stage ? 0ul : batch_size;
  for (size_t f_req_idx = 0; f_req_idx < batch_size; ++f_req_idx) {
    seq_len_host.GetPtr<int32_t>()[f_req_idx] = forward_reqs[f_req_idx].output_tokens->size();
    if (is_context_stage) {
      slot_mapping_dim_1 += forward_reqs[f_req_idx].output_tokens->size();
    } else {
      max_num_blocks_per_query =
          std::max(max_num_blocks_per_query, forward_reqs[f_req_idx].atb_kv_cache_base_blk_ids[rank_].size());
    }
  }

  layers_slot_mapping_host.resize(model_config_.num_layer * slot_mapping_dim_1, 0);
  if (is_context_stage) {
    size_t layers_slot_mapping_offset = 0;
    for (size_t f_req_idx = 0; f_req_idx < batch_size; ++f_req_idx) {
      for (size_t layer_idx = 0; layer_idx < model_config_.num_layer; ++layer_idx) {
        for (size_t token_idx = 0; token_idx < forward_reqs[f_req_idx].output_tokens->size(); ++token_idx) {
          int32_t inner_block_offset = token_idx % model_config_.block_token_num;
          layers_slot_mapping_host[layer_idx * slot_mapping_dim_1 + layers_slot_mapping_offset + token_idx] =
              (forward_reqs[f_req_idx].atb_kv_cache_base_blk_ids[rank_][token_idx / model_config_.block_token_num] +
               layer_idx) *
                  model_config_.block_token_num +
              inner_block_offset;
        }
      }
      layers_slot_mapping_offset += forward_reqs[f_req_idx].output_tokens->size();
    }
  } else {
    layers_block_table_host.resize(model_config_.num_layer * batch_size * max_num_blocks_per_query, -1);
    for (size_t f_req_idx = 0; f_req_idx < batch_size; ++f_req_idx) {
      size_t cur_query_blocks_num = forward_reqs[f_req_idx].atb_kv_cache_base_blk_ids[rank_].size();
      for (size_t layer_idx = 0; layer_idx < model_config_.num_layer; ++layer_idx) {
        for (uint32_t base_block_idx = 0; base_block_idx < cur_query_blocks_num; ++base_block_idx) {
          layers_block_table_host[layer_idx * batch_size * max_num_blocks_per_query +
                                  f_req_idx * max_num_blocks_per_query + base_block_idx] =
              forward_reqs[f_req_idx].atb_kv_cache_base_blk_ids[rank_][base_block_idx] + layer_idx;
        }
      }
      for (size_t layer_idx = 0; layer_idx < model_config_.num_layer; ++layer_idx) {
        int32_t inner_block_offset = seq_len_host.GetPtr<int32_t>()[f_req_idx];
        int32_t base_block_idx = forward_reqs[f_req_idx].atb_kv_cache_base_blk_ids[rank_].size() - 1;
        layers_slot_mapping_host[layer_idx * slot_mapping_dim_1 + f_req_idx] =
            layers_block_table_host[layer_idx * batch_size * max_num_blocks_per_query +
                                    f_req_idx * max_num_blocks_per_query + base_block_idx] *
                model_config_.block_token_num +
            inner_block_offset;
      }
    }
    MemcpyAsync(layers_block_table.GetPtr<void>(), layers_block_table_host.data(),
                layers_block_table_host.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
                context_->GetH2DStreams()[rank_]);
  }
  MemcpyAsync(layers_slot_mapping.GetPtr<void>(), layers_slot_mapping_host.data(),
              layers_slot_mapping_host.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
              context_->GetH2DStreams()[rank_]);

  atb_attention_attr.GetPtr<uint64_t>()[0] = slot_mapping_dim_1;
  atb_attention_attr.GetPtr<uint64_t>()[1] = max_num_blocks_per_query;
}
#endif

void ModelInput::PrepareDecodePositionIds(const std::vector<ForwardRequest>& forward_reqs) {
  std::vector<int64_t> cpu_rotary_pos(batch_size);
  std::vector<int64_t> cpu_rotary_mask(batch_size, 1);
  for (size_t idx = 0; idx < batch_size; ++idx) {
    cpu_rotary_pos[idx] = forward_reqs[idx].output_tokens->size() - 1;
  }
  MemcpyAsync(rotary_embedding_pos.GetPtr<void>(), cpu_rotary_pos.data(), sizeof(int64_t) * batch_size,
              MEMCPY_HOST_TO_DEVICE, context_->GetD2HStreams()[rank_]);
  MemcpyAsync(rotary_embedding_mask.GetPtr<void>(), cpu_rotary_mask.data(), sizeof(int64_t) * batch_size,
              MEMCPY_HOST_TO_DEVICE, context_->GetD2HStreams()[rank_]);
#ifdef ENABLE_ACL
  StreamSynchronize(context_->GetD2HStreams()[rank_]);
#endif
  EventRecord(kvcache_offset_event, context_->GetD2HStreams()[rank_]);
}

void ModelInput::PreparePrefillInputIds(const std::vector<ForwardRequest>& forward_reqs) {
  input_ids.shape = {total_seq_len - total_prefix_len};
  size_t input_offset = 0;
  use_logits_custom_length = false;
  std::vector<int> input_offset_list_int32(batch_size + 1, 0);
  std::vector<size_t> input_offset_list_uint64(batch_size + 1, 0ul);
  std::vector<size_t> logits_custom_length_list_uint64(max_batch_size_ + 1, 0ul);
  std::vector<size_t> logits_length_prefix_list_uint64(max_batch_size_ + 1, 0ul);
  int logits_custom_length_list_uint64_index = 1;
  std::vector<int> input_ids_cpu(0);
  std::vector<int> input_prefix_list_int32(batch_size + 1, 0ul);
  std::vector<size_t> input_prefix_list_uint64(batch_size + 1, 0ul);
  for (size_t idx = 0; idx < batch_size; ++idx) {
    if (forward_reqs[idx].output_tokens->size() < (size_t)forward_reqs[idx].prefix_cache_len) {
      KLLM_LOG_ERROR << fmt::format("Forward Request input tokens {} < prefix cache len {}",
                                    forward_reqs[idx].output_tokens->size(), forward_reqs[idx].prefix_cache_len);
      throw std::runtime_error(fmt::format("Forward Request input tokens {} < prefix cache len {}",
                                           forward_reqs[idx].output_tokens->size(),
                                           forward_reqs[idx].prefix_cache_len));
    }
    std::vector<int>* req_input = forward_reqs[idx].output_tokens;
    size_t prefix_offset = forward_reqs[idx].prefix_cache_len;
    size_t length = req_input->size();
    input_ids_cpu.insert(input_ids_cpu.end(), req_input->begin() + prefix_offset, req_input->end());
    size_t logits_length_prefix_uint64 =
        logits_length_prefix_list_uint64[logits_custom_length_list_uint64_index - 1] + prefix_offset;
    if (forward_reqs[idx].logits_custom_length != 0) {
      for (auto [l, r] : forward_reqs[idx].request_target->at("logits").slice_pos) {
        for (auto i = l; i <= r; i++) {
          logits_custom_length_list_uint64[logits_custom_length_list_uint64_index] = input_offset + i + 1;
          logits_length_prefix_list_uint64[logits_custom_length_list_uint64_index] = logits_length_prefix_uint64;
          logits_custom_length_list_uint64_index++;
        }
      }
      use_logits_custom_length = true;
    }
    input_offset += length;
    input_offset_list_int32[idx + 1] = static_cast<int>(input_offset);
    input_offset_list_uint64[idx + 1] = input_offset;
    input_prefix_list_int32[idx + 1] = input_prefix_list_int32[idx] + forward_reqs[idx].prefix_cache_len;
    input_prefix_list_uint64[idx + 1] = input_prefix_list_uint64[idx] + prefix_offset;
    max_tokens = std::max(max_tokens, length);
  }
  input_offset_list = input_offset_list_int32;
  input_prefix_list = input_prefix_list_int32;
  MemcpyAsync(input_ids.GetPtr<int>(), input_ids_cpu.data(), input_ids_cpu.size() * sizeof(int), MEMCPY_HOST_TO_DEVICE,
              context_->GetH2DStreams()[rank_]);
  MemcpyAsync(input_prefix_uint64_tensor.GetPtr<void>(), input_prefix_list_uint64.data(),
              input_prefix_list_uint64.size() * sizeof(size_t), MEMCPY_HOST_TO_DEVICE,
              context_->GetH2DStreams()[rank_]);
  input_offset_uint64_tensor.shape = {batch_size + 1};
  input_offset_uint64_tensor.dtype = TYPE_UINT64;
  logits_custom_length_uint64_tensor.shape = {(size_t)logits_custom_length_list_uint64_index};
  logits_custom_length_uint64_tensor.dtype = TYPE_UINT64;
  logits_length_prefix_uint64_tensor.shape = logits_custom_length_uint64_tensor.shape;
  logits_length_prefix_uint64_tensor.dtype = logits_custom_length_uint64_tensor.dtype;
  MemcpyAsync(input_offset_uint64_tensor.GetPtr<void>(), input_offset_list_uint64.data(),
              (batch_size + 1) * sizeof(size_t), MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);
  MemcpyAsync(logits_custom_length_uint64_tensor.GetPtr<void>(), logits_custom_length_list_uint64.data(),
              logits_custom_length_list_uint64_index * sizeof(size_t), MEMCPY_HOST_TO_DEVICE,
              context_->GetH2DStreams()[rank_]);
  MemcpyAsync(logits_length_prefix_uint64_tensor.GetPtr<void>(), logits_length_prefix_list_uint64.data(),
              logits_custom_length_list_uint64_index * sizeof(size_t), MEMCPY_HOST_TO_DEVICE,
              context_->GetH2DStreams()[rank_]);
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
  std::vector<size_t> input_prefix_list_uint64(batch_size + 1, 0ul);
  std::fill(input_prefix_list.begin(), input_prefix_list.end(), 0);
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
  input_tokens_int32_tensor.shape = {static_cast<uint64_t>(batch_size)};
  input_offset_uint64_tensor.shape = {static_cast<uint64_t>(batch_size) + 1};
  use_logits_custom_length = false;
  void* input_tokens_int32_ptr = input_tokens_int32_tensor.GetPtr<void>();
  MemcpyAsync(input_tokens_int32_ptr, input_tokens_list_int32.data(), (batch_size) * sizeof(int), MEMCPY_HOST_TO_DEVICE,
              context_->GetH2DStreams()[rank_]);
  MemcpyAsync(input_offset_uint64_tensor.GetPtr<void>(), input_offset_list_uint64.data(),
              (batch_size + 1) * sizeof(size_t), MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);
  MemcpyAsync(input_prefix_uint64_tensor.GetPtr<void>(), input_prefix_list_uint64.data(),
              (batch_size + 1) * sizeof(size_t), MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);
  EventRecord(input_ids_event, context_->GetH2DStreams()[rank_]);

#ifdef ENABLE_ACL
  StreamSynchronize(context_->GetH2DStreams()[rank_]);
#endif
}

}  // namespace ksana_llm
