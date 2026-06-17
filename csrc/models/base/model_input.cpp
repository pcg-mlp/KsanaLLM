/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/base/model_input.h"

#include <torch/csrc/autograd/python_variable.h>
#include <torch/torch.h>

#include "cache_manager/block_allocator/block_allocator_interface.h"
#include "configure/environment.h"
#include "device/device_types.h"
#include "models/base/attention_meta/flash_attention_meta_builder.h"
#include "models/base/attention_meta/paged_attention_meta_builder.h"
#include "profiler/profile_event.h"
#include "runtime/fast_path_controller.h"
#include "utils/singleton.h"

namespace ksana_llm {

Stream& ModelInput::GetParseH2DStream() { return context_->device->GetH2DStreamForSlot(parse_slot_); }

ModelInput::ModelInput(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
                       std::shared_ptr<Context> context, std::shared_ptr<BaseModelCacheLayout> block_cache_layout)
    : model_config_(model_config),
      runtime_config_(runtime_config),
      rank_(rank),
      context_(context),
      block_cache_layout_(block_cache_layout) {
  // AttentionMetaCreator is constructed later, after all Config parameters (e.g. max_table_block_num) are known.

  auto env = Singleton<Environment>::GetInstance();
  env->GetConnectorConfigs(connector_config_);
  PipelineConfig pipeline_config;
  env->GetPipelineConfig(pipeline_config);
  enable_blocked_multi_token_forwarding_kv_ =
      runtime_config.attn_backend_config.enable_blocked_multi_token_forwarding_kv;
  use_flashinfer_for_decode_ = runtime_config.attn_backend_config.use_flashinfer_for_decode;

  // Update runtime config, because runtime_config maybe changed.
  env->GetRuntimeConfig(runtime_config_);

  block_size_ = runtime_config_.attn_backend_config.block_size;
  const size_t max_batch_size = runtime_config_.max_batch_size;
  const size_t max_token_num = runtime_config.max_step_token_num;  // max step token num
  layer_num_on_node_ = pipeline_config.upper_layer_idx - pipeline_config.lower_layer_idx + 1;
  KLLM_LOG_INFO << "ModelInput upper_layer_idx:" << pipeline_config.upper_layer_idx
                << ", lower_layer_idx:" << pipeline_config.lower_layer_idx;

  if (pipeline_config.lower_nextn_layer_idx >= static_cast<int>(model_config_.num_layer)) {
    layer_num_on_node_ += pipeline_config.upper_nextn_layer_idx - pipeline_config.lower_nextn_layer_idx + 1;
    KLLM_LOG_INFO << "ModelInput add next n, now layer: " << layer_num_on_node_;
  }

  // Pre-compute kv cache types and per-layer offsets once. The layout is fixed at startup,
  // so subsequent forward steps can reuse the cached results instead of recomputing them.
  auto kv_block_layout = block_cache_layout_->GetBlockCacheLayout("kv-cache");
  kv_cache_types_ = kv_block_layout->GetActiveKvCacheTypes();
  for (const auto& kv_cache_type : kv_cache_types_) {
    auto& layer_offsets = kv_cache_layer_offsets_[kv_cache_type];
    layer_offsets.reserve(layer_num_on_node_);
    for (int32_t layer_idx = 0; layer_idx < layer_num_on_node_; ++layer_idx) {
      layer_offsets.push_back(kv_block_layout->GetKvCacheLayerOffsets(layer_idx, kv_cache_type));
    }
  }

  attn_dp_group_id_ = rank_ / runtime_config.parallel_basic_config.attn_tensor_parallel_size;
  attn_dp_rank_id_ = rank_ % runtime_config.parallel_basic_config.attn_tensor_parallel_size;
  attn_dp_group_size_ = runtime_config_.parallel_basic_config.attn_data_parallel_size;
  attn_dp_group_offsets_.assign(attn_dp_group_size_, 0);
  KLLM_LOG_INFO << "rank:" << rank_ << ", attn_dp_group_id_: " << attn_dp_group_id_
                << ", attn_dp_rank_id_: " << attn_dp_rank_id_ << ", attn_dp_group_size_: " << attn_dp_group_size_;

  const size_t max_seq_len = runtime_config.max_seq_len;  // max seq len for one request

  size_t max_table_block_num = ((max_seq_len + runtime_config.attn_backend_config.block_token_num - 1) /
                                runtime_config.attn_backend_config.block_token_num) *
                               max_batch_size;

  BlockManagerConfig block_manager_config;
  STATUS_CHECK_FAILURE(env->GetBlockManagerConfig(block_manager_config));

  // For prefix caching, the token will be used multiple times, keep the max possible value.
  if (runtime_config_.attn_backend_config.block_size > 0 && !runtime_config.enable_prefix_caching) {
    size_t device_total, device_free;
    Status status = GetDeviceMemoryInfo(MemoryDevice::MEMORY_DEVICE, &device_free, &device_total);
    if (status.OK()) {
      const size_t reserved_memory_size = device_total * block_manager_config.reserved_device_memory_ratio;
      // The max number of blocks that can actually appear in the system.
      const size_t max_block_num =
          (device_free - reserved_memory_size) / runtime_config_.attn_backend_config.block_size;
      max_table_block_num = std::min(max_table_block_num, max_block_num);
    }
  }
  KLLM_LOG_INFO << "max_table_block_num: " << max_table_block_num;

  input_ids = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
  cur_batch_pairs_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {2 * max_batch_size});
  input_offset_uint64_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size + 1});
  dp_input_offset_uint64_host = Tensor(MemoryLocation::LOCATION_HOST, TYPE_UINT64, {max_batch_size + 1});
  dp_input_offset_uint64_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size + 1});
  dp_prefill_q_offset_uint64_host = Tensor(MemoryLocation::LOCATION_HOST, TYPE_UINT64, {max_batch_size + 1});
  dp_prefill_q_offset_uint64_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size + 1});
  input_prefix_uint64_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size + 1});
  dp_input_prefix_uint64_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size + 1});

  BatchSchedulerConfig batch_scheduler_config;
  env->GetBatchSchedulerConfig(batch_scheduler_config);
  const size_t max_logits_tokens = max_batch_size * batch_scheduler_config.max_decode_tokens_per_req;
  logits_idx_uint64_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_logits_tokens});

  nextn_hidden_idx_uint64_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_token_num});

  attention_meta_creator_ = std::make_unique<AttentionMetaCreator>(AttentionMetaCreator::Config{
      model_config_, runtime_config_, layer_num_on_node_, max_table_block_num, max_token_num,
      GetDecodeTokenNumThreshold(), enable_blocked_multi_token_forwarding_kv_, rank_, pipeline_config.lower_layer_idx,
      block_cache_layout_, context_});
  flash_meta = attention_meta_creator_->CreateFlashAttentionMeta();
  shared_attention_meta = attention_meta_creator_->CreateSharedAttentionMeta();

  cpu_input_refit_tensor.pos_tensor = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {input_ids.shape[0]});

  CreateVLTensors();

  // Create optional meta if needed (IndexerMeta/C4IndexerMeta removed; now managed via
  // IndexerSharedAttentionMeta/C4IndexerSharedAttentionMeta inside AttentionMetaCreator).

  EventCreateWithFlags(&kvcache_offset_event, EVENT_DISABLE_TIMING);
  EventCreateWithFlags(&rotary_embedding_event, EVENT_DISABLE_TIMING);
  EventCreateWithFlags(&input_ids_event, EVENT_DISABLE_TIMING);

  AttentionBuilderConfig builder_config{model_config_,
                                        runtime_config_,
                                        enable_blocked_multi_token_forwarding_kv_,
                                        use_flashinfer_for_decode_,
                                        rank_,
                                        layer_num_on_node_,
                                        block_size_,
                                        kv_cache_types_,
                                        kv_cache_layer_offsets_,
                                        attn_dp_group_id_,
                                        attn_dp_rank_id_,
                                        context_,
                                        kvcache_offset_event,
                                        rotary_embedding_event};
  flash_builder_ = attention_meta_creator_->CreateFlashBuilder(builder_config);
  paged_builder_ = attention_meta_creator_->CreatePagedBuilder(builder_config);

#if defined(ENABLE_ACL)
  // NOTE(karlluo): for ATB, all device blocks locate on a flatten plane memory space.
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
  seq_len_host = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT32, {static_cast<uint64_t>(max_batch_size)});
  layers_slot_mapping = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
                               {static_cast<uint64_t>(layer_num_on_node_), static_cast<uint64_t>(max_token_num)});
  layers_block_table = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
                              {static_cast<uint64_t>(layer_num_on_node_), static_cast<uint64_t>(max_table_block_num)});
  // https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/acce/ascendtb/ascendtb_01_0070.html
  // k/v_cache_blocks_base only support float16
  k_cache_blocks_base = Tensor(
      MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
      {1, runtime_config.attn_backend_config.block_token_num, model_config.head_num, model_config.size_per_head});
  v_cache_blocks_base = Tensor(
      MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
      {1, runtime_config.attn_backend_config.block_token_num, model_config.head_num, model_config.size_per_head});
  // 0: layers_slot_mapping_dim_1, 1: max_num_blocks_per_query
  atb_attention_attr = Tensor(MemoryLocation::LOCATION_HOST, TYPE_UINT64, {2});
  last_token_index_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {max_batch_size});
  kv_cache_ptrs_tensor =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_POINTER, {static_cast<uint64_t>(max_table_block_num)});
#endif
}

const AttentionMetaCreator::Config& ModelInput::GetCreatorConfig() const {
  return attention_meta_creator_->GetConfig();
}

void ModelInput::ReplaceAttentionMetaCreator(std::unique_ptr<AttentionMetaCreator> creator) {
  attention_meta_creator_ = std::move(creator);
  flash_meta = attention_meta_creator_->CreateFlashAttentionMeta();
  shared_attention_meta = attention_meta_creator_->CreateSharedAttentionMeta();

  AttentionBuilderConfig builder_config{model_config_,
                                        runtime_config_,
                                        enable_blocked_multi_token_forwarding_kv_,
                                        use_flashinfer_for_decode_,
                                        rank_,
                                        layer_num_on_node_,
                                        block_size_,
                                        kv_cache_types_,
                                        kv_cache_layer_offsets_,
                                        attn_dp_group_id_,
                                        attn_dp_rank_id_,
                                        context_,
                                        kvcache_offset_event,
                                        rotary_embedding_event};
  flash_builder_ = attention_meta_creator_->CreateFlashBuilder(builder_config);
  paged_builder_ = attention_meta_creator_->CreatePagedBuilder(builder_config);
}

ModelInput::~ModelInput() {
  EventDestroy(kvcache_offset_event);
  EventDestroy(rotary_embedding_event);
  EventDestroy(input_ids_event);
}

void ModelInput::CreateVLTensors() {
  if (!model_config_.rope_scaling_factor_config.mrope_section.empty()) {
    dp_mrotary_embedding_pos =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {3, runtime_config_.max_step_token_num});
  }
  if (model_config_.type == "arc_hunyuan_video") {
    dp_xdrotary_embedding_pos =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {4, runtime_config_.max_step_token_num});
  }
  if (model_config_.type == "internlmxcomposer2") {
    im_mask = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {runtime_config_.max_step_token_num});
  }
  // Pre-allocate DeepStack GPU buffers to avoid runtime OOM.
  if (model_config_.is_visual && model_config_.num_deepstack_layers > 0) {
    deepstack_data.num_layers = model_config_.num_deepstack_layers;
    deepstack_data.layer_embeds.reserve(model_config_.num_deepstack_layers);
    DataType embed_dtype = model_config_.weight_data_type;
    for (int i = 0; i < model_config_.num_deepstack_layers; ++i) {
      deepstack_data.layer_embeds.emplace_back(
          MemoryLocation::LOCATION_DEVICE, embed_dtype,
          std::vector<size_t>{runtime_config_.max_step_token_num, static_cast<size_t>(model_config_.hidden_units)});
    }
    KLLM_LOG_INFO << fmt::format("Pre-allocated {} DeepStack GPU buffers, each [{}, {}]",
                                 model_config_.num_deepstack_layers, runtime_config_.max_step_token_num,
                                 model_config_.hidden_units);
  }
}

void ModelInput::PrepareInputInfo(const std::vector<ForwardRequest>& forward_reqs) {
  auto& shared_attention_meta = static_cast<StandardSharedAttentionMeta&>(*this->shared_attention_meta);

  shared_attention_meta.input_length.shape = {0};
  shared_attention_meta.kv_list.shape = {0};
  shared_attention_meta.kv_cache_offset.shape = {0};
  shared_attention_meta.rotary_embedding_pos.shape = {0};
  shared_attention_meta.rotary_embedding_mask.shape = {0};
  shared_attention_meta.block_table.shape = {0};
  if (model_config_.use_dsa) {
    auto& dsa_shared = *shared_attention_meta.GetDSA();
    dsa_shared.expanded_block_table.shape = {0};
    dsa_shared.store_out_loc.shape = {0};
  }
  if (model_config_.use_mla) {
    shared_attention_meta.GetMLA()->num_splits.shape = {0};
  }

  // Delegate request routing to creator; flash_meta / page_metas are replaced each step.
  // Model-specific shared-meta resets (e.g. MQA tensors) are performed inside BuildAttentionMetas.
  auto arrays =
      attention_meta_creator_->BuildAttentionMetas(forward_reqs, attn_dp_group_id_, *this->shared_attention_meta);
  flash_meta = std::move(arrays.flash_meta);
  page_metas = std::move(arrays.page_metas);
  multi_token_request_num = arrays.multi_token_request_num;
  dp_multi_token_request_num = arrays.dp_multi_token_request_num;
  single_token_request_num = arrays.single_token_request_num;
  dp_single_token_request_num = arrays.dp_single_token_request_num;
  dp_batch_size = dp_single_token_request_num + dp_multi_token_request_num;
}

void ModelInput::SetInflightResourceManager(InflightResourceManager* mgr) {
  if (attention_meta_creator_) {
    attention_meta_creator_->WireInflightResources(mgr, shared_attention_meta.get());
  }
}

void ModelInput::ParseFromRequests(const std::vector<ForwardRequest>& forward_reqs, const RunMode run_mode) {
  // NOTE(karlluo): check batch size
  PROFILE_EVENT_SCOPE(StartPrepareReqs, "StartPrepareReqs", rank_);
  batch_size = forward_reqs.size();
  KLLM_CHECK_WITH_INFO(batch_size > 0, "ModelInput empty forward requests, batch_size == 0");
  KLLM_CHECK_WITH_INFO(
      !(connector_config_.group_role == GroupRole::DECODE && batch_size > runtime_config_.max_step_token_num),
      fmt::format("ModelInput batch_size exceed max_step_token_num at PD disaggregation. {} > {}", batch_size,
                  runtime_config_.max_step_token_num));
  KLLM_CHECK_WITH_INFO(
      !(connector_config_.group_role == GroupRole::NONE &&
        batch_size > static_cast<size_t>(runtime_config_.max_batch_size)),
      fmt::format("ModelInput batch_size exceed max_batch_size. {} > {}", batch_size, runtime_config_.max_batch_size));

  infer_stage = forward_reqs.front().infer_stage;  // for NPU

  PrepareInputInfo(forward_reqs);

  dp_context_tokens = 0;
  dp_decode_tokens = 0;
  dp_total_prefix_len = 0;
  total_sampling_token_num_ = 0;
  context_kv_cache_block_num = 0;
  decode_kv_cache_block_num = 0;
  has_request_target = false;
  for (const auto& req : forward_reqs) {
    if (req.attn_dp_group_id == attn_dp_group_id_) {
      if (req.GetType() == ForwardRequestType::kFlash) {
        dp_context_tokens += req.GetInputIdsLength();
        dp_total_prefix_len += req.prefix_cache_len;
        context_kv_cache_block_num += req.kv_cache_ptrs.size();
      } else {  // req.GetType() == ForwardRequestType::kPage
        dp_decode_tokens += req.GetInputIdsLength();
        decode_kv_cache_block_num += req.kv_cache_ptrs.size();
      }
    }
    total_sampling_token_num_ += req.sampling_token_num;
    has_request_target |= (req.request_target != nullptr && !req.request_target->empty());
  }

  KLLM_LOG_DEBUG << fmt::format(
      "run_mode: {}, dp_multi_token_request_num: {}, dp_context_tokens: {}, dp_single_token_request_num: {}, "
      "dp_decode_tokens: {}, dp_total_prefix_len: {}, page_metas.size(): {}",
      (run_mode == RunMode::kMain ? "main" : "next"), dp_multi_token_request_num, dp_context_tokens,
      dp_single_token_request_num, dp_decode_tokens, dp_total_prefix_len, page_metas.size());

  PrepareInputIds(forward_reqs);
  PrepareVLInputRefit(forward_reqs);
  PrepareInputRefit(forward_reqs);

  PrepareVLRequest(forward_reqs);
  PrepareCutoffLayer(forward_reqs);
  PrepareNextNGatherIdx(forward_reqs, run_mode);

  PreparePrefill();
  PrepareDecode();

#ifdef ENABLE_CUDA
  PrepareCudagraphParams(forward_reqs);
#endif

#ifdef ENABLE_ACL
  // NOTE(karlluo): please keep PrepareATBKVCache at the last of prepare process
  PrepareATBKVCache(forward_reqs, multi_token_request_num > 0);
#endif
}

// TODO(ttsybyweng): VL_Model :Prepare moved into each Model Class
void ModelInput::PrepareVLInputRefit(const std::vector<ForwardRequest>& forward_reqs) {
  if (!model_config_.rope_scaling_factor_config.mrope_section.empty()) {
    PrepareMRopePos(forward_reqs);
  }
  if (model_config_.type == "arc_hunyuan_video") {
    PrepareXDRopePos(forward_reqs);
  }
  if (model_config_.type == "qwen3_vl") {
    PrepareDeepstack(forward_reqs);
  }
}

void ModelInput::PrepareCutoffLayer(const std::vector<ForwardRequest>& forward_reqs) {
  if (model_config_.type != "minicpm") {
    return;
  }
  cutoff_layer = 0;
  for (const ForwardRequest& req : forward_reqs) {
    if (req.request_target == nullptr) {
      continue;
    }
    const auto it = req.request_target->find("lm_head");
    if (it != req.request_target->end()) {
      std::vector<int> cutoff_layers = it->second.cutoff_layer;
      if (cutoff_layers.empty()) {
        cutoff_layer = model_config_.num_layer;
        continue;
      }
      auto max_layer = std::max_element(cutoff_layers.begin(), cutoff_layers.end());
      cutoff_layer = std::max((*max_layer), cutoff_layer);
    }
  }
}

void ModelInput::PrepareVLRequest(const std::vector<ForwardRequest>& forward_reqs) {
  PROFILE_EVENT_SCOPE(PrepareVLRequest, "PrepareVLRequest", rank_);
  if (model_config_.type == "internlmxcomposer2") {
    is_mask = false;
    size_t pos_num = cpu_input_refit_tensor.pos_tensor.shape[0];
    if ((multi_token_request_num > 0) && (pos_num > 0)) {
#ifdef ENABLE_CUDA
      DataType weight_data_type_ = model_config_.weight_data_type;
      if (weight_data_type_ == TYPE_FP16) {
        PrepareImgMask<half>(pos_num);
      } else if (weight_data_type_ == TYPE_BF16) {
        PrepareImgMask<bfloat16>(pos_num);
      }
#endif
    }
  }
}

void ModelInput::PrepareNextNGatherIdx(const std::vector<ForwardRequest>& forward_reqs, const RunMode run_mode) {
  PROFILE_EVENT_SCOPE(PrepareNextnGatherIdx, "PrepareNextnGatherIdx", rank_);
  std::unordered_map<size_t, size_t> updated_mtp_req_id_to_pos;
  updated_mtp_req_id_to_pos.reserve(forward_reqs.size());

  std::vector<size_t> mtp_hidden_gather_idx;
  mtp_hidden_gather_idx.reserve(forward_reqs.size());
  size_t total_len = 0;
  for (const auto& req : forward_reqs) {
    const size_t input_ids_len = req.GetInputIdsLength();
    if (run_mode == RunMode::kMain) {
      updated_mtp_req_id_to_pos[req.req_id] = total_len;
    } else {
      updated_mtp_req_id_to_pos[req.req_id] = total_len + input_ids_len - 1;
    }
    total_len += input_ids_len;

    if (run_mode == RunMode::kNextN) {
      const size_t begin_idx = mtp_req_id_to_pos_[req.req_id];
      for (size_t idx = begin_idx; idx < begin_idx + input_ids_len; ++idx) {
        mtp_hidden_gather_idx.emplace_back(idx);
      }
    }
  }

  mtp_req_id_to_pos_.swap(updated_mtp_req_id_to_pos);

  if (run_mode == RunMode::kMain) {
    return;
  }

  nextn_hidden_idx_uint64_tensor.shape = {mtp_hidden_gather_idx.size()};
  MemcpyAsync(nextn_hidden_idx_uint64_tensor.GetPtr<void>(), mtp_hidden_gather_idx.data(),
              mtp_hidden_gather_idx.size() * sizeof(decltype(mtp_hidden_gather_idx)::value_type), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetH2DStream());
  KLLM_LOG_DEBUG << "mtp_hidden_gather_idx: " << mtp_hidden_gather_idx;
}

#ifdef ENABLE_CUDA
void ModelInput::PrepareCudagraphParams(const std::vector<ForwardRequest>& forward_reqs) {
  is_cudagraph_batchsize_matched = false;
  if (multi_token_request_num == 0 &&
      (single_token_request_num == 1 || single_token_request_num == 2 || single_token_request_num == 3)) {
    is_cudagraph_batchsize_matched = true;
  }
}
#endif

/**
 * Process the input refit information for the current batch of requests.
 *
 * Inputs:
 * 1. input_refit_embeddings (`std::vector<std::vector<float>>`) is obtained from the user request and placed on the
 * CPU.
 * 2. input_refit_embedding_tensors (`std::vector<py::object>)` is obtained from the Python plugin, which can be placed
 * on the CPU or GPU (not supported yet).
 *
 * Outputs:
 * 1. input_refit_pos_pair contains pairs of (start refit position offset in this batch, embedding length) for each
 * input refit. e.g., [(emb_pos_offset1, emb_length1), (emb_pos_offset2, emb_length2), ...]
 * 2. input_refit_emb_fp32_ptr contains pointers to all input refit on the CPU. e.g., [emb_ptr1, emb_ptr2, ...]
 *
 * After embedding lookup, the input refit embeddings will be placed to their respective intervals according to the
 * above outputs (by `input_refit_layer`).
 */
void ModelInput::PrepareInputRefit(const std::vector<ForwardRequest>& forward_reqs) {
  size_t pos_offset = 0;
  size_t cpu_input_refit_pos_idx = 0;

  int64_t* cpu_input_refit_pos_pair = cpu_input_refit_tensor.pos_tensor.GetPtr<int64_t>();
  std::vector<Tensor>& emb_tensors = cpu_input_refit_tensor.emb_tensors;
  emb_tensors.clear();

  for (const auto& forward_req : forward_reqs) {
    // Only handle input refit for prefill requests
    if (forward_req.GetType() == ForwardRequestType::kFlash) {
      const std::vector<int>& input_refit_pos = (*forward_req.input_refit_embedding).pos;
      const std::vector<Tensor>& input_refit_embedding_tensors = (*forward_req.input_refit_embedding).embedding_tensors;
      KLLM_CHECK_WITH_INFO(input_refit_pos.size() == input_refit_embedding_tensors.size(),
                           "`input_refit_pos.size()` should be equal to `input_refit_embedding_tensors.size()`.");

      // Iterate over the input_refit positions and embeddings
      for (size_t input_refit_idx = 0; input_refit_idx < input_refit_pos.size(); input_refit_idx++) {
        int64_t input_refit_pos_offset = input_refit_pos[input_refit_idx] + pos_offset;
        cpu_input_refit_pos_pair[cpu_input_refit_pos_idx++] = input_refit_pos_offset;
        emb_tensors.push_back(input_refit_embedding_tensors[input_refit_idx]);
      }
    }
    pos_offset += forward_req.GetInputIdsLength();
  }

  cpu_input_refit_tensor.pos_tensor.shape = {cpu_input_refit_pos_idx};
}

/**
 * The MRope position information (position and offset) of qwen2_vl is computed by the `_get_input_positions` function
 * in the Python plugin and is passed as additional tensors.
 * Before model inference, copy the position tensor (`additional_tensors[0]`) to the corresponding GPU tensor
 * (`dp_mrotary_embedding_pos`), and record the offset value (`additional_tensors[1]`).
 */
void ModelInput::PrepareMRopePos(const std::vector<ForwardRequest>& forward_reqs) {
  constexpr int kMRotaryEmbeddingPosFactor = 3;
  std::vector<int64_t> mrotary_embedding_pos_host;

  for (size_t bs_idx = 0; bs_idx < forward_reqs.size(); ++bs_idx) {
    const std::vector<Tensor>& additional_tensors = (*forward_reqs[bs_idx].input_refit_embedding).additional_tensors;
    if (additional_tensors.empty()) {
      *forward_reqs[bs_idx].mrotary_embedding_pos_offset = 0;
    } else {
      KLLM_CHECK_WITH_INFO(
          additional_tensors.size() >= 2,
          "For visual inputs, additional_tensors must contain at least 2 tensors: position tensor and offset tensor.");
      *forward_reqs[bs_idx].mrotary_embedding_pos_offset = additional_tensors[1].GetPtr<int64_t>()[0];
    }
    // only needs to handle the prefill requests in its dp group
    if (forward_reqs[bs_idx].GetType() != ForwardRequestType::kFlash ||
        forward_reqs[bs_idx].attn_dp_group_id != attn_dp_group_id_) {
      continue;
    }
    if (forward_reqs[bs_idx].kv_cached_token_num == 0) {  // 首token的prefill情况
      if (additional_tensors.empty()) {                   // This is a plain text input.
        int64_t num_tokens = forward_reqs[bs_idx].forwarding_tokens->size();
        mrotary_embedding_pos_host.reserve(mrotary_embedding_pos_host.size() + num_tokens * kMRotaryEmbeddingPosFactor);
        for (int64_t i = 0; i < num_tokens; ++i) {
          for (int t = 0; t < kMRotaryEmbeddingPosFactor; ++t) {
            mrotary_embedding_pos_host.emplace_back(i);
          }
        }
        continue;
      } else {  // This is a input with visual information.
        int64_t tensor_size = additional_tensors[0].GetElementNumber();
        int64_t mrotart_embedding_pos_index = mrotary_embedding_pos_host.size();
        mrotary_embedding_pos_host.resize(mrotary_embedding_pos_host.size() + tensor_size);
        std::memcpy(mrotary_embedding_pos_host.data() + mrotart_embedding_pos_index,
                    additional_tensors[0].GetPtr<void>(), sizeof(int64_t) * tensor_size);
      }
    } else {  // decode情况，对应get_next_input_positions
      const size_t input_len =
          forward_reqs[bs_idx].forwarding_tokens->size() - forward_reqs[bs_idx].kv_cached_token_num;
      const auto pos_offset = *forward_reqs[bs_idx].mrotary_embedding_pos_offset;
      mrotary_embedding_pos_host.reserve(mrotary_embedding_pos_host.size() + kMRotaryEmbeddingPosFactor * input_len);
      for (int i = 0; i < input_len; ++i) {
        for (int t = 0; t < kMRotaryEmbeddingPosFactor; ++t) {
          mrotary_embedding_pos_host.emplace_back(forward_reqs[bs_idx].kv_cached_token_num + pos_offset + i);
        }
      }
    }
  }
  MemcpyAsync(dp_mrotary_embedding_pos.GetPtr<void>(), mrotary_embedding_pos_host.data(),
              sizeof(int64_t) * mrotary_embedding_pos_host.size(), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetD2HStream());

#ifdef ENABLE_ACL
  StreamSynchronize(context_->device->GetD2HStream());
#endif
}

/**
 * Prepare DeepStack injection data for Qwen3-VL.
 *
 * additional_tensors layout from Python plugin:
 *   [2]: visual_segment_starts  (INT32, [S])
 *   [3]: visual_segment_lengths (INT32, [S])
 *   [4]: deepstack_layer_ids    (INT32, [N])
 *   [5..5+N-1]: per-layer visual embeddings (model dtype, [total_visual_tokens, hidden_size])
 */
void ModelInput::PrepareDeepstack(const std::vector<ForwardRequest>& forward_reqs) {
  deepstack_data.Reset();
  if (deepstack_data.layer_embeds.empty()) return;

  size_t batch_offset = 0;
  size_t global_embed_rows = 0;
  size_t hidden_size = model_config_.hidden_units;

  for (const auto& req : forward_reqs) {
    const auto& additional_tensors = (*req.input_refit_embedding).additional_tensors;

    if (req.GetType() == ForwardRequestType::kFlash && additional_tensors.size() >= 5) {
      const int32_t* seg_starts = additional_tensors[2].GetPtr<int32_t>();
      const int32_t* seg_lengths = additional_tensors[3].GetPtr<int32_t>();
      size_t num_segments = additional_tensors[2].GetElementNumber();
      const int32_t* layer_ids = additional_tensors[4].GetPtr<int32_t>();
      size_t num_deepstack_layers = additional_tensors[4].GetElementNumber();
      KLLM_CHECK_WITH_INFO(additional_tensors.size() >= 5 + num_deepstack_layers,
                           "additional_tensors must have at least 5 + num_deepstack_layers elements.");

      // Build layer-to-index mapping on first visual request.
      if (deepstack_data.layer_to_deepstack_idx.empty()) {
        for (size_t i = 0; i < num_deepstack_layers; ++i) {
          deepstack_data.layer_to_deepstack_idx[static_cast<int>(layer_ids[i])] = static_cast<int>(i);
        }
      }

      // Collect segments with global offsets.
      int embed_row_offset = 0;
      for (size_t seg_idx = 0; seg_idx < num_segments; ++seg_idx) {
        deepstack_data.segments.push_back({static_cast<int>(batch_offset) + seg_starts[seg_idx],
                                           static_cast<int>(global_embed_rows) + embed_row_offset,
                                           seg_lengths[seg_idx]});
        embed_row_offset += seg_lengths[seg_idx];
      }

      // Copy per-layer embeds directly to pre-allocated GPU buffers.
      size_t total_visual_tokens = static_cast<size_t>(embed_row_offset);
      for (size_t layer_idx = 0; layer_idx < num_deepstack_layers; ++layer_idx) {
        const Tensor& embed_tensor = additional_tensors[5 + layer_idx];
        KLLM_CHECK(embed_tensor.shape.size() == 2 && embed_tensor.shape[1] == hidden_size);
        Tensor& gpu_tensor = deepstack_data.layer_embeds[layer_idx];
        MemcpyAsync(reinterpret_cast<uint8_t*>(gpu_tensor.GetPtr<void>()) +
                        global_embed_rows * hidden_size * embed_tensor.GetDTypeSize(),
                    embed_tensor.GetPtr<void>(), embed_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                    context_->device->GetD2HStream());
      }
      global_embed_rows += total_visual_tokens;
    }
    batch_offset += req.GetInputIdsLength();
  }

  if (global_embed_rows == 0) return;

  deepstack_data.active = true;
  KLLM_LOG_DEBUG << fmt::format("PrepareDeepstack: active={} num_layers={} segments={} total_rows={} hidden_size={}",
                                deepstack_data.active, deepstack_data.num_layers, deepstack_data.segments.size(),
                                global_embed_rows, hidden_size);
}

/**
 * xdrope
 * # https://github.com/TencentARC/ARC-Hunyuan-Video-7B/model_vllm/hunyuan.py
 * 实现上与mrope比较类似
 */
void ModelInput::PrepareXDRopePos(const std::vector<ForwardRequest>& forward_reqs) {
  constexpr int kXDRotaryEmbeddingPosFactor = 4;
  std::vector<int64_t> xdrotary_embedding_pos_host;

  for (size_t bs_idx = 0; bs_idx < forward_reqs.size(); ++bs_idx) {
    const std::vector<Tensor>& additional_tensors = (*forward_reqs[bs_idx].input_refit_embedding).additional_tensors;
    if (additional_tensors.empty()) {
      *forward_reqs[bs_idx].xdrotary_embedding_pos_offset = 0;
    } else {
      KLLM_CHECK_WITH_INFO(
          additional_tensors.size() == 2,
          "For visual inputs, additional_tensors must contain 2 tensors: position tensor and offset tensor.");
      *forward_reqs[bs_idx].xdrotary_embedding_pos_offset = additional_tensors[1].GetPtr<int64_t>()[0];
    }
    // only needs to handle the prefill requests in its dp group
    if (forward_reqs[bs_idx].GetType() != ForwardRequestType::kFlash ||
        forward_reqs[bs_idx].attn_dp_group_id != attn_dp_group_id_) {
      continue;
    }
    if (forward_reqs[bs_idx].kv_cached_token_num == 0) {  // 首token的prefill情况
      if (additional_tensors.empty()) {                   // This is a plain text input.
        int64_t list_size = forward_reqs[bs_idx].forwarding_tokens->size() * kXDRotaryEmbeddingPosFactor;
        xdrotary_embedding_pos_host.reserve(xdrotary_embedding_pos_host.size() + list_size);
        for (int64_t i = 0; i < list_size; i += kXDRotaryEmbeddingPosFactor) {
          for (int t = 0; t < kXDRotaryEmbeddingPosFactor; ++t) {
            xdrotary_embedding_pos_host.emplace_back(i);
          }
        }
        continue;
      } else {  // This is a input with visual information.
        int64_t tensor_size = additional_tensors[0].GetElementNumber();
        int64_t xdrotart_embedding_pos_index = xdrotary_embedding_pos_host.size();
        xdrotary_embedding_pos_host.resize(xdrotary_embedding_pos_host.size() + tensor_size);
        std::memcpy(xdrotary_embedding_pos_host.data() + xdrotart_embedding_pos_index,
                    additional_tensors[0].GetPtr<void>(), sizeof(int64_t) * tensor_size);
      }
    } else {  // decode情况，对应get_next_input_positions
      const size_t input_len =
          forward_reqs[bs_idx].forwarding_tokens->size() - forward_reqs[bs_idx].kv_cached_token_num;
      const auto pos_offset = *forward_reqs[bs_idx].xdrotary_embedding_pos_offset;
      xdrotary_embedding_pos_host.reserve(xdrotary_embedding_pos_host.size() + kXDRotaryEmbeddingPosFactor * input_len);
      for (int i = 0; i < input_len; ++i) {
        for (int t = 0; t < kXDRotaryEmbeddingPosFactor; ++t) {
          xdrotary_embedding_pos_host.emplace_back(forward_reqs[bs_idx].kv_cached_token_num + pos_offset + i);
        }
      }
    }
  }
  MemcpyAsync(dp_xdrotary_embedding_pos.GetPtr<void>(), xdrotary_embedding_pos_host.data(),
              sizeof(int64_t) * xdrotary_embedding_pos_host.size(), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetD2HStream());

#ifdef ENABLE_ACL
  StreamSynchronize(context_->device->GetD2HStream());
#endif
}

#ifdef ENABLE_ACL
void ModelInput::PrepareATBKVCache(const std::vector<ForwardRequest>& forward_reqs, bool is_multi_token_forward) {
  std::shared_ptr<BlockAllocatorManager> block_allocator_manager = forward_reqs.front().block_allocator_manager;
  std::shared_ptr<BlockAllocatorInterface> device_allocator = block_allocator_manager->GetDeviceBlockAllocator(rank_);

  const auto block_allocator_group = block_allocator_manager->GetBlockAllocatorGroup(attn_dp_group_id_);
  size_t device_block_num = block_allocator_group->GetBlockAllocatorGroupConfig().device_block_num;

  // NOTE(karlluo): block manager will change the block number in
  // ResetPreAllocatedBlocks, block_manager's allocator's blocks_num is difference from the allocator's member config,
  // so we need get it from allocator instance.
  size_t total_block_num = device_block_num * 2 * layer_num_on_node_;
  if (total_block_num != k_cache_blocks_base.shape[0]) {
    void* cur_rank_block_base_ptr = device_allocator->GetBlocksBasePtr();
    void* k_cache_base_ptr = cur_rank_block_base_ptr;
    void* v_cache_base_ptr = cur_rank_block_base_ptr + block_size_ / 2;
    k_cache_blocks_base =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
               {device_block_num * 2 * layer_num_on_node_, runtime_config_.attn_backend_config.block_token_num,
                model_config_.head_num, model_config_.size_per_head},
               k_cache_base_ptr);
    v_cache_blocks_base =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
               {device_block_num * 2 * layer_num_on_node_, runtime_config_.attn_backend_config.block_token_num,
                model_config_.head_num, model_config_.size_per_head},
               v_cache_base_ptr);
  }

  uint32_t batch_size = forward_reqs.size();
  layers_slot_mapping_host.clear();
  layers_block_table_host.clear();
  size_t max_num_blocks_per_query = 0;
  last_token_index_tensor.shape = {batch_size};
  last_token_index_tensor.dtype = TYPE_UINT64;
  std::vector<int64_t> last_token_index_host(batch_size, 0);
  // for multi-token forwarding: slot_mapping shape is [num_layers, all_reqs_tokens]
  // for single-token forwarding: slot_mapping shape is [num_layers, batch_size]
  size_t all_seq_len = 0;
  size_t slot_mapping_dim_1 = is_multi_token_forward ? 0ul : batch_size;
  for (size_t f_req_idx = 0; f_req_idx < batch_size; ++f_req_idx) {
    seq_len_host.GetPtr<int32_t>()[f_req_idx] = forward_reqs[f_req_idx].forwarding_tokens->size();
    if (is_multi_token_forward) {
      slot_mapping_dim_1 += forward_reqs[f_req_idx].forwarding_tokens->size();
      last_token_index_host[f_req_idx] = all_seq_len + forward_reqs[f_req_idx].forwarding_tokens->size() - 1;
    } else {
      max_num_blocks_per_query = std::max(max_num_blocks_per_query,
                                          forward_reqs[f_req_idx].atb_kv_cache_base_blk_ids[attn_dp_rank_id_].size());
      last_token_index_host[f_req_idx] = f_req_idx;
    }
    all_seq_len += forward_reqs[f_req_idx].forwarding_tokens->size();
  }
  layers_slot_mapping_host.resize(layer_num_on_node_ * slot_mapping_dim_1, 0);
  // NOTE(karlluo): for ATB, all device blocks locate on a flatten plane memory space.
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
  // More detail refer to docs/Technology/kvcache-relationship-between-ascend-atb-and-ksana.md

  kv_cache_ptrs.clear();
  for (size_t f_req_idx = 0; f_req_idx < batch_size; ++f_req_idx) {
    if (forward_reqs[f_req_idx].attn_dp_group_id == attn_dp_group_id_) {
      kv_cache_ptrs.insert(kv_cache_ptrs.end(), forward_reqs[f_req_idx].kv_cache_ptrs.begin(),
                           forward_reqs[f_req_idx].kv_cache_ptrs.end());
    }
  }
  if (!kv_cache_ptrs.empty()) {
    memcpy(kv_cache_ptrs_tensor.GetPtr<void>(), kv_cache_ptrs.data(), kv_cache_ptrs.size() * sizeof(void*));
  }

  if (is_multi_token_forward) {
    size_t layers_slot_mapping_offset = 0;
    for (size_t f_req_idx = 0; f_req_idx < batch_size; ++f_req_idx) {
      if (forward_reqs[f_req_idx].attn_dp_group_id == attn_dp_group_id_) {
        for (size_t layer_idx = 0; layer_idx < layer_num_on_node_; ++layer_idx) {
          for (size_t token_idx = 0; token_idx < forward_reqs[f_req_idx].forwarding_tokens->size(); ++token_idx) {
            int32_t inner_block_offset = token_idx % runtime_config_.attn_backend_config.block_token_num;
            layers_slot_mapping_host[layer_idx * slot_mapping_dim_1 + layers_slot_mapping_offset + token_idx] =
                (forward_reqs[f_req_idx]
                     .atb_kv_cache_base_blk_ids[attn_dp_rank_id_]
                                               [token_idx / runtime_config_.attn_backend_config.block_token_num] +
                 layer_idx) *
                    runtime_config_.attn_backend_config.block_token_num +
                inner_block_offset;
          }
        }
        layers_slot_mapping_offset += forward_reqs[f_req_idx].forwarding_tokens->size();
      }
    }
  } else {
    layers_block_table_host.resize(layer_num_on_node_ * batch_size * max_num_blocks_per_query, -1);
    for (size_t f_req_idx = 0; f_req_idx < batch_size; ++f_req_idx) {
      if (forward_reqs[f_req_idx].attn_dp_group_id == attn_dp_group_id_) {
        size_t cur_query_blocks_num = forward_reqs[f_req_idx].atb_kv_cache_base_blk_ids[attn_dp_rank_id_].size();
        for (size_t layer_idx = 0; layer_idx < layer_num_on_node_; ++layer_idx) {
          for (uint32_t base_block_idx = 0; base_block_idx < cur_query_blocks_num; ++base_block_idx) {
            layers_block_table_host[layer_idx * batch_size * max_num_blocks_per_query +
                                    f_req_idx * max_num_blocks_per_query + base_block_idx] =
                forward_reqs[f_req_idx].atb_kv_cache_base_blk_ids[attn_dp_rank_id_][base_block_idx] + layer_idx;
          }
        }
        for (size_t layer_idx = 0; layer_idx < layer_num_on_node_; ++layer_idx) {
          int32_t block_id =
              forward_reqs[f_req_idx]
                  .atb_kv_cache_base_blk_ids[attn_dp_rank_id_][(seq_len_host.GetPtr<int32_t>()[f_req_idx] - 1) /
                                                               runtime_config_.attn_backend_config.block_token_num];
          layers_slot_mapping_host[layer_idx * slot_mapping_dim_1 + f_req_idx] =
              (block_id + layer_idx) * runtime_config_.attn_backend_config.block_token_num +
              ((seq_len_host.GetPtr<int32_t>()[f_req_idx] - 1) % runtime_config_.attn_backend_config.block_token_num);
        }
      }
    }
    if (!layers_block_table_host.empty()) {
      MemcpyAsync(layers_block_table.GetPtr<void>(), layers_block_table_host.data(),
                  layers_block_table_host.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
                  context_->device->GetH2DStream());
    }
  }
  MemcpyAsync(last_token_index_tensor.GetPtr<void>(), last_token_index_host.data(), batch_size * sizeof(int64_t),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetH2DStream());
  MemcpyAsync(layers_slot_mapping.GetPtr<void>(), layers_slot_mapping_host.data(),
              layer_num_on_node_ * slot_mapping_dim_1 * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetH2DStream());
  atb_attention_attr.GetPtr<uint64_t>()[0] = slot_mapping_dim_1;
  atb_attention_attr.GetPtr<uint64_t>()[1] = max_num_blocks_per_query;
  StreamSynchronize(context_->device->GetH2DStream());
}
#endif

#ifdef ENABLE_CUDA
template <typename T>
void ModelInput::PrepareImgMask(size_t pos_num) {
  std::vector<T> mask(input_ids.shape[0], 0.0f);
  int64_t* cpu_input_refit_pos_pair = reinterpret_cast<int64_t*>(cpu_input_refit_tensor.pos_tensor.GetPtr<void>());
  size_t hidden_size = model_config_.hidden_units;
  for (size_t i = 0; i < pos_num; i++) {
    int64_t pos = cpu_input_refit_pos_pair[i * 2];
    int64_t len = cpu_input_refit_pos_pair[i * 2 + 1] / hidden_size;
    KLLM_LOG_DEBUG << "PrepareImgMask mask : " << static_cast<int>(input_ids.shape[0]) << " , start pos : " << pos
                   << " , pos len : " << len;

    if (pos + len > static_cast<int64_t>(input_ids.shape[0])) {
      KLLM_LOG_INFO << "pos + len exceeds input_ids length, set is_mask -> False";
      return;
    }
    for (int64_t j = pos; j < pos + len; j++) {
      mask[j] = 1.0f;
    }
  }
  is_mask = true;
  im_mask.shape = {input_ids.shape[0], 1};
  MemcpyAsync(im_mask.GetPtr<void>(), mask.data(), input_ids.shape[0] * sizeof(T), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetH2DStream());
}
#endif

void ModelInput::PreparePrefill() {
  auto& shared_attention_meta = *this->shared_attention_meta->GetStandard();
  auto& flash_meta = *this->flash_meta;

  shared_attention_meta.dp_dst_flexible_kv_cache_tensor.shape = {0};
  if (flash_meta.dp_reqs.empty()) {
    return;
  }

  PROFILE_EVENT_SCOPE(PreparePrefill, "PreparePrefill", rank_);
  flash_builder_->Build(flash_meta, shared_attention_meta, dp_total_prefix_len);
  use_cache = flash_meta.use_cache;
}

void ModelInput::PrepareDecode() {
  PROFILE_EVENT_SCOPE(PrepareDecode, "PrepareDecode", rank_);
  for (auto& page_meta : page_metas) {
    paged_builder_->Build(*page_meta, *shared_attention_meta);
  }
}

void ModelInput::PrepareInputIds(const std::vector<ForwardRequest>& forward_reqs) {
  PROFILE_EVENT_SCOPE(PrepareInputIds, "PrepareInputIds", rank_);
  BatchSchedulerConfig batch_scheduler_config;
  Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);
  const bool fill_cur_batch_pairs = FastPathController::GetInstance().IsEnabledAtStartup();

  input_ids_cpu.clear();
  cur_batch_pairs_cpu_.clear();
  if (fill_cur_batch_pairs) {
    cur_batch_pairs_cpu_.reserve(2 * forward_reqs.size());
  }
  input_offset_list_uint64.assign(1, 0);
  input_prefix_list_uint64.assign(1, 0);
  dp_input_prefix_list_uint64.assign(1, 0);
  multi_token_request_max_tokens = 0;
  single_token_request_max_tokens = 0;
  dp_multi_token_request_max_tokens = 0;
  dp_single_token_request_max_tokens = 0;
  dp_max_forwarding_tokens = 0;  // used for blocked_prefill

  std::vector<size_t> logits_idx_list(total_sampling_token_num_);
  size_t logits_idx_list_idx = 0;
  size_t* dp_prefill_q_offset_uint64_ptr = dp_prefill_q_offset_uint64_host.GetPtr<size_t>();
  dp_prefill_q_offset_uint64_ptr[0] = 0;
  size_t* dp_input_offset_uint64_ptr = dp_input_offset_uint64_host.GetPtr<size_t>();
  dp_input_offset_uint64_ptr[0] = 0;
  std::vector<size_t> dp_input_ids_lens(attn_dp_group_size_, 0);

  auto process_func = [&](const ForwardRequest& req) {
    const auto& forwarding_tokens = *(req.forwarding_tokens);
    const size_t input_length = forwarding_tokens.size();
    const bool in_dp_group = req.attn_dp_group_id == attn_dp_group_id_;

    // Skip prefix token(include flexible cache token)
    const size_t skip_token_num = std::max(req.kv_cached_token_num, req.prefix_cache_len);
    const size_t input_ids_len = input_length - skip_token_num;
    KLLM_LOG_DEBUG << "forwarding_tokens_num " << input_length << ", skip_token_num " << skip_token_num
                   << ", kv_cached_token_num " << req.kv_cached_token_num << ", prefix_cache_len "
                   << req.prefix_cache_len;

    input_ids_cpu.insert(input_ids_cpu.end(), forwarding_tokens.begin() + skip_token_num, forwarding_tokens.end());
    // build-ahead: 记录该 req 末位 token 在 flat input_ids 中的下标，供 Replace/WriteRing kernel 使用。
    if (fill_cur_batch_pairs && input_ids_len > 0) {
      const int64_t last_offset = static_cast<int64_t>(input_ids_cpu.size() - 1);
      cur_batch_pairs_cpu_.push_back(req.req_id);
      cur_batch_pairs_cpu_.push_back(last_offset);
    }
    dp_max_forwarding_tokens = std::max(dp_max_forwarding_tokens, input_ids_len);
    dp_input_ids_lens[req.attn_dp_group_id] += input_ids_len;
    input_offset_list_uint64.emplace_back(input_offset_list_uint64.back() + input_length);
    input_prefix_list_uint64.emplace_back(input_prefix_list_uint64.back() + skip_token_num);

    if (req.GetType() == ForwardRequestType::kFlash) {
      multi_token_request_max_tokens = std::max(multi_token_request_max_tokens, input_length);
      if (in_dp_group) {
        dp_multi_token_request_max_tokens = std::max(dp_multi_token_request_max_tokens, input_length);
        *(dp_prefill_q_offset_uint64_ptr + 1) = *dp_prefill_q_offset_uint64_ptr + input_ids_len;
        ++dp_prefill_q_offset_uint64_ptr;
        *(dp_input_offset_uint64_ptr + 1) = *dp_input_offset_uint64_ptr + input_length;
        ++dp_input_offset_uint64_ptr;
        dp_input_prefix_list_uint64.emplace_back(dp_input_prefix_list_uint64.back() + skip_token_num);
      }
    } else {
      single_token_request_max_tokens = std::max(single_token_request_max_tokens, input_length);
      if (in_dp_group) {
        dp_single_token_request_max_tokens = std::max(dp_single_token_request_max_tokens, input_length);
      }
    }

    if (req.GetType() == ForwardRequestType::kFlash &&
        req.logits_custom_length > 0) {  // Specify the range of logits required
      for (const auto& [l, r] : req.request_target->at("logits").slice_pos) {
        std::iota(logits_idx_list.begin() + logits_idx_list_idx,
                  logits_idx_list.begin() + logits_idx_list_idx + r - l + 1, input_ids_cpu.size() - input_length + l);
        logits_idx_list_idx += r - l + 1;
      }
    } else {
      // In the standard case, only the logits of the last token are needed
      // In the case of speculative decoding, logits are required for both the last token and the predicted token
      std::iota(logits_idx_list.begin() + logits_idx_list_idx,
                logits_idx_list.begin() + logits_idx_list_idx + req.sampling_token_num,
                input_ids_cpu.size() - req.sampling_token_num);
      logits_idx_list_idx += req.sampling_token_num;
    }
  };

  for (const auto& req : forward_reqs) {
    process_func(req);
  }

  KLLM_LOG_DEBUG << "input_ids_cpu " << input_ids_cpu;
  KLLM_LOG_DEBUG << "logits_idx_list " << logits_idx_list;
  KLLM_LOG_DEBUG << "input_offset_list_uint64 " << input_offset_list_uint64;
  KLLM_LOG_DEBUG << "input_prefix_list_uint64 " << input_prefix_list_uint64;
  KLLM_LOG_DEBUG << "dp_input_prefix_list_uint64 " << dp_input_prefix_list_uint64;

  input_ids.shape = {input_ids_cpu.size()};
  input_ids.dtype = TYPE_INT32;
  MemcpyAsync(input_ids.GetPtr<void>(), input_ids_cpu.data(),
              input_ids_cpu.size() * sizeof(decltype(input_ids_cpu)::value_type), MEMCPY_HOST_TO_DEVICE,
              GetParseH2DStream());

  if (fill_cur_batch_pairs) {
    cur_batch_pairs_tensor.shape = {cur_batch_pairs_cpu_.size()};
    cur_batch_pairs_tensor.dtype = TYPE_INT64;
    if (!cur_batch_pairs_cpu_.empty()) {
      MemcpyAsync(cur_batch_pairs_tensor.GetPtr<void>(), cur_batch_pairs_cpu_.data(),
                  cur_batch_pairs_cpu_.size() * sizeof(int64_t), MEMCPY_HOST_TO_DEVICE, GetParseH2DStream());
    }
  } else {
    cur_batch_pairs_tensor.shape = {0};
  }

  logits_idx_uint64_tensor.shape = {logits_idx_list.size()};
  logits_idx_uint64_tensor.dtype = TYPE_UINT64;
  MemcpyAsync(logits_idx_uint64_tensor.GetPtr<void>(), logits_idx_list.data(),
              logits_idx_list.size() * sizeof(decltype(logits_idx_list)::value_type), MEMCPY_HOST_TO_DEVICE,
              GetParseH2DStream());

  dp_prefill_q_offset_uint64_host.shape = {forward_reqs.size() + 1};
  dp_prefill_q_offset_uint64_tensor.shape = {dp_prefill_q_offset_uint64_host.shape[0]};
  MemcpyAsync(dp_prefill_q_offset_uint64_tensor.GetPtr<void>(), dp_prefill_q_offset_uint64_host.GetPtr<void>(),
              dp_prefill_q_offset_uint64_host.shape[0] * sizeof(size_t), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetH2DStream());

  input_offset_uint64_tensor.shape = {input_offset_list_uint64.size()};
  input_offset_uint64_tensor.dtype = TYPE_UINT64;
  MemcpyAsync(input_offset_uint64_tensor.GetPtr<void>(), input_offset_list_uint64.data(),
              input_offset_list_uint64.size() * sizeof(decltype(input_offset_list_uint64)::value_type),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetH2DStream());

  dp_input_offset_uint64_host.shape = {forward_reqs.size() + 1};
  dp_input_offset_uint64_tensor.shape = {dp_input_offset_uint64_host.shape[0]};
  MemcpyAsync(dp_input_offset_uint64_tensor.GetPtr<void>(), dp_input_offset_uint64_host.GetPtr<void>(),
              dp_input_offset_uint64_host.shape[0] * sizeof(size_t), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetH2DStream());

  input_prefix_uint64_tensor.shape = {input_prefix_list_uint64.size()};
  input_prefix_uint64_tensor.dtype = TYPE_UINT64;
  MemcpyAsync(input_prefix_uint64_tensor.GetPtr<void>(), input_prefix_list_uint64.data(),
              input_prefix_list_uint64.size() * sizeof(decltype(input_prefix_list_uint64)::value_type),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetH2DStream());

  dp_input_prefix_uint64_tensor.shape = {dp_input_prefix_list_uint64.size()};
  dp_input_prefix_uint64_tensor.dtype = TYPE_UINT64;
  MemcpyAsync(dp_input_prefix_uint64_tensor.GetPtr<void>(), dp_input_prefix_list_uint64.data(),
              dp_input_prefix_list_uint64.size() * sizeof(decltype(dp_input_prefix_list_uint64)::value_type),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetH2DStream());

  size_t attn_dp_group_offset = 0;
  for (size_t i = 0; i < attn_dp_group_size_; ++i) {
    attn_dp_group_offsets_[i] = attn_dp_group_offset;
    attn_dp_group_offset += dp_input_ids_lens[i];
  }
  KLLM_LOG_DEBUG << "attn_dp_group_offsets_ " << attn_dp_group_offsets_;

  EventRecord(input_ids_event, GetParseH2DStream());
#ifdef ENABLE_ACL
  // Event wait between streams seems not work, force sync here.
  StreamSynchronize(GetParseH2DStream());
#endif
}

}  // namespace ksana_llm
