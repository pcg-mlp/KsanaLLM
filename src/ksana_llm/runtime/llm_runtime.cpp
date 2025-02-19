/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/llm_runtime.h"
#include <algorithm>
#include <execution>
#include <memory>
#include <unordered_map>
#include <vector>

#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/runtime/model_instance.h"
#include "ksana_llm/runtime/sampling_request.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

template <typename Key, typename Value, typename T>
inline Value& GetMapValue(std::unordered_map<Key, Value>& m, const Key& key, T&& default_value) {
  return m.emplace(key, std::forward<T>(default_value)).first->second;
}

LlmRuntime::LlmRuntime(const BatchSchedulerConfig& batch_scheduler_config, std::shared_ptr<Context> context)
    : batch_schedule_config_(batch_scheduler_config), context_(context) {
  worker_group_ =
      std::make_shared<WorkerGroup>(context_->GetTensorParallelSize(), context_->GetPipeLineParallelSize(), context_);

  for (int worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
    samplers_.push_back(std::make_shared<Sampler>(batch_schedule_config_, worker_id, context_));
  }
  threadpool_ = std::make_shared<ThreadPool>(2);
  threadpool_->Start();
}

void LlmRuntime::SetCacheManager(std::shared_ptr<CacheManagerInterface> cache_manager) {
  cache_manager_ = cache_manager;
}

#ifdef ENABLE_CUDA
// In a CUDA environment, it's necessary to compute perfill and decode together.
// Therefore, kGroupStageMap is required to map to a single stage for synchronized processing.
static std::unordered_map<InferStage, InferStage> kGroupStageMap = {
    {InferStage::STAGE_CONTEXT, InferStage::STATE_DECODE}, {InferStage::STATE_DECODE, InferStage::STATE_DECODE}};
#else
static std::unordered_map<InferStage, InferStage> kGroupStageMap = {
    {InferStage::STAGE_CONTEXT, InferStage::STAGE_CONTEXT}, {InferStage::STATE_DECODE, InferStage::STATE_DECODE}};
#endif

void LlmRuntime::BuildForwardRequests(
    std::vector<std::shared_ptr<InferRequest>>& reqs,
    std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>>& grouped_reqs) {
  int logits_offset = 0;
  for (size_t i = 0; i < reqs.size(); ++i) {
    std::shared_ptr<InferRequest>& req_ptr = reqs[i];
    req_ptr->step += 1;
    req_ptr->logits_offset = logits_offset;
    // When the logits_custom_length is greater than 0, the size of logits to be calculated is logits_custom_length.
    if (req_ptr->logits_custom_length > 0) {
      logits_offset += req_ptr->logits_custom_length;
    } else {
      logits_offset++;
    }
    ModelInstance* key = req_ptr->model_instance.get();
    InferStage grouped_stage = kGroupStageMap[req_ptr->infer_stage];

    if (grouped_reqs.find(key) == grouped_reqs.end()) {
      grouped_reqs[key] = {};
    }

    if (grouped_reqs[key].find(grouped_stage) == grouped_reqs[key].end()) {
      grouped_reqs[key][grouped_stage] = {};
    }

    ForwardRequest forward_req;
    forward_req.req_id = req_ptr->req_id;
    forward_req.infer_stage = req_ptr->infer_stage;
    forward_req.step = req_ptr->step;
    forward_req.kv_cached_token_num = req_ptr->kv_cached_token_num;
    forward_req.logits_custom_length = req_ptr->logits_custom_length;
    forward_req.kv_cache_ptrs = req_ptr->GetBlockPtrs();
    forward_req.logits_buf = req_ptr->GetLogitsPtr();
    forward_req.logits_offset = req_ptr->logits_offset;
    forward_req.request_target = &req_ptr->request_target;
    forward_req.response = &req_ptr->response;
    forward_req.output_tokens = &(req_ptr->output_tokens);
    forward_req.flexible_cached_copy_tasks = &(req_ptr->flexible_cached_copy_tasks);
    forward_req.input_refit_embedding = &(req_ptr->input_refit_embedding);
    forward_req.mrotary_embedding_pos_offset = &(req_ptr->mrotary_embedding_pos_offset);
    forward_req.is_use_prefix_cache = req_ptr->is_use_prefix_cache;
    forward_req.flexible_cache_len = req_ptr->flexible_cached_copy_tasks.size();
    forward_req.prefix_cache_len = req_ptr->prefix_cache_len + forward_req.flexible_cache_len;
    forward_req.prefix_cache_blocks_number = req_ptr->prefix_cache_blocks_number;
    forward_req.is_cudagraph_capture_request = req_ptr->is_cudagraph_capture_request;
    forward_req.sampling_config = &(req_ptr->sampling_config);
    forward_req.span_context = req_ptr->span_context;
    forward_req.timestamp_in_ms = req_ptr->timestamp_in_ms;
    forward_req.req_ctx = req_ptr->req_ctx;
#if defined(ENABLE_ACL) || defined(ENABLE_FLASH_ATTN_WITH_CACHE)
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
    // More detail refer to docs/Technology/kvcache-relationship-between-ascend-atb-and-ksana.md
    uint32_t layer_num = req_ptr->model_instance->GetLayerNum();
    size_t rank_num = req_ptr->kv_cache_blocks.size();
    // prepare base block ids
    forward_req.atb_kv_cache_base_blk_ids.resize(rank_num);
    for (size_t rank = 0; rank < rank_num; ++rank) {
      GetBlockManager()->SetDeviceId(rank);
      forward_req.atb_kv_cache_base_blk_ids[rank].clear();
      // for dedicate device kv blocks
      std::vector<int32_t> device_kv_cache_block_ids;
      for (int ksana_block_id : req_ptr->kv_cache_blocks[rank]) {
        size_t original_block_id = (ksana_block_id - GetBlockManager()->GetBlocksBaseId());
        // NOTE(karlluo): only support bfloat16 or float16, so we just dedicate sizeof(float16) here is fine
        int32_t ascend_atb_block_id = original_block_id * layer_num * 2;
        device_kv_cache_block_ids.push_back(ascend_atb_block_id);
      }
      forward_req.atb_kv_cache_base_blk_ids[rank].insert(forward_req.atb_kv_cache_base_blk_ids[rank].end(),
                                                         device_kv_cache_block_ids.begin(),
                                                         device_kv_cache_block_ids.end());
    }
#endif
    grouped_reqs[key][grouped_stage].push_back(forward_req);
  }
}

void LlmRuntime::BuildForwardRequests(
    std::vector<std::shared_ptr<WorkerInferRequest>>& reqs,
    std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>>& grouped_reqs) {
  int logits_offset = 0;
  for (size_t i = 0; i < reqs.size(); ++i) {
    std::shared_ptr<WorkerInferRequest>& req_ptr = reqs[i];
    req_ptr->step += 1;
    ModelInstance* key = req_ptr->model_instance.get();
    InferStage grouped_stage = kGroupStageMap[req_ptr->infer_stage];

    if (grouped_reqs.find(key) == grouped_reqs.end()) {
      grouped_reqs[key] = {};
    }

    if (grouped_reqs[key].find(grouped_stage) == grouped_reqs[key].end()) {
      grouped_reqs[key][grouped_stage] = {};
    }

    ForwardRequest forward_req;
    forward_req.req_id = req_ptr->req_id;
    forward_req.infer_stage = req_ptr->infer_stage;
    forward_req.step = req_ptr->step;
    forward_req.kv_cached_token_num = req_ptr->kv_cached_token_num;
    forward_req.logits_custom_length = 0;
    forward_req.kv_cache_ptrs = req_ptr->GetBlockPtrs();
    forward_req.logits_buf = {};
    forward_req.logits_offset = ++logits_offset;
    forward_req.request_target = &req_ptr->request_target;
    forward_req.response = &req_ptr->response;
    forward_req.output_tokens = &(req_ptr->output_tokens);
    forward_req.flexible_cached_copy_tasks = &(req_ptr->flexible_cached_copy_tasks);
    forward_req.input_refit_embedding = &(req_ptr->input_refit_embedding);
    forward_req.mrotary_embedding_pos_offset = &(req_ptr->mrotary_embedding_pos_offset);
    forward_req.is_use_prefix_cache = req_ptr->is_use_prefix_cache;
    forward_req.flexible_cache_len = 0;
    forward_req.prefix_cache_len = req_ptr->prefix_cache_len + forward_req.flexible_cache_len;
    forward_req.prefix_cache_blocks_number = req_ptr->prefix_cache_blocks_number;
    forward_req.is_cudagraph_capture_request = false;
    forward_req.timestamp_in_ms = 0;
#if defined(ENABLE_ACL) || defined(ENABLE_FLASH_ATTN_WITH_CACHE)
    uint32_t layer_num = req_ptr->model_instance->GetLayerNum();
    size_t rank_num = req_ptr->kv_cache_blocks.size();
    // prepare base block ids
    forward_req.atb_kv_cache_base_blk_ids.resize(rank_num);
    for (size_t rank = 0; rank < rank_num; ++rank) {
      GetBlockManager()->SetDeviceId(rank);
      forward_req.atb_kv_cache_base_blk_ids[rank].clear();
      // for dedicate device kv blocks
      std::vector<int32_t> device_kv_cache_block_ids;
      for (int ksana_block_id : req_ptr->kv_cache_blocks[rank]) {
        size_t original_block_id = (ksana_block_id - GetBlockManager()->GetBlocksBaseId());
        // NOTE(karlluo): only support bfloat16 or float16, so we just dedicate sizeof(float16) here is fine
        int32_t ascend_atb_block_id = original_block_id * layer_num * 2;
        device_kv_cache_block_ids.push_back(ascend_atb_block_id);
      }
      forward_req.atb_kv_cache_base_blk_ids[rank].insert(forward_req.atb_kv_cache_base_blk_ids[rank].end(),
                                                         device_kv_cache_block_ids.begin(),
                                                         device_kv_cache_block_ids.end());
    }
#endif
    grouped_reqs[key][grouped_stage].push_back(forward_req);
  }
}

Status LlmRuntime::RunSerially(
    std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>>& grouped_reqs,
    bool epilogue) {
  Status result_status = Status();
  for (auto& [model_inst, stage_vec_reqs] : grouped_reqs) {
    for (auto& [stage, vec_req] : stage_vec_reqs) {
      std::vector<std::future<Status>> inst_results = model_inst->ForwardAsync(worker_group_, stage, vec_req, epilogue);
      for (auto& worker_result : inst_results) {
        Status status = worker_result.get();
        if (!status.OK()) {
          result_status = status;
        }
      }
    }
  }
  return result_status;
}

void LlmRuntime::ReorderInferRequests(std::vector<std::shared_ptr<InferRequest>>& reqs) {
  // Due to the different calculation logic used for multi-token and single-token in the Attention layer,
  // the requests are first sorted to utilize contiguous space for accelerated inference.
  // Sort the infer_reqs list based on the number of tokens that need to be calculated for the KV cache.
  std::sort(reqs.begin(), reqs.end(),
            [](const std::shared_ptr<InferRequest>& a, const std::shared_ptr<InferRequest>& b) {
              int a_token_num = a->output_tokens.size() - a->kv_cached_token_num;
              int b_token_num = b->output_tokens.size() - b->kv_cached_token_num;
              if (a_token_num != b_token_num) {
                return a_token_num > b_token_num;
              }
              if (a->kv_cached_token_num != b->kv_cached_token_num) {
                return a->kv_cached_token_num < b->kv_cached_token_num;
              }
              return a->req_id < b->req_id;
            });
}

void LlmRuntime::ReorderInferRequests(std::vector<std::shared_ptr<WorkerInferRequest>>& reqs) {
  // Due to the different calculation logic used for multi-token and single-token in the Attention layer,
  // the requests are first sorted to utilize contiguous space for accelerated inference.
  // Sort the infer_reqs list based on the number of tokens that need to be calculated for the KV cache.
  std::sort(reqs.begin(), reqs.end(),
            [](const std::shared_ptr<WorkerInferRequest>& a, const std::shared_ptr<WorkerInferRequest>& b) {
              int a_token_num = a->output_tokens.size() - a->kv_cached_token_num;
              int b_token_num = b->output_tokens.size() - b->kv_cached_token_num;
              if (a_token_num != b_token_num) {
                return a_token_num > b_token_num;
              }
              if (a->kv_cached_token_num != b->kv_cached_token_num) {
                return a->kv_cached_token_num < b->kv_cached_token_num;
              }
              return a->req_id < b->req_id;
            });
}

void LlmRuntime::UpdateRequestKVCachedTokenNum(std::vector<std::shared_ptr<InferRequest>>& reqs) {
  for (size_t i = 0; i < reqs.size(); ++i) {
    std::shared_ptr<InferRequest>& req_ptr = reqs[i];
    req_ptr->kv_cached_token_num = req_ptr->output_tokens.size() - 1;
  }
}

Status LlmRuntime::Forward(std::vector<std::shared_ptr<InferRequest>>& reqs, bool epilogue) {
  std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>> grouped_reqs;
  BuildForwardRequests(reqs, grouped_reqs);

  return AuxForward(grouped_reqs, epilogue);
}

Status LlmRuntime::Forward(std::vector<std::shared_ptr<WorkerInferRequest>>& reqs, bool epilogue) {
  std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>> grouped_reqs;
  BuildForwardRequests(reqs, grouped_reqs);

  return AuxForward(grouped_reqs, epilogue);
}

Status LlmRuntime::AuxForward(
    std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>>& grouped_reqs,
    bool epilogue) {
  // context decode and decode run serially in single thread
  if (context_->IsRunContextDecodeAndDecodeSerially()) {
    // Wait all instances done and check status.
    return RunSerially(grouped_reqs, epilogue);
  }

  std::vector<std::vector<std::future<Status>>> results;
  for (auto& [model_inst, stage_vec_reqs] : grouped_reqs) {
    for (auto& [stage, vec_req] : stage_vec_reqs) {
      results.push_back(model_inst->ForwardAsync(worker_group_, stage, vec_req, epilogue));
    }
  }

  // Wait all instances done and check status.
  Status result_status = Status();
  for (auto& inst_results : results) {
    for (auto& worker_result : inst_results) {
      Status status = worker_result.get();
      if (!status.OK()) {
        result_status = status;
      }
    }
  }

  return result_status;
}

void LlmRuntime::BuildSamplingRequest(std::vector<std::shared_ptr<InferRequest>>& reqs,
                                      std::vector<SamplingRequest>& sampling_reqs) {
  for (std::shared_ptr<InferRequest> req_ptr : reqs) {
    SamplingRequest sampling_req;
    sampling_req.req_id = req_ptr->req_id;
    sampling_req.logits_custom_length = req_ptr->logits_custom_length;
    sampling_req.input_tokens = &(req_ptr->input_tokens);
    sampling_req.output_tokens = &(req_ptr->output_tokens);
    sampling_req.response = &(req_ptr->response);
    sampling_req.request_target = &(req_ptr->request_target);
    sampling_req.logprobs = &(req_ptr->logprobs);
    sampling_req.output_mutex = &(req_ptr->output_mutex);
    sampling_req.logits_offset = req_ptr->logits_offset;
    sampling_req.logits_buf = req_ptr->GetLogitsPtr();
    sampling_req.sampling_config = &(req_ptr->sampling_config);
    sampling_req.req_group = &(req_ptr->req_group);
    sampling_req.req_ctx = req_ptr->req_ctx;
    if (sampling_req.sampling_config->num_beams > 1) {
      sampling_req.sampling_config->logprobs_num =
          std::max(sampling_req.sampling_config->logprobs_num, sampling_req.sampling_config->num_beams);
      sampling_req.sampling_config->topk =
          std::max(sampling_req.sampling_config->topk, sampling_req.sampling_config->num_beams);
    }
    sampling_req.ngram_dict = &(req_ptr->ngram_dict);
    sampling_req.model_config = &(req_ptr->model_instance->GetModelConfig());
    sampling_reqs.push_back(sampling_req);
  }
}

Status LlmRuntime::Sampling(std::vector<std::shared_ptr<InferRequest>>& reqs) {
  std::vector<SamplingRequest> sampling_reqs;
  BuildSamplingRequest(reqs, sampling_reqs);

  std::vector<std::future<Status>> results;
  for (int worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
    results.push_back(worker_group_->GetWorker(worker_id)->SamplingAsync(samplers_[worker_id], sampling_reqs));
  }

  // Wait all instances done and check status.
  Status result_status = Status();
  for (auto& result : results) {
    Status status = result.get();
    if (!status.OK()) {
      result_status = status;
    }
  }

  threadpool_->Submit([reqs]() mutable {
    const auto current_time = ProfileTimer::GetCurrentTimeInMs();
    std::for_each(std::execution::par_unseq, reqs.begin(), reqs.end(), [current_time](const auto& req) {
      if (req->infer_stage == STAGE_CONTEXT) {
        REPORT_METRIC(time_to_first_token_ms, current_time - req->timestamp_in_ms);
      }
    });
  });

  return result_status;
}

Status LlmRuntime::Step(ScheduleOutput* schedule_output, bool epilogue) {
  KLLM_LOG_DEBUG << "llm runtime step invoked.";

  if (context_->IsChief()) {
    ReorderInferRequests(schedule_output->running_reqs);
  } else {
    ReorderInferRequests(schedule_output->worker_running_reqs);
  }

  if (!epilogue) {
    if (!context_->IsChief()) {
      if (!schedule_output->swapout_req_block_ids.empty()) {
        for (auto it = schedule_output->swapout_req_block_ids.begin();
             it != schedule_output->swapout_req_block_ids.end(); ++it) {
          KLLM_LOG_DEBUG << "Swapout req " << it->first << ", block num:" << it->second.size();
          Status status = cache_manager_->SwapoutRequestMemoryBlockAsync(it->first, it->second);
          if (!status.OK()) {
            return status;
          }
        }
      }

      if (!schedule_output->swapin_req_block_ids.empty()) {
        for (auto it = schedule_output->swapin_req_block_ids.begin(); it != schedule_output->swapin_req_block_ids.end();
             ++it) {
          KLLM_LOG_DEBUG << "Swapin req " << it->first << ", block num:" << it->second.size();
          Status status = cache_manager_->SwapinRequestMemoryBlockAsync(it->first, it->second);
          if (!status.OK()) {
            return status;
          }
        }
      }

      // For worker, Waiting necessary swappiness finished.
      // Must invoked after SwapxxxRequestMemoryBlockAsync,
      // because we may need swap and wait same block id in one step.
      KLLM_LOG_DEBUG << "Wait swapout reqs finish, size:" << schedule_output->merged_swapout_req_ids.size();
      cache_manager_->WaitSwapoutRequestMemoryBlock(schedule_output->merged_swapout_req_ids);

      KLLM_LOG_DEBUG << "Wait swapin reqs finish, size:" << schedule_output->merged_swapin_req_ids.size();
      cache_manager_->WaitSwapinRequestMemoryBlock(schedule_output->merged_swapin_req_ids);
    }
  }

  // Inference forward.
  if (context_->IsChief()) {
    Forward(schedule_output->running_reqs, epilogue);
  } else {
    Forward(schedule_output->worker_running_reqs, epilogue);
  }

  // Sampling only in standalone mode.
  if (context_->IsStandalone() || epilogue) {
    Sampling(schedule_output->running_reqs);
    UpdateRequestKVCachedTokenNum(schedule_output->running_reqs);
  }

  return Status();
}

}  // namespace ksana_llm
