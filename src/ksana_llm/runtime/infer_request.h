/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/runtime/model_instance.h"
#include "ksana_llm/utils/calc_intvec_hash.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

// The infer request, it is the unit of batch manager's scheduler.
class InferRequest {
 public:
  InferRequest(std::shared_ptr<Request> &request, int index);
  ~InferRequest();

  void SetReqGroup(const std::vector<std::shared_ptr<InferRequest>> &beam_search_infer_group) {
    req_group = beam_search_infer_group;
  }

  // Clear the group of requests.
  void ClearReqGroup() { req_group.clear(); }

  // Notify after request finished.
  void Notify();

  // Notify after step finished.
  void NotifyStep();

  // Get logits ptr on every device, that is, output of forward and input of sampling.
  std::vector<float *> GetLogitsPtr();

  // Get addr ptr of blocks.
  std::vector<std::vector<void *>> GetBlockPtrs();

 public:
  // The req id of the user's request.
  int64_t req_id;

  // The name of model instance.
  std::string &model_name;

  // The custom length for the logits output, allowing for a specific size of logits to be generated.
  size_t logits_custom_length = 0;

  // The input tokens.
  std::vector<int> &input_tokens;

  // Embedding slice used to refit input embedding
  EmbeddingSlice &input_refit_embedding;

  // The offset for multimodal rotary position embedding, computed in prefill phase by Python plugin,
  // and used in decode phase.
  int64_t mrotary_embedding_pos_offset = 0;

  // output_tokens is used during computation. When split fuse is enabled, output_tokens contains only the split
  // part. This variable is dynamically updated based on the current computation phase and may not always represent the
  // complete output.
  std::vector<int> &output_tokens;

  // complete_output_tokens holds the entire set of output tokens. It is used to preserve the full output state,
  // especially when working with split fuse operations. Before each scheduling when split fuse is not fully completed,
  // output_tokens is restored from complete_output_tokens to ensure consistency. Additionally, upon re-computation,
  // complete_output_tokens is updated to reflect the latest state of output_tokens, ensuring that it always
  // represents the most current complete output.
  std::vector<int> complete_output_tokens;

  // Store token and their corresponding float probability values.
  std::vector<std::vector<std::pair<int, float>>> &logprobs;

  // The key is the request target, which can only be a predefined set of requestable targets {embedding_lookup,
  // layernorm, transformer, logits}.
  const std::map<std::string, TargetDescribe> &request_target;

  // The result of request_target.
  std::map<std::string, PythonTensor> &response;

  float cumulative_score;

  // The sampling config of this request.
  SamplingConfig &sampling_config;

  // The waiter used to notify when request finished.
  std::shared_ptr<Waiter> &waiter;

  // The waiter used to notify when step finished.
  std::shared_ptr<Waiter> &step_waiter;

  // The waiter used to notify when request aborted..
  std::shared_ptr<Waiter> &abort_waiter;

  // Whether the request is finished.
  bool &finished;

  // whether the request is aborted.
  bool &aborted;

  // The final status of this request.
  Status &finish_status;

  // Protect parallel access for output token.
  std::mutex &output_mutex;

  // The padded token num.
  int &padded_size;

  std::vector<std::shared_ptr<InferRequest>> req_group;

  // The intermediate result of beam_search
  std::vector<OutputTuple> &beam_search_group;

  // The model instance pointer.
  std::shared_ptr<ModelInstance> model_instance;

  // Padding token id of the model.
  int pad_id;

  // context decode or decode stage.
  InferStage infer_stage;

  // The decode step, 0 for context decode, and then 1, 2, 3...
  int step = 0;

  // The number of tokens for which kv caches have been generated.
  int kv_cached_token_num = 0;

  // The kv cache blocks this request used, the index is used as device_id.
  // The key and value are stored in same blocks.
  std::vector<std::vector<int>> kv_cache_blocks;

  // The max token number of one block.
  size_t block_token_num;

  // The offset for model forward's logits output.
  size_t logits_offset = 0;

  // Whether the current req is in pending status of swappiness.
  bool swap_pending = false;

  // The swappiness future.
  std::future<void> swap_future;

  // The flag for tagging request prefix cache usage
  bool is_use_prefix_cache = false;

  // The prefix cache tokens number
  int prefix_cache_len = 0;

  // The prefix cache blocks number
  int prefix_cache_blocks_number = 0;

  // A vector containing pointers to FlexibleCachedCopyTask objects, which represent tasks that involve copying data
  // flexibly between different memory regions.
  std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks;

  // The no_repeate ngram sampling map
  NgramDict ngram_dict;

  // is cudagraph capture call
  bool &is_cudagraph_capture_request;

  // Opentelemetry SpanContext
  opentelemetry::trace::SpanContext span_context;

  // The arrive time.
  uint64_t timestamp_in_ms;

  // request context
  std::shared_ptr<std::unordered_map<std::string, std::string>> req_ctx;

  // Incremental decoded str used in stop strings
  std::string incremental_decoded_str;

  // The output structure fsm
  std::shared_ptr<FiniteStateMachine> req_fsm;

  // The current state id of the FSM (Finite State Machine).
  size_t fsm_state_id = 0;
};

}  // namespace ksana_llm
