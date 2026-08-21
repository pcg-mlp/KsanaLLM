/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <sys/stat.h>
#include <atomic>
#include <functional>
#include <unordered_map>
#include <vector>

#include "pd_v2/public/pd_v2_types.h"
#include "pybind11/pybind11.h"
#include "runtime/structured_generation/structured_generator_interface.h"
#include "utils/grammar_matcher.h"
#include "utils/id_generator.h"
#include "utils/status.h"
#include "utils/tensor.h"
#include "utils/waiter.h"

namespace py = pybind11;

namespace ksana_llm {

struct SamplingConfig {
  int topk = 1;
  int num_beams = 1;
  int num_return_sequences = 1;

  // The smallest set of most probable tokens with probabilities that add up
  // to or higher than `topp` are considered
  float topp = 1.0f;

  // Modulate the next token probabilities
  float temperature = 1.0f;

  // The parameter for repetition penalty. 1.0 means no penalty
  float repetition_penalty = 1.0f;

  // The parameter for length penalty. 1.0 means no penalty
  float length_penalty = 1.0f;

  // Tokens that stop the generation when they are generated.
  // The returned tokens will contain the stop tokens.
  std::vector<int> stop_token_ids;

  // Whether to ignore any EOS tokens and continue generating
  // tokens after an EOS token is generated.
  bool ignore_eos = false;

  // The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
  int max_new_tokens = 1024;

  int logprobs_num = 0;

  // The parameter for no_repeat_ngram_size.
  int no_repeat_ngram_size = 0;
  int encoder_no_repeat_ngram_size = 0;
  int decoder_no_repeat_ngram_size = 0;

  // In generation phasse, when stop strings are meet, the request will be stopped and truncated
  std::vector<std::string> stop_strings;

  // Structured output constraints. At most one may be set for a request.
  std::string json_schema;
  std::string regex;
  std::string grammar;
  std::string structural_tag;
  bool enable_structured_output = false;

  // The current flag is used for controlling structured output.
  bool enable_thinking = true;

  // Per-request PTP draft token number.
  // -1 means unset (use global ptp_step_num), 0 means disable PTP, >0 means use specified number.
  int ptp_token_num = -1;

  // Check and adjust sampling config arguments.
  Status VerifyArgs();
};

typedef std::tuple<std::vector<int>, std::vector<std::vector<std::pair<int, float>>>, float> OutputTuple;

struct __attribute__((visibility("default"))) EmbeddingSlice {
  // The pos indicates the start position of the embedding to be replaced.
  std::vector<int> pos;

  // embeddings is the embedding value to be used for the replacement, from the request.
  std::vector<std::vector<float>> embeddings;

  // embeddings reinterpreted as uint8 bytes
  std::vector<std::vector<uint8_t>> embedding_bytes;
  // original data type of emebddings (fp32/fp16/bf16)
  std::string embedding_dtype = "fp32";

  // The same as embeddings but is python object
  std::vector<py::object> embedding_pyobjects;
  std::vector<Tensor> embedding_tensors;

  // Additional tensors computed on Python side and required by C++ for further processing.
  std::vector<py::object> additional_pyobjects;
  std::vector<Tensor> additional_tensors;
};

enum TokenReduceMode {
  GATHER_ALL,
  GATHER_TOKEN_ID,
  INVALID_TYPE,
};

inline TokenReduceMode GetTokenReduceMode(const std::string& token_reduce_mode_str) {
  if (token_reduce_mode_str == "GATHER_ALL") {
    return TokenReduceMode::GATHER_ALL;
  } else if (token_reduce_mode_str == "GATHER_TOKEN_ID") {
    return TokenReduceMode::GATHER_TOKEN_ID;
  }
  return TokenReduceMode::INVALID_TYPE;
}

struct TargetDescribe {
  std::vector<int> cutoff_layer = {0};
  // The IDs of special tokens in the request target. Based on these IDs, the corresponding target tensor (hidden state,
  // logits, etc.) should be returned.
  std::vector<int> token_id;
  // The position intervals (inclusive of both ends) of token segments in the request target. The target tensor (hidden
  // state, logits, etc.) should be returned based on the defined intervals.
  std::vector<std::pair<int, int>> slice_pos;
  // The reduction operation mode for each token_id when returning values.
  TokenReduceMode token_reduce_mode;
  // The number of top logprobs to return for each token in the input.
  int input_top_logprobs_num = 0;
};

struct KsanaPythonInput {
  // The requested model name.
  std::string model_name;

  // The config of sampling.
  SamplingConfig sampling_config;

  // If return cache hit status
  bool return_cache_stat = false;

  // The tokens of this request.
  std::vector<int> input_tokens;

  // Embedding slice used to refit input embedding
  EmbeddingSlice input_refit_embedding;

  // The key is the request target, which can only be a predefined set of requestable targets {embedding_lookup,
  // layernorm, transformer, logits}.
  std::unordered_map<std::string, TargetDescribe> request_target;

  // The structured output regex to build the finite state machine.
  std::string structured_output_regex;

  // User-provided draft tokens
  std::vector<int> draft_tokens;

  // Verifiy that the above `request_target` describes valid targets.
  // This function also converts negative `slice_pos` parameters to their corresponding positive values, if exist.
  Status VerifyRequestTarget();
};

// In the Python environment, define tensor class.
struct PythonTensor {
  std::vector<uint8_t> data;
  std::vector<size_t> shape;
  std::string dtype;
};

// A vector containing pointers to FlexibleCachedCopyTask objects, which represent tasks that involve copying data
// flexibly between different memory regions.
struct FlexibleCachedCopyTask {
  FlexibleCachedCopyTask() = default;
  void Update(int dst_token_idx, int src_token_idx, int dst_block_id, int src_block_id) {
    dst_token_idx_ = dst_token_idx;
    src_token_idx_ = src_token_idx;
    dst_block_id_ = dst_block_id;
    src_block_id_ = src_block_id;
  }
  int dst_token_idx_ = 0;
  int src_token_idx_ = 0;
  int dst_block_id_ = -1;
  int src_block_id_ = -1;
};

class Request {
 public:
  // Build Request based on the given KsanaPythonInput.
  // The lifetime of the KsanaPythonInput object must be longer than Request,
  // since some members in Request are references to KsanaPythonInput.
  explicit Request(const std::shared_ptr<KsanaPythonInput>& ksana_python_input,
                   const std::shared_ptr<std::unordered_map<std::string, std::string>>& req_ctx);

  // pd_v2 Prefill intake: self-contained Request without KsanaPythonInput.
  // req_id is the cross-node id from PrefillRequest wire (HTTP req_ctx stays on Decode).
  Request(int64_t req_id, std::vector<int> input_tokens, SamplingConfig sampling_config);

  ~Request() { KLLM_LOG_DEBUG << "Request " << req_id << " destroyed"; }

  // 必须先于下面的 input_refit_embedding / request_target 引用成员声明：pd_v2 构造器会把这些
  // 引用绑定到本结构体成员，而 C++ 成员按声明顺序初始化，声明晚了引用会绑到未初始化的对象上。
  struct OwnedRefTargets {
    EmbeddingSlice input_refit_embedding;
    std::unordered_map<std::string, TargetDescribe> request_target;
  };
  std::shared_ptr<OwnedRefTargets> owned_ref_targets_;

  // Request-level ID. Uses request-id header value when available, otherwise auto-generated.
  int64_t req_id;

  // TODO(zakwang): Replace req_id
  std::vector<int64_t> req_ids;

  // The requested model name.
  std::string model_name;

  // The input tokens of this request.
  std::vector<int> input_tokens;

  // The custom length for the logits output, allowing for a specific size of logits to be generated.
  size_t logits_custom_length = 0;

  // The number of top logprobs to return for each token in the input.
  int input_top_logprobs_num = 0;

  // Embedding slice used to refit input embedding
  EmbeddingSlice& input_refit_embedding;

  // TODO(zakwang): Replace output_tokens
  std::vector<OutputTuple> output_group;

  // The intermediate result of beam_search
  std::vector<OutputTuple> beam_search_group;

  // The output tokens of this request.
  std::vector<int>& output_tokens;

  // Store token and their corresponding float probability values.
  std::vector<std::vector<std::pair<int, float>>>& logprobs;

  // Store request cache hit status.
  // Prefix cache: (prefix_len, prefix_len), hit [0, prefix_len)
  // Flexible cache v1: (prefix_len, prefix_len + flexible_len), hit [prefix_len, prefix_len + flexible_len)
  // Flexible cache v2: (start1, end1), (start2, end2), ..., hit [start1, end1), [start2, end2), ...
  bool return_cache_stat;
  std::vector<std::pair<size_t, size_t>> cache_stat;

  // The config of sampling.
  SamplingConfig sampling_config;

  // The waiter notified when request finished.
  std::shared_ptr<Waiter> waiter = nullptr;

  // The waiter notified when step finished.
  std::shared_ptr<Waiter> step_waiter = nullptr;

  // The waiter notified when request abortd.
  std::shared_ptr<Waiter> abort_waiter = nullptr;

  // TODO(zakwang): Replace finished
  std::deque<bool> finisheds;

  // Whether the request is finished.
  bool& finished;

  // Whether the request hve been aborted by client.
  std::atomic<bool> aborted{false};

  // The hits nums of prefix caching.
  int prefix_cache_hit_tokens = 0;

  // The finish status of this request.
  Status finish_status;

  // Protect parallel access for output token.
  std::mutex output_mutex;

  // The key is the request target, which can only be a predefined set of requestable targets {embedding_lookup,
  // layernorm, transformer, logits}.
  std::unordered_map<std::string, TargetDescribe>& request_target;

  // The result of request_target.
  std::unordered_map<std::string, PythonTensor> response;

  // Whether this request is the last one in a batch of requests.
  bool last_in_batch = true;

  // The arrive time (microseconds, ProfileTimer monotonic clock).
  uint64_t timestamp_in_us;

  // Access-log timestamps (microseconds, monotonic via ProfileTimer).
  // Stamped once each by atomic CAS at scheduler/runtime hooks; 0 = not yet stamped.
  std::atomic<uint64_t> scheduled_time_us{0};    // first time entered running_reqs / decoding_queue
  std::atomic<uint64_t> first_token_time_us{0};  // first decoded step observed
  std::atomic<uint64_t> finish_time_us{0};       // request stop / finish

  // pd_v2 Decode-side phase timestamps (microseconds; 0 = not stamped). Break the otherwise
  // opaque arrive→scheduled gap into admission / remote-prefill / handoff so the disaggregation
  // cost is attributable per request. Only stamped on the Decode node.
  std::atomic<uint64_t> pd_v2_blocks_allocated_time_us{0};    // KV blocks allocated (admitted)
  std::atomic<uint64_t> pd_v2_prefill_dispatched_time_us{0};  // request sent to peer Prefill
  std::atomic<uint64_t> pd_v2_prefill_complete_time_us{0};    // peer PrefillComplete (KV/first token) arrived

  // pd_v2 Decode-side: Prefill metrics received over the wire on PrefillComplete (see
  // PdPrefillMetrics). Durations are the peer Prefill's own-clock segments; logged verbatim
  // (never subtracted against this node's clock). Written once by the decode hook (EDS thread),
  // read at access-log time (other thread) — atomic for that cross-thread handoff. 0 = none.
  std::atomic<int> pd_v2_pf_prefix_cache_hit_tokens{0};
  std::atomic<uint64_t> pd_v2_pf_queue_us{0};
  std::atomic<uint64_t> pd_v2_pf_compute_us{0};
  std::atomic<uint64_t> pd_v2_pf_send_us{0};

  std::shared_ptr<std::unordered_map<std::string, std::string>> req_ctx;

  // pd_v2 Prefill admission (PdV2PrefillEndpoint → ModelDriver::Enqueue).
  bool is_pd_v2_prefill = false;
  std::vector<pd_v2::DecodeTargetLite> pd_v2_decode_targets;
  std::string pd_v2_decode_peer_addr;
  uint32_t pd_v2_decode_attn_dp_group_id = 0;
  std::function<void()> pd_v2_on_destroy;

  // Whether the request contains stop strings
  bool has_stop_strings = false;

  // grammar matcher
  std::shared_ptr<GrammarMatcherWrapper> grammar_matcher = nullptr;

  // Structured generator configuration for structured output
  StructuredGeneratorConfig structured_generator_config;

  // User-provided draft tokens
  std::vector<int> draft_tokens;

  // This key for kv transformer group.
  std::string kv_comm_group_key;

  // PD 分离：HandleSchedule 分发到 executor 前由 ModelDriver 置 true，
  // StreamingIterator 读取后清零，向 Router 发送空 chunk 作为 prefill 通知。
  std::atomic<bool> prefill_entering_forward{false};

 private:
  // The id generator
  inline static IdGenerator id_generator_;
};

struct KsanaPythonOutput {
  KsanaPythonOutput() = default;

  // Build KsanaPythonOutput based on the specified Request.
  explicit KsanaPythonOutput(std::shared_ptr<Request> req);

  // The input tokens of this request.
  std::vector<int> input_tokens;

  // The output tokens of this request.
  std::vector<std::vector<int>> output_tokens;

  // The hits nums of prefix caching.
  int prefix_cache_hit_tokens = 0;

  // Cumulative draft acceptance stats over the request lifetime, for access-log hit-rate.
  int draft_hit_tokens = 0;  // accepted draft tokens
  int draft_tokens = 0;      // proposed draft tokens

  // Store token and their corresponding float probability values.
  std::vector<std::vector<std::vector<std::pair<int, float>>>> logprobs;

  // Store request cache hit status.
  std::vector<std::pair<size_t, size_t>> cache_stat;

  // Embedding value for plugin output
  std::vector<std::vector<float>> embedding;

  // The result of request_target.
  std::unordered_map<std::string, PythonTensor> response;

  // The top-k token IDs and logprobs at specified consecutive positions of the input tokens for the
  // request. shape: [slice_pos, input_top_logprobs_num]
  std::vector<std::vector<std::pair<int, float>>> input_top_logprobs;

  // The finish status of this request.
  Status finish_status;

  // Access-log timestamps copied from Request (microseconds).
  uint64_t arrive_time_us = 0;
  uint64_t scheduled_time_us = 0;
  uint64_t first_token_time_us = 0;
  uint64_t finish_time_us = 0;

  // pd_v2 Decode-side phase timestamps copied from Request (microseconds; 0 = not stamped).
  uint64_t pd_v2_blocks_allocated_time_us = 0;
  uint64_t pd_v2_prefill_dispatched_time_us = 0;
  uint64_t pd_v2_prefill_complete_time_us = 0;

  // pd_v2 peer-Prefill metrics copied from Request (Prefill own-clock durations + prefix hit).
  int pd_v2_pf_prefix_cache_hit_tokens = 0;
  uint64_t pd_v2_pf_queue_us = 0;
  uint64_t pd_v2_pf_compute_us = 0;
  uint64_t pd_v2_pf_send_us = 0;
  // Selected peer Prefill engine server_name (empty / "-" for local prefill).
  std::string pd_v2_prefill_peer;
};

}  // namespace ksana_llm
