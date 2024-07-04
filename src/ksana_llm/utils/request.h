/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <sys/stat.h>
#include <map>
#include <vector>

#include "ksana_llm/utils/id_generator.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/waiter.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace ksana_llm {

struct SamplingConfig {
  int topk = 1;
  int num_beams = 1;
  int num_return_sequences = 1;
  float topp = 0.0f;
  float temperature = 0.0f;
  // The parameter for repetition penalty. 1.0 means no penalty
  float repetition_penalty = 1.0f;
  float length_penalty = 1.0f;

  // Tokens that stop the generation when they are generated.
  // The returned tokens will contain the stop tokens.
  std::vector<int> stop_token_ids;

  // Whether to ignore the EOS token and continue generating
  // tokens after the EOS token is generated.
  bool ignore_eos;

  // The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
  int max_new_tokens = 1024;

  int logprobs_num = 0;
};

typedef std::tuple<std::vector<int>, std::vector<std::vector<std::pair<int, float>>>, float> OutputTuple;

struct EmbeddingSlice{
  // The pos indicates the start position of the embedding to be replaced.     
  std::vector<int> pos;

  // embeddings is the embedding value to be used for the replacement, from the request.          
  std::vector<std::vector<float>> embeddings;

  // The same as embeddings but is python object
  std::vector<py::object> embedding_tensors;
};

enum TokenReduceMode {
  GATHER_ALL,
  GATHER_TOKEN_ID,
};

struct TargetDescribe {
  // The IDs of special tokens in the request target. Based on these IDs, the corresponding target tensor (hidden state,
  // logits, etc.) should be returned.
  std::vector<int> token_id;
  // The position intervals (inclusive of both ends) of token segments in the request target. The target tensor (hidden
  // state, logits, etc.) should be returned based on the defined intervals.
  std::vector<std::pair<size_t, size_t>> slice_pos;
  // The reduction operation mode for each token_id when returning values.
  TokenReduceMode token_reduce_mode;
};

struct KsanaPythonInput {
  // The requested model name.
  std::string model_name;

  // The config of sampling.
  SamplingConfig sampling_config;

  // The tokens of this request.
  std::vector<int> input_tokens;

  // Embedding slice used to refit input embedding    
  EmbeddingSlice input_refit_embedding; 

  // The key is the request target, which can only be a predefined set of requestable targets {embedding_lookup,
  // layernorm, transformer, logits}.
  std::map<std::string, TargetDescribe> request_target;
};

// In the Python environment, define tensor class.
struct PythonTensor {
  std::vector<uint8_t> data;
  std::vector<size_t> shape;
  std::string dtype;
};

struct KsanaPythonOutput {
  // The output tokens of this request.
  std::vector<std::vector<int>> output_tokens;

  // Store token and their corresponding float probability values.
  std::vector<std::vector<std::vector<std::pair<int, float>>>> logprobs;

  // Embedding value for plugin output
  std::vector<std::vector<float>> embedding;

  // The result of request_target.
  std::map<std::string, PythonTensor> response;
};

class Request {
 public:
  Request(const ksana_llm::KsanaPythonInput& ksana_python_input);

  // The unique id of a request.
  int64_t req_id;

  // TODO(zakwang): Replace req_id
  std::vector<int64_t> req_ids;

  // The requested model name.
  std::string model_name;

  // The tokens of this request.
  std::vector<int> input_tokens;

  // The custom length for the logits output, allowing for a specific size of logits to be generated.
  size_t logits_custom_length = 0;

  // Embedding slice used to refit input embedding    
  EmbeddingSlice input_refit_embedding; 

  // TODO(zakwang): Replace output_tokens
  std::vector<OutputTuple> output_group;

  // The intermediate result of beam_search
  std::vector<OutputTuple> beam_search_group;

  // The output tokens of this request.
  std::vector<int>& output_tokens;

  // The padded token num.
  int padded_size = 0;

  // Store token and their corresponding float probability values.
  std::vector<std::vector<std::pair<int, float>>>& logprobs;

  // The config of sampling.
  SamplingConfig sampling_config;

  // The waiter notified when request finished.
  std::shared_ptr<Waiter> waiter = nullptr;

  // The waiter notified when step finished.
  std::shared_ptr<Waiter> step_waiter = nullptr;

  // TODO(zakwang): Replace finished
  std::deque<bool> finisheds;

  // Whether the request is finished.
  bool& finished;

  // The finish statu of this request.
  Status finish_status;

  // Protect parallel access for output token.
  std::mutex output_mutex;

  // The key is the request target, which can only be a predefined set of requestable targets {embedding_lookup,
  // layernorm, transformer, logits}.
  const std::map<std::string, TargetDescribe>& request_target;

  // The result of request_target.
  std::map<std::string, PythonTensor> response;

 private:
  // The id generator
  static IdGenerator id_generator_;
};

}  // namespace ksana_llm
