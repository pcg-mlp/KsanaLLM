/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <vector>

#include "runtime/infer_request.h"
#include "utils/grammar_matcher.h"
#include "utils/request.h"

namespace ksana_llm {

// The information used for sampling.
struct SamplingRequest {
  // The req id of the user's request.
  int64_t req_id;

  // The custom length for the logits output, allowing for a specific size of logits to be generated.
  size_t logits_custom_length = 0;

  // The sampling config.
  SamplingConfig* sampling_config = nullptr;

  // The logits buf and offset.
  float* logits_buf = nullptr;
  size_t logits_offset = 0;

  std::shared_ptr<std::vector<int>> input_tokens;

  std::vector<int>* forwarding_tokens = nullptr;

  // Generated tokens in this sampling.
  std::vector<int>* sampling_result_tokens = nullptr;

  size_t sampling_token_num = 1;

  size_t last_step_token_num = 1;

  // The key is the request target, which can only be a predefined set of requestable targets {embedding_lookup,
  // layernorm, transformer, logits}.
  std::shared_ptr<const std::unordered_map<std::string, TargetDescribe>> request_target;

  // The result of request_target.
  std::unordered_map<std::string, PythonTensor>* response = nullptr;

  // Store token and their corresponding float probability values.
  std::shared_ptr<std::vector<std::vector<std::pair<int, float>>>> logprobs;

  // The no_reapete_ngram sampling map
  NgramDict* ngram_dict = nullptr;

  // Grammar matcher for constrained generation, defaults to nullptr (no grammar constraints)
  std::shared_ptr<GrammarMatcherWrapper> grammar_matcher;

  // Flag to control whether to apply grammar constraints during sampling
  // Used in MTP mode to disable grammar for draft token generation
  bool apply_grammar_constraint = true;

  // Flag to control whether to skip ngram_no_repeat sampling and output logprobs calculation during sampling used in
  // MTP mode to disable constraints for draft token generation
  bool enable_mtp_sampler = false;

  // Number of top logprobs to return for input tokens
  int input_top_logprobs_num = 0;
};

}  // namespace ksana_llm
