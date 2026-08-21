/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "utils/request.h"
#include "profiler/reporter.h"
#include "utils/singleton.h"

namespace ksana_llm {

Status SamplingConfig::VerifyArgs() {
  if (topp == 0.f) {
    topp = 1.f;
  }
  if (temperature == 0.f) {
    temperature = 1.f;
  }
  // The TopkSampling kernel of TensorRT requires that `1 <= topk <= 1024`.
  // Refer to
  // https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/samplingTopKKernels.cu#L377
  if (topk < 1 || topk > 1024) {
    return Status(RET_INVALID_ARGUMENT, fmt::format("topk should be between 1 and 1024, but {} was provided", topk));
  }
  if ((encoder_no_repeat_ngram_size > 0) + (decoder_no_repeat_ngram_size > 0) + (no_repeat_ngram_size > 0) > 1) {
    return Status(RET_INVALID_ARGUMENT,
                  "no_repeat_ngram_size/encoder_no_repeat_ngram_size/decoder_no_repeat_ngram_size"
                  " can not be used at the same time");
  }
  const int structured_constraint_count = !json_schema.empty() + !regex.empty() + !grammar.empty() +
                                          !structural_tag.empty();
  if (structured_constraint_count > 1) {
    return Status(RET_INVALID_ARGUMENT,
                  "json_schema/regex/grammar/structural_tag can not be used at the same time");
  }
  return Status();
}

namespace {

size_t OutputGroupSize(const SamplingConfig& sc) {
  return std::max(std::max(sc.num_beams, sc.num_return_sequences), 1);
}

const char* StructuredConstraintTypeName(StructuredConstraintType type) {
  switch (type) {
    case StructuredConstraintType::NONE:
      return "NONE";
    case StructuredConstraintType::JSON:
      return "JSON";
    case StructuredConstraintType::REGEX:
      return "REGEX";
    case StructuredConstraintType::GRAMMAR:
      return "GRAMMAR";
    case StructuredConstraintType::STRUCTURAL_TAG:
      return "STRUCTURAL_TAG";
  }
  return "UNKNOWN";
}

void ApplyStructuredGeneratorConfig(Request* req) {
  if (req == nullptr) {
    return;
  }
  req->structured_generator_config = {};
  if (!req->sampling_config.enable_structured_output) {
    KLLM_LOG_DEBUG << "Request " << req->req_id << " structured output disabled; cleared structured generator config: "
                   << "json_schema.empty()=" << req->sampling_config.json_schema.empty()
                   << ", regex.empty()=" << req->sampling_config.regex.empty()
                   << ", grammar.empty()=" << req->sampling_config.grammar.empty()
                   << ", structural_tag.empty()=" << req->sampling_config.structural_tag.empty();
    return;
  }

  const std::pair<StructuredConstraintType, const std::string*> constraints[] = {
      {StructuredConstraintType::JSON, &req->sampling_config.json_schema},
      {StructuredConstraintType::REGEX, &req->sampling_config.regex},
      {StructuredConstraintType::GRAMMAR, &req->sampling_config.grammar},
      {StructuredConstraintType::STRUCTURAL_TAG, &req->sampling_config.structural_tag},
  };
  for (const auto& [type, spec] : constraints) {
    if (!spec->empty()) {
      req->structured_generator_config = StructuredGeneratorConfig(type, *spec);
      KLLM_LOG_DEBUG << "Request " << req->req_id << " structured_generator_config set: "
                     << "constraint_type=" << StructuredConstraintTypeName(type)
                     << ", constraint_spec_size=" << spec->size() << ", constraint_spec=" << *spec;
      return;
    }
  }

  KLLM_LOG_DEBUG << "Request " << req->req_id
                 << " structured output enabled but no constraint spec was provided: "
                 << "json_schema.empty()=" << req->sampling_config.json_schema.empty()
                 << ", regex.empty()=" << req->sampling_config.regex.empty()
                 << ", grammar.empty()=" << req->sampling_config.grammar.empty()
                 << ", structural_tag.empty()=" << req->sampling_config.structural_tag.empty();
}

}  // namespace

Request::Request(int64_t cross_node_req_id, std::vector<int> input_tokens_in, SamplingConfig sampling_config_in)
    : owned_ref_targets_(std::make_shared<OwnedRefTargets>()),
      req_id(cross_node_req_id),
      model_name(),
      input_tokens(std::move(input_tokens_in)),
      input_refit_embedding(owned_ref_targets_->input_refit_embedding),
      output_group(OutputGroupSize(sampling_config_in)),
      output_tokens(std::get<0>(output_group[0])),
      logprobs(std::get<1>(output_group[0])),
      return_cache_stat(false),
      sampling_config(std::move(sampling_config_in)),
      finisheds(output_group.size(), false),
      finished(finisheds[0]),
      request_target(owned_ref_targets_->request_target),
      timestamp_in_us(ProfileTimer::GetCurrentTimeInUs()),
      req_ctx(nullptr),
      prefix_cache_hit_tokens(0) {
  static constexpr size_t kMaxOutputGroup = (1u << IdGenerator::kSubIdBits) - 1;
  if (output_group.size() > kMaxOutputGroup) {
    KLLM_THROW(fmt::format("output_group size {} exceeds max {}", output_group.size(), kMaxOutputGroup));
  }

  for (size_t i = 0; i < output_group.size(); ++i) {
    auto& [output, req_logprobs, total_score] = output_group[i];
    output = this->input_tokens;
    req_ids.push_back(cross_node_req_id | static_cast<int64_t>(i));
  }

  if (!sampling_config.stop_strings.empty()) {
    has_stop_strings = true;
  }

  ApplyStructuredGeneratorConfig(this);
}

Request::Request(const std::shared_ptr<KsanaPythonInput>& ksana_python_input,
                 const std::shared_ptr<std::unordered_map<std::string, std::string>>& req_ctx)
    : req_id(0),
      req_ids(),
      model_name(ksana_python_input->model_name),
      input_tokens(ksana_python_input->input_tokens),
      logits_custom_length(0),
      input_refit_embedding(ksana_python_input->input_refit_embedding),
      output_group(OutputGroupSize(ksana_python_input->sampling_config)),
      beam_search_group(),
      output_tokens(std::get<0>(output_group[0])),
      logprobs(std::get<1>(output_group[0])),
      return_cache_stat(ksana_python_input->return_cache_stat),
      cache_stat(),
      sampling_config(ksana_python_input->sampling_config),
      waiter(nullptr),
      step_waiter(nullptr),
      finisheds(output_group.size(), false),
      finished(finisheds[0]),
      finish_status(),
      output_mutex(),
      request_target(ksana_python_input->request_target),
      response(),
      timestamp_in_us(ProfileTimer::GetCurrentTimeInUs()),
      req_ctx(req_ctx),
      draft_tokens(ksana_python_input->draft_tokens) {
  static constexpr size_t kMaxOutputGroup = (1u << IdGenerator::kSubIdBits) - 1;  // 255
  if (output_group.size() > kMaxOutputGroup) {
    KLLM_THROW(fmt::format("output_group size {} exceeds max {}", output_group.size(), kMaxOutputGroup));
  }

  // Determine base_id: prefer header, otherwise auto-generate.
  int64_t base_id = id_generator_.Gen();
  if (req_ctx && req_ctx->count("request-id")) {
    try {
      base_id = std::stoll((*req_ctx)["request-id"]);
    } catch (const std::exception& e) {
      KLLM_LOG_WARNING << "Invalid request-id header '" << (*req_ctx)["request-id"]
                       << "', falling back to auto-generated id: " << e.what();
    }
  }
  req_id = base_id;
  for (size_t i = 0; i < output_group.size(); ++i) {
    auto& [output, req_logprobs, total_score] = output_group[i];
    output = ksana_python_input->input_tokens;
    req_ids.push_back(base_id | static_cast<int64_t>(i));
  }
  if (const auto it = request_target.find("logits"); it != request_target.end()) {
    for (auto [l, r] : it->second.slice_pos) {
      logits_custom_length += (r - l + 1);
    }
    input_top_logprobs_num = it->second.input_top_logprobs_num;
  }

  if (req_ctx) {
    if (req_ctx->count("kv-comm-group-key")) {
      kv_comm_group_key = (*req_ctx)["kv-comm-group-key"];
    }
    if (req_ctx->count("prefix_cache_hit_tokens")) {
      try {
        prefix_cache_hit_tokens = std::stoi(req_ctx->at("prefix_cache_hit_tokens"));
      } catch (const std::exception& e) {
        KLLM_LOG_WARNING << "Request " << req_id << " invalid prefix_cache_hit_tokens: " << e.what();
        prefix_cache_hit_tokens = 0;
      }
    }
  }

  ApplyStructuredGeneratorConfig(this);
}

KsanaPythonOutput::KsanaPythonOutput(std::shared_ptr<Request> req) {
  finish_status = req->finish_status;
  input_tokens = req->input_tokens;
  // Prefix-cache hit comes from the Request member (set by the scheduler; works regardless of
  // req_ctx, so pd_v2 Prefill reports it too). The constructor seeds the member from any
  // client-provided req_ctx header, so that path still flows through here.
  prefix_cache_hit_tokens = req->prefix_cache_hit_tokens;
  draft_hit_tokens = 0;
  draft_tokens = 0;
  if (req->req_ctx) {
    if (req->req_ctx->count("mtp_hit_tokens")) {
      draft_hit_tokens = std::stoi(req->req_ctx->at("mtp_hit_tokens"));
    }
    if (req->req_ctx->count("mtp_draft_tokens")) {
      draft_tokens = std::stoi(req->req_ctx->at("mtp_draft_tokens"));
    }
  }
  for (const auto& [output, req_logprobs, total_score] : req->output_group) {
    std::vector<int> req_output = {output.begin() + input_tokens.size(), output.end()};
    output_tokens.emplace_back(req_output);
    if (req->sampling_config.logprobs_num > 0 || req->input_top_logprobs_num > 0) {
      logprobs.emplace_back(req_logprobs);
    }
  }
  if (req->return_cache_stat) {
    cache_stat = req->cache_stat;
  }
  response = std::move(req->response);

  // Access-log timestamps snapshot.
  arrive_time_us = req->timestamp_in_us;
  scheduled_time_us = req->scheduled_time_us.load();
  first_token_time_us = req->first_token_time_us.load();
  finish_time_us = req->finish_time_us.load();
  pd_v2_blocks_allocated_time_us = req->pd_v2_blocks_allocated_time_us.load();
  pd_v2_prefill_dispatched_time_us = req->pd_v2_prefill_dispatched_time_us.load();
  pd_v2_prefill_complete_time_us = req->pd_v2_prefill_complete_time_us.load();
  pd_v2_pf_prefix_cache_hit_tokens = req->pd_v2_pf_prefix_cache_hit_tokens.load();
  pd_v2_pf_queue_us = req->pd_v2_pf_queue_us.load();
  pd_v2_pf_compute_us = req->pd_v2_pf_compute_us.load();
  pd_v2_pf_send_us = req->pd_v2_pf_send_us.load();
  if (req->req_ctx && req->req_ctx->count("pd_v2_prefill_peer")) {
    pd_v2_prefill_peer = req->req_ctx->at("pd_v2_prefill_peer");
  }
}

Status KsanaPythonInput::VerifyRequestTarget() {
  /**
   * Verify each target specified by 'request_target' in this KsanaPythonInput object.
   *
   * This function iterates through each target description of a request, checks its validity, and throws a
   * std::runtime_error if:
   *   1. 'target_name' is missing.
   *   2. 'slice_pos' does not represent valid ordered intervals.
   *   3. both 'token_id' and 'slice_pos' are specified for the same target.
   *   4. 'token_reduce_mode" is invalid.
   *   5. GATHER_TOKEN_ID is specified for a transformer or layernorm target.
   *   6. 'token_ids' or the last logits is specified in a logits target.
   */

  // Iterate through each target
  for (auto& [target_name, target_desc] : request_target) {
    // Ensure 'target_name' is specified
    if (target_name.empty()) {
      KLLM_THROW("Missing 'target_name' in target description.");
    }
    // 'target_name' should be 'transformer', 'layernorm' or 'logits'
    if (!std::unordered_set<std::string>{"transformer", "layernorm", "logits", "lm_head"}.count(target_name)) {
      KLLM_THROW(fmt::format("Invalid target name {}.", target_name));
    }

    const int input_tokens_num = static_cast<int>(input_tokens.size());

    // Validate 'slice_pos' is a valid ordered intervals if specified
    if (!target_desc.slice_pos.empty()) {
      int min_required_begin = 0;
      for (auto& [slice_begin, slice_end] : target_desc.slice_pos) {
        // We allow negative indices to count from the end
        slice_begin = slice_begin < 0 ? slice_begin + input_tokens_num : slice_begin;
        slice_end = slice_end < 0 ? slice_end + input_tokens_num : slice_end;
        // Check if the end position is greater than or equal to the begin position
        if (slice_end < slice_begin) {
          KLLM_THROW(fmt::format("Error: The end position of interval [{}, {}] is less than its start position.",
                                 slice_begin, slice_end));
        }
        // Validate that the end position does not exceed the number of input tokens
        if (slice_end >= input_tokens_num) {
          KLLM_THROW(
              fmt::format("Error: The end position of interval [{}, {}] exceeds the total number of input tokens ({}).",
                          slice_begin, slice_end, input_tokens_num));
        }
        // Check for overlap with the previous interval
        if (slice_begin < min_required_begin) {
          KLLM_THROW(fmt::format("Error: Interval [{}, {}] overlaps with the previous interval ending at position {}.",
                                 slice_begin, slice_end, min_required_begin - 1));
        }
        min_required_begin = slice_end + 1;
      }
    }

    // Ensure that 'token_id' and 'slice_pos' are not both set for the same target
    if (!target_desc.token_id.empty() && !target_desc.slice_pos.empty()) {
      KLLM_THROW("Unable to set both token_id and slice_pos at the same time.");
    }

    // Validate the token reduce mode
    if (target_desc.token_reduce_mode == TokenReduceMode::INVALID_TYPE) {
      KLLM_THROW(fmt::format("The specified token reduce mode in {} is invalid.", target_name));
    }
    if (target_desc.token_reduce_mode == TokenReduceMode::GATHER_TOKEN_ID) {
      // Ensure GATHER_TOKEN_ID is not used with transformer, layernorm, lm_head targets
      if (target_name == "transformer" || target_name == "layernorm" || target_name == "lm_head") {
        KLLM_THROW(fmt::format("The output of the {} does not support 'GATHER_TOKEN_ID'.", target_name));
      }
    }

    // TODO(zakwang): Enhance support for additional request parameters
    if (target_name == "logits") {
      // Verify that no token IDs are specified, as they are not supported for logits output.
      if (!target_desc.token_id.empty()) {
        KLLM_THROW(fmt::format("Specifying token_id for {} output is not supported.", target_name));
      }

      if (target_desc.input_top_logprobs_num < 0) {
        KLLM_THROW(fmt::format("Specifying input_top_logprobs_num for {} output is not supported. it should be >= 0.",
                               target_name));
      }

      if (target_desc.input_top_logprobs_num > 0 && target_desc.token_reduce_mode == TokenReduceMode::GATHER_ALL) {
        KLLM_THROW(
            fmt::format("Specifying return input_top_logprobs_num > 0 for {} output with GATHER_ALL token_reduce_mode "
                        "is not supported.",
                        target_name));
      }
    }
  }

  return Status();
}

}  // namespace ksana_llm
