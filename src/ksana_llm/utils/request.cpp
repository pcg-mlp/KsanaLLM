/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/request.h"

namespace ksana_llm {

IdGenerator Request::id_generator_;

Request::Request(const ksana_llm::KsanaPythonInput &ksana_python_input)
    : req_id(0),
      req_ids(),
      model_name(ksana_python_input.model_name),
      input_tokens(ksana_python_input.input_tokens),
      logits_custom_length(0),
      input_refit_embedding(ksana_python_input.input_refit_embedding),
      output_group(std::max(std::max(ksana_python_input.sampling_config.num_beams,
                                     ksana_python_input.sampling_config.num_return_sequences),
                            1)),
      beam_search_group(),
      output_tokens(std::get<0>(output_group[0])),
      padded_size(0),
      logprobs(std::get<1>(output_group[0])),
      sampling_config(ksana_python_input.sampling_config),
      waiter(nullptr),
      step_waiter(nullptr),
      finisheds(output_group.size(), false),
      finished(finisheds[0]),
      finish_status(),
      output_mutex(),
      request_target(ksana_python_input.request_target),
      response() {
  for (auto output : output_group) {
    req_ids.push_back(id_generator_.Gen());
  }
  req_id = req_ids[0];
  auto it = request_target.find("logits");
  if (it != request_target.end()) {
    for (auto [l, r] : it->second.slice_pos) {
      logits_custom_length += (r - l + 1);
    }
  }
}

}  // namespace ksana_llm
