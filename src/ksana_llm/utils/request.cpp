/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/request.h"

namespace ksana_llm {

IdGenerator Request::id_generator_;

Request::Request(const ksana_llm::KsanaPythonInput &ksana_python_input)
    : output_group(std::max(std::max(ksana_python_input.sampling_config.num_beams,
                                     ksana_python_input.sampling_config.num_return_sequences),
                            1)),
      finisheds(output_group.size(), false),
      finished(finisheds[0]),
      output_tokens(std::get<0>(output_group[0])),
      logprobs(std::get<1>(output_group[0])),
      model_name(ksana_python_input.model_name),
      input_tokens(ksana_python_input.input_tokens),
      request_target(ksana_python_input.request_target),
      sampling_config(ksana_python_input.sampling_config),
      input_refit_embedding(ksana_python_input.input_refit_embedding) {
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
