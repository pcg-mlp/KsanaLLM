/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/request.h"

namespace ksana_llm {

IdGenerator Request::id_generator_;

Request::Request(const SamplingConfig& sampling_config)
    : output_group(std::max(std::max(sampling_config.num_beams, sampling_config.num_return_sequences), 1)),
      finisheds(output_group.size(), false),
      finished(finisheds[0]),
      output_tokens(std::get<0>(output_group[0])),
      logprobs(std::get<1>(output_group[0])) {
  for (auto output : output_group) {
    req_ids.push_back(id_generator_.Gen());
  }
  req_id = req_ids[0];
}

}  // namespace ksana_llm
