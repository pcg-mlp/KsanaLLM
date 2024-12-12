/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/status.h"

namespace py = pybind11;

namespace ksana_llm {

// RequestPacker is responsible for packing and unpacking requests and responses serialized in msgpack format
// bytes into corresponding KsanaPythonInput and KsanaPythonOutput objects.
class RequestPacker {
 public:
  // Initialize the tokenizer from the given tokenizer_path.
  void InitTokenizer(const std::string& tokenizer_path);

  // Destroy the tokenier
  void DestroyTokenizer();

  // Unpack a serialized request into KsanaPythonInput objects.
  Status Unpack(const std::string& request_bytes, std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs);

  // Pack KsanaPythonOutput objects into a serialized response.
  Status Pack(const std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs,
              const std::vector<KsanaPythonOutput>& ksana_python_outputs, const Status& response_status,
              std::string& response_bytes);

  // Detokenize the giver token_list into prompt.
  Status DeTokenize(const std::vector<int>& input_tokens, std::string& prompt);

  // Tokenize the given prompt into input tokens.
  Status Tokenize(const std::string& prompt, std::vector<int>& input_tokens, bool add_special_tokens = true);

 private:
  py::object tokenizer_;
};

}  // namespace ksana_llm
