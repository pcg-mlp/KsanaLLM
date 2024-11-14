/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include "ksana_llm/utils/request.h"

namespace py = pybind11;

namespace ksana_llm {

// Wraps the tokenizer for internal various usage
class Tokenizer {
 public:
  // Initialize the tokenizer from the given tokenizer_path.
  Status InitTokenizer(const std::string& tokenizer_path);

  // Decode the given input token ids into string
  Status Decode(std::vector<int>& input_tokens, std::string& output);

  // Encode the given prompt into token ids
  Status Encode(const std::string& prompt, std::vector<int>& input_tokens);
 public:
  py::object tokenizer;
};

}  // namespace ksana_llm
