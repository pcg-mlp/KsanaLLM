/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/

#include "ksana_llm/utils/tokenizer.h"
#include <pybind11/embed.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace ksana_llm {

Status Tokenizer::InitTokenizer(const std::string& tokenizer_path) {
  pybind11::gil_scoped_acquire acquire;
  py::module transformers = py::module::import("transformers");
  py::object auto_tokenizer = transformers.attr("AutoTokenizer");
  tokenizer = auto_tokenizer.attr("from_pretrained")(tokenizer_path, py::arg("trust_remote_code") = true);
  pybind11::gil_scoped_release release;
  return Status();
}

Status Tokenizer::Decode(std::vector<int>& output_tokens, std::string& output) {
  pybind11::gil_scoped_acquire acquire;
  py::object tokens = tokenizer.attr("decode")(output_tokens, py::arg("skip_special_tokens") = true);
  output = tokens.cast<std::string>();
  pybind11::gil_scoped_release release;
  return Status();
}

Status Tokenizer::Encode(const std::string& prompt, std::vector<int>& input_tokens) {
  pybind11::gil_scoped_acquire acquire;
  py::object tokens = tokenizer.attr("encode")(prompt, py::arg("add_special_tokens") = true);
  input_tokens = tokens.cast<std::vector<int>>();
  pybind11::gil_scoped_release release;
  return Status();
}
}  // namespace ksana_llm
