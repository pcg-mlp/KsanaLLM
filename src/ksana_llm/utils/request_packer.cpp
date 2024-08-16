/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/

#include "ksana_llm/utils/request_packer.h"

#include "base64.hpp"
#include "ksana_llm/utils/request_serial.h"

namespace ksana_llm {

void RequestPacker::InitTokenizer(const std::string& tokenizer_path) {
  py::module transformers = py::module::import("transformers");
  py::object auto_tokenizer = transformers.attr("AutoTokenizer");
  tokenizer_ = auto_tokenizer.attr("from_pretrained")(tokenizer_path, py::arg("trust_remote_code") = true);
}

Status RequestPacker::Unpack(const std::string& request_bytes,
                             std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs) {
  // Construct a KsanaPythonInput object from a RequestSerial object.
  auto GetKsanaPythonInput = [this](const RequestSerial& req) -> KsanaPythonInput {
    KsanaPythonInput ksana_python_input;
    if (req.input_tokens.empty()) {  // If input tokens are empty, tokenize the prompt.
      Tokenize(req.prompt, ksana_python_input.input_tokens);
    } else {
      ksana_python_input.input_tokens = req.input_tokens;
    }
    ksana_python_input.input_refit_embedding.pos = req.input_refit_embedding.pos;
    ksana_python_input.input_refit_embedding.embeddings = req.input_refit_embedding.embeddings;
    for (const auto& [target_name, token_id, slice_pos, token_reduce_mode] : req.request_target) {
      ksana_python_input.request_target.emplace(
          target_name, TargetDescribe{token_id, slice_pos, GetTokenReduceMode(token_reduce_mode)});
    }
    // Verify the request target and throw an exception if anything is invalid.
    ksana_python_input.VerifyRequestTarget();
    // For forward interface.
    ksana_python_input.sampling_config.max_new_tokens = 1;
    return ksana_python_input;
  };

  // Try unpack the request bytes and parse into a batch of KsanaPythonInput objects.
  try {
    auto handle = msgpack::unpack(request_bytes.data(), request_bytes.size());
    auto object = handle.get();
    auto batch_req = object.as<BatchRequestSerial>();
    ksana_python_inputs.clear();
    ksana_python_inputs.reserve(batch_req.requests.size());
    for (auto& req : batch_req.requests) {
      ksana_python_inputs.push_back(std::make_shared<KsanaPythonInput>(GetKsanaPythonInput(req)));
    }
    return Status();
  } catch (const msgpack::unpack_error& e) {
    return Status(RET_INVALID_ARGUMENT, "Failed to unpack the request bytes.");
  } catch (const msgpack::type_error& e) {
    return Status(RET_INVALID_ARGUMENT, "Failed to parse the request.");
  } catch (const py::error_already_set& e) {
    PyErr_Clear();
    return Status(RET_INVALID_ARGUMENT, fmt::format("Failed to decode the input prompt."));
  } catch (const std::runtime_error& e) {  // The request specifies invalid target.
    return Status(RET_INVALID_ARGUMENT, e.what());
  } catch (...) {
    return Status(RET_INVALID_ARGUMENT, "Unknown error occurred during request unpack.");
  }
}

Status RequestPacker::Pack(const std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs,
                           const std::vector<KsanaPythonOutput>& ksana_python_outputs, std::string& response_bytes) {
  // Construct a ResponseSerial object from a KsanaPythonInput and a KsanaPythonOutput object.
  auto GetResponseSerial = [](const std::shared_ptr<KsanaPythonInput>& ksana_python_input,
                              const KsanaPythonOutput& ksana_python_output) -> ResponseSerial {
    ResponseSerial rsp;
    rsp.input_token_ids = ksana_python_input->input_tokens;
    for (const auto& [target_name, tensor] : ksana_python_output.response) {
      rsp.response.push_back(TargetResponseSerial{
          target_name,
          PythonTensorSerial{base64::encode_into<std::vector<uint8_t>>(tensor.data.begin(), tensor.data.end()),
                             tensor.shape, tensor.dtype}});
    }
    return rsp;
  };

  // Convert the batch of KsanaPythonOutput objects into BatchResponseSerial objects and pack to response bytes.
  msgpack::sbuffer sbuf;
  BatchResponseSerial batch_rsp;
  const size_t batch_size = ksana_python_outputs.size();
  batch_rsp.responses.reserve(batch_size);
  for (size_t i = 0; i < batch_size; i++) {
    batch_rsp.responses.push_back(GetResponseSerial(ksana_python_inputs[i], ksana_python_outputs[i]));
  }
  msgpack::pack(sbuf, batch_rsp);

  response_bytes.assign(sbuf.data(), sbuf.size());
  return Status();
}

Status RequestPacker::Tokenize(const std::string& prompt, std::vector<int>& input_tokens) {
  pybind11::gil_scoped_acquire acquire;
  py::object tokens = tokenizer_.attr("encode")(prompt, py::arg("add_special_tokens") = true);
  input_tokens = tokens.cast<std::vector<int>>();
  pybind11::gil_scoped_release release;
  return Status();
}

}  // namespace ksana_llm
