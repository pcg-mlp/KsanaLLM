/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "torch_op/serving_op.h"

#include <unistd.h>

#include <csignal>
#include <iostream>
#include <memory>
#include <string>

#include "endpoints/endpoint_factory.h"
#include "endpoints/streaming/streaming_iterator.h"
#include "profiler/profiler.h"
#include "profiler/reporter.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"
#include "supervisor/service_lifetime.h"
#include "utils/logger.h"
#include "utils/request.h"
#include "utils/service_utils.h"

PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string, ksana_llm::TargetDescribe>);
PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string, ksana_llm::PythonTensor>);
PYBIND11_MAKE_OPAQUE(std::vector<std::pair<int, int>>);

namespace ksana_llm {

void SignalHandler(int signal_num) { GetServiceLifetimeManager()->ShutdownService(); }

ServingOp::ServingOp(const std::string &config_file, bool is_engine, int executor_id) {
  config_file_ = config_file;
  is_engine_ = is_engine;
  executor_id_ = executor_id;

  if (is_engine_) {
    SetProcessLogger("engine", -1);
    KLLM_LOG_INFO << "Engine pid " << getpid() << std::endl;
  } else {
    SetProcessLogger("executor", executor_id_);
    KLLM_LOG_INFO << "Executor " << executor_id_ << " pid " << getpid() << std::endl;
  }

  inference_supervisor_ = std::make_shared<InferenceSupervisor>(config_file_, is_engine_, executor_id_);

  // Create lifetime manager.
  SetServiceLifetimeManager(std::make_shared<KsanaServiceLifetimeManager>(inference_supervisor_));

  // Use TERM signal to shutdown service.
  signal(SIGTERM, SignalHandler);
}

ServingOp::~ServingOp() {
  inference_supervisor_->Destroy();

  // unreferenced inference_server.
  SetServiceLifetimeManager(nullptr);
}

void ServingOp::InitIpc() { inference_supervisor_->InitIpc(); }

void ServingOp::InitServing(int think_end_token_id) {
  STATUS_CHECK_FAILURE(inference_supervisor_->InitService(endpoint_config_, think_end_token_id));
}

void ServingOp::StartExecutor() { STATUS_CHECK_FAILURE(inference_supervisor_->InitExecutor()); }

Status ServingOp::Generate(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                           const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                           ksana_llm::KsanaPythonOutput &ksana_python_output) {
  STATUS_CHECK_AND_REPORT(
      inference_supervisor_->GetService()->Handle(ksana_python_input, req_ctx, ksana_python_output));
}

Status ServingOp::GenerateStreaming(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                                    const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                                    std::shared_ptr<StreamingIterator> &streaming_iterator) {
  STATUS_CHECK_AND_REPORT(
      inference_supervisor_->GetService()->HandleStreaming(ksana_python_input, req_ctx, streaming_iterator));
}

Status ServingOp::Forward(const std::string &request_bytes,
                          const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                          std::string &response_bytes) {
  STATUS_CHECK_AND_REPORT(inference_supervisor_->GetService()->HandleForward(request_bytes, req_ctx, response_bytes));
}

void ServingOp::SetProcessLogger(const std::string &role, int device_id) {
  std::string log_postfix = std::string("_") + role;
  if (device_id >= 0) {
    log_postfix += std::to_string(device_id);
  }

  InitLoguru(false, log_postfix);
  KLLM_LOG_INFO << role << " log levels: " << Vector2Str(GetLogLevels());
}

}  // namespace ksana_llm

PYBIND11_MODULE(libtorch_serving, m) {
  // Export `Status` to python.
  pybind11::class_<ksana_llm::Status, std::shared_ptr<ksana_llm::Status>>(m, "Status")
      .def(pybind11::init())
      .def(pybind11::init<ksana_llm::RetCode, const std::string &>())
      .def("OK", &ksana_llm::Status::OK)
      .def("GetMessage", &ksana_llm::Status::GetMessage)
      .def("GetCode", &ksana_llm::Status::GetCode)
      .def("ToString", &ksana_llm::Status::ToString);

  // Export `RetCode` to python.
  pybind11::enum_<ksana_llm::RetCode>(m, "RetCode", pybind11::arithmetic())
      .value("RET_SUCCESS", ksana_llm::RetCode::RET_SUCCESS)
      .value("RET_INVALID_ARGUMENT", ksana_llm::RetCode::RET_INVALID_ARGUMENT)
      .value("RET_NO_PREDICT_TARGET", ksana_llm::RetCode::RET_NO_PREDICT_TARGET)
      .value("RET_INPUT_LENGTH_EXCEEDED", ksana_llm::RetCode::RET_INPUT_LENGTH_EXCEEDED)
      .value("RET_EXCEED_CAPACITY", ksana_llm::RetCode::RET_EXCEED_CAPACITY)
      .value("RET_REQUEST_TERMINATED", ksana_llm::RetCode::RET_REQUEST_TERMINATED)
      .value("RET_TOKENIZER_FAILED", ksana_llm::RetCode::RET_TOKENIZER_FAILED)
      .value("RET_INFER_FAILED", ksana_llm::RetCode::RET_INFER_FAILED)
      .value("RET_REQUEST_TIMEOUT", ksana_llm::RetCode::RET_REQUEST_TIMEOUT)
      .value("RET_PREDICTOR_DISCARD", ksana_llm::RetCode::RET_PREDICTOR_DISCARD)
      .value("RET_OUT_OF_DEVICE_MEMORY", ksana_llm::RetCode::RET_OUT_OF_DEVICE_MEMORY)
      .value("RET_RUNTIME_FAILED", ksana_llm::RetCode::RET_RUNTIME_FAILED)
      .value("RET_STOP_ITERATION", ksana_llm::RetCode::RET_STOP_ITERATION)
      .export_values();

  // Export `SamplingConfig` to python.
  pybind11::class_<ksana_llm::SamplingConfig, std::shared_ptr<ksana_llm::SamplingConfig>>(m, "SamplingConfig")
      .def(pybind11::init<>())
      .def_readwrite("topk", &ksana_llm::SamplingConfig::topk)
      .def_readwrite("topp", &ksana_llm::SamplingConfig::topp)
      .def_readwrite("temperature", &ksana_llm::SamplingConfig::temperature)
      .def_readwrite("max_new_tokens", &ksana_llm::SamplingConfig::max_new_tokens)
      .def_readwrite("logprobs_num", &ksana_llm::SamplingConfig::logprobs_num)
      .def_readwrite("repetition_penalty", &ksana_llm::SamplingConfig::repetition_penalty)
      .def_readwrite("no_repeat_ngram_size", &ksana_llm::SamplingConfig::no_repeat_ngram_size)
      .def_readwrite("encoder_no_repeat_ngram_size", &ksana_llm::SamplingConfig::encoder_no_repeat_ngram_size)
      .def_readwrite("decoder_no_repeat_ngram_size", &ksana_llm::SamplingConfig::decoder_no_repeat_ngram_size)
      .def_readwrite("num_beams", &ksana_llm::SamplingConfig::num_beams)
      .def_readwrite("num_return_sequences", &ksana_llm::SamplingConfig::num_return_sequences)
      .def_readwrite("length_penalty", &ksana_llm::SamplingConfig::length_penalty)
      .def_readwrite("stop_token_ids", &ksana_llm::SamplingConfig::stop_token_ids)
      .def_readwrite("ignore_eos", &ksana_llm::SamplingConfig::ignore_eos)
      .def_readwrite("stop_strings", &ksana_llm::SamplingConfig::stop_strings)
      .def_readwrite("enable_structured_output", &ksana_llm::SamplingConfig::enable_structured_output)
      .def_readwrite("json_schema", &ksana_llm::SamplingConfig::json_schema)
      .def_readwrite("enable_thinking", &ksana_llm::SamplingConfig::enable_thinking);

  // Export `EmbeddingSlice` to python.
  pybind11::class_<ksana_llm::EmbeddingSlice, std::shared_ptr<ksana_llm::EmbeddingSlice>>(m, "EmbeddingSlice")
      .def(pybind11::init<>())
      .def_readwrite("pos", &ksana_llm::EmbeddingSlice::pos)
      .def_readwrite("embeddings", &ksana_llm::EmbeddingSlice::embeddings)
      .def_readwrite("embedding_bytes", &ksana_llm::EmbeddingSlice::embedding_bytes)
      .def_readwrite("embedding_dtype", &ksana_llm::EmbeddingSlice::embedding_dtype)
      .def_readwrite("embedding_tensors", &ksana_llm::EmbeddingSlice::embedding_pyobjects)
      .def_readwrite("additional_tensors", &ksana_llm::EmbeddingSlice::additional_pyobjects);

  // Export `TokenReduceMode` to python.
  pybind11::enum_<ksana_llm::TokenReduceMode>(m, "TokenReduceMode", pybind11::arithmetic())
      .value("GATHER_ALL", ksana_llm::TokenReduceMode::GATHER_ALL)
      .value("GATHER_TOKEN_ID", ksana_llm::TokenReduceMode::GATHER_TOKEN_ID)
      .export_values();

  // Export `TargetDescribe` to python.
  pybind11::class_<ksana_llm::TargetDescribe, std::shared_ptr<ksana_llm::TargetDescribe>>(m, "TargetDescribe")
      .def(pybind11::init<>())
      .def_readwrite("token_id", &ksana_llm::TargetDescribe::token_id)
      .def_readwrite("slice_pos", &ksana_llm::TargetDescribe::slice_pos)
      .def_readwrite("token_reduce_mode", &ksana_llm::TargetDescribe::token_reduce_mode);

  py::bind_vector<std::vector<std::pair<size_t, size_t>>>(m, "VectorSizeTPair");
  py::bind_map<std::unordered_map<std::string, ksana_llm::TargetDescribe>>(m, "TargetDescribeMap");

  // Export `KsanaPythonInput` to python.
  pybind11::class_<ksana_llm::KsanaPythonInput, std::shared_ptr<ksana_llm::KsanaPythonInput>>(m, "KsanaPythonInput")
      .def(pybind11::init<>())
      .def_readwrite("model_name", &ksana_llm::KsanaPythonInput::model_name)
      .def_readwrite("sampling_config", &ksana_llm::KsanaPythonInput::sampling_config)
      .def_readwrite("return_cache_stat", &ksana_llm::KsanaPythonInput::return_cache_stat)
      .def_readwrite("input_tokens", &ksana_llm::KsanaPythonInput::input_tokens)
      .def_readwrite("request_target", &ksana_llm::KsanaPythonInput::request_target)
      .def_readwrite("input_refit_embedding", &ksana_llm::KsanaPythonInput::input_refit_embedding)
      .def_readwrite("structured_output_regex", &ksana_llm::KsanaPythonInput::structured_output_regex)
      .def_readwrite("draft_tokens", &ksana_llm::KsanaPythonInput::draft_tokens);

  // Export `PythonTensor` to python.
  pybind11::class_<ksana_llm::PythonTensor, std::shared_ptr<ksana_llm::PythonTensor>>(m, "PythonTensor")
      .def(pybind11::init<>())
      .def_readwrite("data", &ksana_llm::PythonTensor::data)
      .def_readwrite("shape", &ksana_llm::PythonTensor::shape)
      .def_readwrite("dtype", &ksana_llm::PythonTensor::dtype);

  py::bind_map<std::unordered_map<std::string, ksana_llm::PythonTensor>>(m, "PythonTensorMap");

  // Export `KsanaPythonOutput` to python.
  pybind11::class_<ksana_llm::KsanaPythonOutput, std::shared_ptr<ksana_llm::KsanaPythonOutput>>(m, "KsanaPythonOutput")
      .def(pybind11::init<>())
      .def_readwrite("input_tokens", &ksana_llm::KsanaPythonOutput::input_tokens)
      .def_readwrite("output_tokens", &ksana_llm::KsanaPythonOutput::output_tokens)
      .def_readwrite("logprobs", &ksana_llm::KsanaPythonOutput::logprobs)
      .def_readwrite("cache_stat", &ksana_llm::KsanaPythonOutput::cache_stat)
      .def_readwrite("response", &ksana_llm::KsanaPythonOutput::response)
      .def_readwrite("embedding", &ksana_llm::KsanaPythonOutput::embedding)
      .def_readwrite("prefix_cache_hit_tokens", &ksana_llm::KsanaPythonOutput::prefix_cache_hit_tokens)
      .def_readwrite("finish_status", &ksana_llm::KsanaPythonOutput::finish_status);

  // Export `StreamingIterator` to python.
  pybind11::class_<ksana_llm::StreamingIterator, std::shared_ptr<ksana_llm::StreamingIterator>>(m, "StreamingIterator")
      .def(pybind11::init<>())
      .def("GetNext", [](std::shared_ptr<ksana_llm::StreamingIterator> &self) {
        pybind11::gil_scoped_release release;
        ksana_llm::KsanaPythonOutput ksana_python_output;
        ksana_llm::Status status = self->GetNext(ksana_python_output);
        pybind11::gil_scoped_acquire acquire;
        return std::make_tuple(status, std::move(ksana_python_output));
      });

  // Export `EndpointType` to python, only export the values used in python.
  pybind11::enum_<ksana_llm::EndpointType>(m, "EndpointType", pybind11::arithmetic())
      .value("RPC", ksana_llm::EndpointType::RPC)
      .export_values();

  // Export `EndpointConfig` to python.
  pybind11::class_<ksana_llm::EndpointConfig, std::shared_ptr<ksana_llm::EndpointConfig>>(m, "EndpointConfig")
      .def(pybind11::init<>())
      .def_readwrite("type", &ksana_llm::EndpointConfig::type)
      .def_readwrite("rpc_plugin_name", &ksana_llm::EndpointConfig::rpc_plugin_name)
      .def_readwrite("host", &ksana_llm::EndpointConfig::host)
      .def_readwrite("port", &ksana_llm::EndpointConfig::port)
      .def_readwrite("access_log", &ksana_llm::EndpointConfig::access_log);

  // Export `ServingOp` to python.
  pybind11::class_<ksana_llm::ServingOp, std::shared_ptr<ksana_llm::ServingOp>>(m, "Serving")
      .def(pybind11::init<std::string, bool, int>())
      .def("init_ipc", &ksana_llm::ServingOp::InitIpc)
      .def("init_serving", &ksana_llm::ServingOp::InitServing, py::arg("think_end_token_id") = -1)
      .def("start_executor", &ksana_llm::ServingOp::StartExecutor)
      .def_readwrite("endpoint_config", &ksana_llm::ServingOp::endpoint_config_)
      .def("generate",
           [](std::shared_ptr<ksana_llm::ServingOp> &self,
              const std::shared_ptr<ksana_llm::KsanaPythonInput> &ksana_python_input,
              std::unordered_map<std::string, std::string> &req_ctx) {
             pybind11::gil_scoped_release release;
             ksana_llm::KsanaPythonOutput ksana_python_output;
             auto req_ctx_ptr = std::make_shared<std::unordered_map<std::string, std::string>>(req_ctx);
             ksana_llm::Status status = self->Generate(ksana_python_input, req_ctx_ptr, ksana_python_output);
             pybind11::gil_scoped_acquire acquire;
             return std::make_tuple(status, std::move(ksana_python_output));
           })
      .def("generate_streaming",
           [](std::shared_ptr<ksana_llm::ServingOp> &self,
              const std::shared_ptr<ksana_llm::KsanaPythonInput> &ksana_python_input,
              std::unordered_map<std::string, std::string> &req_ctx) {
             pybind11::gil_scoped_release release;
             std::shared_ptr<ksana_llm::StreamingIterator> streaming_iterator;
             auto req_ctx_ptr = std::make_shared<std::unordered_map<std::string, std::string>>(req_ctx);
             ksana_llm::Status status = self->GenerateStreaming(ksana_python_input, req_ctx_ptr, streaming_iterator);
             pybind11::gil_scoped_acquire acquire;
             return std::make_tuple(status, streaming_iterator);
           })
      .def("forward", [](std::shared_ptr<ksana_llm::ServingOp> &self, const std::string &request_bytes,
                         std::unordered_map<std::string, std::string> &req_ctx) {
        pybind11::gil_scoped_release release;
        std::string response_bytes;
        auto req_ctx_ptr = std::make_shared<std::unordered_map<std::string, std::string>>(req_ctx);
        ksana_llm::Status status = self->Forward(request_bytes, req_ctx_ptr, response_bytes);
        pybind11::gil_scoped_acquire acquire;
        return std::make_tuple(status, py::bytes(response_bytes));
      });
}
