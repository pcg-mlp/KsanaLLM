/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/torch_op/serving_op.h"

#include <iostream>
#include <memory>
#include <string>

#include "ksana_llm/endpoints/endpoint_factory.h"
#include "ksana_llm/endpoints/streaming/streaming_iterator.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"

#include "ksana_llm/utils/request.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

namespace ksana_llm {

ServingOp::ServingOp() {}

ServingOp::~ServingOp() { serving_impl_->Stop(); }

void ServingOp::InitServing(const std::string &config_file) {
  NLLM_LOG_DEBUG << "ServingOp::InitServing invoked.";

  InitLoguru();
  NLLM_LOG_INFO << "Log INFO level: " << GetLevelName(GetLogLevel());

  NLLM_LOG_INFO << "InitServing with config file: " << config_file;
  Status status = Singleton<Environment>::GetInstance()->ParseConfig(config_file);
  if (!status.OK()) {
    std::cerr << status.ToString() << std::endl;
    NLLM_LOG_FATAL << "InitServing error, " << status.ToString();
  }

  serving_impl_ = std::make_shared<ServingImpl>();
  serving_impl_->Start();
  NLLM_LOG_DEBUG << "ServingOp::InitServing finished.";
}

Status ServingOp::Generate(const std::string &model_name, const std::vector<int> &input_tokens,
                           const SamplingConfig &sampling_config, std::vector<int> &output_tokens) {
  NLLM_LOG_DEBUG << "ServingOp::Generate invoked.";
  return serving_impl_->Handle(model_name, input_tokens, sampling_config, output_tokens);
}

Status ServingOp::GenerateStreaming(const std::string &model_name, const std::vector<int> &input_tokens,
                                    const SamplingConfig &sampling_config,
                                    std::shared_ptr<StreamingIterator> &streaming_iterator) {
  NLLM_LOG_DEBUG << "ServingOp::GenerateStreaming invoked.";
  return serving_impl_->HandleStreaming(model_name, input_tokens, sampling_config, streaming_iterator);
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

  // Export `RetCode` to python, only export the values used in python.
  pybind11::enum_<ksana_llm::RetCode>(m, "RetCode", pybind11::arithmetic())
      .value("RET_SUCCESS", ksana_llm::RetCode::RET_SUCCESS)
      .value("RET_STOP_ITERATION", ksana_llm::RetCode::RET_STOP_ITERATION)
      .export_values();

  // Export `SamplingConfig` to python.
  pybind11::class_<ksana_llm::SamplingConfig, std::shared_ptr<ksana_llm::SamplingConfig>>(m, "SamplingConfig")
      .def(pybind11::init<>())
      .def_readwrite("beam_width", &ksana_llm::SamplingConfig::beam_width)
      .def_readwrite("topk", &ksana_llm::SamplingConfig::topk)
      .def_readwrite("topp", &ksana_llm::SamplingConfig::topp)
      .def_readwrite("temperature", &ksana_llm::SamplingConfig::temperature)
      .def_readwrite("max_new_tokens", &ksana_llm::SamplingConfig::max_new_tokens)
      .def_readwrite("repetition_penalty", &ksana_llm::SamplingConfig::repetition_penalty);

  // Export `StreamingIterator` to python.
  pybind11::class_<ksana_llm::StreamingIterator, std::shared_ptr<ksana_llm::StreamingIterator>>(m, "StreamingIterator")
      .def(pybind11::init<>())
      .def("GetNext", [](std::shared_ptr<ksana_llm::StreamingIterator> &self) {
        pybind11::gil_scoped_release release;
        int token_id;
        ksana_llm::Status status = self->GetNext(token_id);
        pybind11::gil_scoped_acquire acquire;
        return std::make_tuple(status, token_id);
      });

  // Export `ServingOp` to python.
  pybind11::class_<ksana_llm::ServingOp, std::shared_ptr<ksana_llm::ServingOp>>(m, "Serving")
      .def(pybind11::init<>())
      .def("init_serving", &ksana_llm::ServingOp::InitServing)
      .def("generate",
           [](std::shared_ptr<ksana_llm::ServingOp> &self, const std::string &model_name,
              const std::vector<int> &input_tokens, const ksana_llm::SamplingConfig &sampling_config) {
             pybind11::gil_scoped_release release;
             std::vector<int> output_tokens;
             ksana_llm::Status status = self->Generate(model_name, input_tokens, sampling_config, output_tokens);
             pybind11::gil_scoped_acquire acquire;
             return std::make_tuple(status, output_tokens);
           })
      .def("generate_streaming",
           [](std::shared_ptr<ksana_llm::ServingOp> &self, const std::string &model_name,
              const std::vector<int> &input_tokens, const ksana_llm::SamplingConfig &sampling_config) {
             pybind11::gil_scoped_release release;
             std::shared_ptr<ksana_llm::StreamingIterator> streaming_iterator;
             ksana_llm::Status status =
                 self->GenerateStreaming(model_name, input_tokens, sampling_config, streaming_iterator);
             pybind11::gil_scoped_acquire acquire;
             return std::make_tuple(status, streaming_iterator);
           });
}
