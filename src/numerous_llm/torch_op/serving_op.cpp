/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/torch_op/serving_op.h"
#include <memory>
#include <string>

#include "numerous_llm/endpoints/endpoint_factory.h"
#include "numerous_llm/endpoints/streaming/streaming_iterator.h"
#include "numerous_llm/utils/logger.h"

#include "numerous_llm/utils/request.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

namespace numerous_llm {

ServingOp::ServingOp() {}

ServingOp::~ServingOp() { serving_impl_->Stop(); }

void ServingOp::InitServing(const std::string &model_dir) {
  NLLM_LOG_INFO << "ServingOp::InitServing invoked.";

  std::string config_file = model_dir + "/config.ini";
  Status status = Singleton<Environment>::GetInstance()->ParseConfig(config_file);
  if (!status.OK()) {
    std::cerr << status.ToString() << std::endl;
  }

  serving_impl_ = std::make_shared<ServingImpl>();
  serving_impl_->Start();
  NLLM_LOG_INFO << "ServingOp::InitServing finished.";
}

Status ServingOp::Generate(const std::string &model_name, const std::vector<int> &input_tokens,
                           const SamplingConfig &sampling_config, std::vector<int> &output_tokens) {
  NLLM_LOG_INFO << "ServingOp::Generate invoked.";
  return serving_impl_->Handle(model_name, input_tokens, sampling_config, output_tokens);
}

Status ServingOp::GenerateStreaming(const std::string &model_name, const std::vector<int> &input_tokens,
                                    const SamplingConfig &sampling_config,
                                    std::shared_ptr<StreamingIterator> &streaming_iterator) {
  NLLM_LOG_INFO << "ServingOp::GenerateStreaming invoked.";
  return serving_impl_->HandleStreaming(model_name, input_tokens, sampling_config, streaming_iterator);
}

}  // namespace numerous_llm

PYBIND11_MODULE(libtorch_serving, m) {
  // Export `Status` to python.
  pybind11::class_<numerous_llm::Status, std::shared_ptr<numerous_llm::Status>>(m, "Status")
      .def(pybind11::init())
      .def(pybind11::init<numerous_llm::RetCode, const std::string &>())
      .def("OK", &numerous_llm::Status::OK)
      .def("GetMessage", &numerous_llm::Status::GetMessage)
      .def("GetCode", &numerous_llm::Status::GetCode)
      .def("ToString", &numerous_llm::Status::ToString);

  // Export `RetCode` to python, only export the values used in python.
  pybind11::enum_<numerous_llm::RetCode>(m, "RetCode", pybind11::arithmetic())
      .value("RET_SUCCESS", numerous_llm::RetCode::RET_SUCCESS)
      .value("RET_STOP_ITERATION", numerous_llm::RetCode::RET_STOP_ITERATION)
      .export_values();

  // Export `SamplingConfig` to python.
  pybind11::class_<numerous_llm::SamplingConfig, std::shared_ptr<numerous_llm::SamplingConfig>>(m, "SamplingConfig")
      .def(pybind11::init<>())
      .def_readwrite("beam_width", &numerous_llm::SamplingConfig::beam_width)
      .def_readwrite("topk", &numerous_llm::SamplingConfig::topk)
      .def_readwrite("topp", &numerous_llm::SamplingConfig::topp)
      .def_readwrite("temperature", &numerous_llm::SamplingConfig::temperature);

  // Export `StreamingIterator` to python.
  pybind11::class_<numerous_llm::StreamingIterator, std::shared_ptr<numerous_llm::StreamingIterator>>(
      m, "StreamingIterator")
      .def(pybind11::init<>())
      .def("GetNext", [](std::shared_ptr<numerous_llm::StreamingIterator> &self) {
        int token_id;
        numerous_llm::Status status = self->GetNext(token_id);
        return std::make_tuple(status, token_id);
      });

  // Export `ServingOp` to python.
  pybind11::class_<numerous_llm::ServingOp, std::shared_ptr<numerous_llm::ServingOp>>(m, "Serving")
      .def(pybind11::init<>())
      .def("init_serving", &numerous_llm::ServingOp::InitServing)
      .def("generate",
           [](std::shared_ptr<numerous_llm::ServingOp> &self, const std::string &model_name,
              const std::vector<int> &input_tokens, const numerous_llm::SamplingConfig &sampling_config) {
             pybind11::gil_scoped_release release;
             std::vector<int> output_tokens;
             numerous_llm::Status status = self->Generate(model_name, input_tokens, sampling_config, output_tokens);
             pybind11::gil_scoped_acquire acquire;
             return std::make_tuple(status, output_tokens);
           })
      .def("generate_streaming",
           [](std::shared_ptr<numerous_llm::ServingOp> &self, const std::string &model_name,
              const std::vector<int> &input_tokens, const numerous_llm::SamplingConfig &sampling_config) {
             pybind11::gil_scoped_release release;
             std::shared_ptr<numerous_llm::StreamingIterator> streaming_iterator;
             numerous_llm::Status status =
                 self->GenerateStreaming(model_name, input_tokens, sampling_config, streaming_iterator);
             pybind11::gil_scoped_acquire acquire;
             return std::make_tuple(status, streaming_iterator);
           });
}
