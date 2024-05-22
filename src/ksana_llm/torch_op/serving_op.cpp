/* Copyright 2024 Tencent Inc.  All rights reserved.

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
                           const SamplingConfig &sampling_config, 
                           const std::vector<int> &subinput_pos, const std::vector<std::vector<float>> &subinput_embedding,
                           std::vector<std::vector<int>> &output_tokens,
                           std::vector<std::vector<std::vector<std::pair<int, float>>>> &logprobs) {
  NLLM_LOG_DEBUG << "ServingOp::Generate invoked.";
  return serving_impl_->Handle(model_name, input_tokens, sampling_config, subinput_pos, subinput_embedding, output_tokens, logprobs);
}

Status ServingOp::GenerateStreaming(const std::string &model_name, const std::vector<int> &input_tokens,
                                    const SamplingConfig &sampling_config,
                                    const std::vector<int> &subinput_pos, const std::vector<std::vector<float>> &subinput_embedding,
                                    std::shared_ptr<StreamingIterator> &streaming_iterator) {
  NLLM_LOG_DEBUG << "ServingOp::GenerateStreaming invoked.";
  return serving_impl_->HandleStreaming(model_name, input_tokens, sampling_config, subinput_pos, subinput_embedding, streaming_iterator);
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
    .def_readwrite("topk", &ksana_llm::SamplingConfig::topk)
    .def_readwrite("topp", &ksana_llm::SamplingConfig::topp)
    .def_readwrite("temperature", &ksana_llm::SamplingConfig::temperature)
    .def_readwrite("max_new_tokens", &ksana_llm::SamplingConfig::max_new_tokens)
    .def_readwrite("logprobs_num", &ksana_llm::SamplingConfig::logprobs_num)
    .def_readwrite("repetition_penalty", &ksana_llm::SamplingConfig::repetition_penalty)
    .def_readwrite("num_beams", &ksana_llm::SamplingConfig::num_beams)
    .def_readwrite("num_return_sequences", &ksana_llm::SamplingConfig::num_return_sequences)
    .def_readwrite("length_penalty", &ksana_llm::SamplingConfig::length_penalty)
    .def_readwrite("stop_token_ids", &ksana_llm::SamplingConfig::stop_token_ids);

  // Export `StreamingIterator` to python.
  pybind11::class_<ksana_llm::StreamingIterator, std::shared_ptr<ksana_llm::StreamingIterator>>(m, "StreamingIterator")
    .def(pybind11::init<>())
    .def("GetNext", [](std::shared_ptr<ksana_llm::StreamingIterator> &self) {
      pybind11::gil_scoped_release release;
      std::vector<std::vector<int>> output_tokens;
      std::vector<std::vector<std::vector<std::pair<int, float>>>> logprobs;
      ksana_llm::Status status = self->GetNext(output_tokens, logprobs);
      pybind11::gil_scoped_acquire acquire;
      return std::make_tuple(status, output_tokens, logprobs);
    });

  // Export `ServingOp` to python.
  pybind11::class_<ksana_llm::ServingOp, std::shared_ptr<ksana_llm::ServingOp>>(m, "Serving")
    .def(pybind11::init<>())
    .def("init_serving", &ksana_llm::ServingOp::InitServing)
    .def("generate",
         [](std::shared_ptr<ksana_llm::ServingOp> &self, const std::string &model_name,
            const std::vector<int> &input_tokens, const ksana_llm::SamplingConfig &sampling_config,
            const std::vector<int> &subinput_pos, const std::vector<std::vector<float>> &subinput_embedding) {
           pybind11::gil_scoped_release release;
           std::vector<std::vector<int>> output_tokens;
           std::vector<std::vector<std::vector<std::pair<int, float>>>> logprobs;
           ksana_llm::Status status =
             self->Generate(model_name, input_tokens, sampling_config, subinput_pos, subinput_embedding, output_tokens, logprobs);
           pybind11::gil_scoped_acquire acquire;
           return std::make_tuple(status, output_tokens, logprobs);
         })
    .def("generate_streaming", [](std::shared_ptr<ksana_llm::ServingOp> &self, const std::string &model_name,
                                  const std::vector<int> &input_tokens,
                                  const ksana_llm::SamplingConfig &sampling_config,
                                  const std::vector<int> &subinput_pos, const std::vector<std::vector<float>> &subinput_embedding) {
      pybind11::gil_scoped_release release;
      std::shared_ptr<ksana_llm::StreamingIterator> streaming_iterator;
      ksana_llm::Status status = self->GenerateStreaming(model_name, input_tokens, sampling_config, subinput_pos, subinput_embedding, streaming_iterator);
      pybind11::gil_scoped_acquire acquire;
      return std::make_tuple(status, streaming_iterator);
    });
}
