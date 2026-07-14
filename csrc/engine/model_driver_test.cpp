/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "engine/model_driver.h"

#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/torch.h>

#include <cstring>
#include <memory>
#include <vector>

#include "test.h"
#include "utils/request.h"
#include "utils/tensor.h"

namespace py = pybind11;

namespace ksana_llm {

// Test helper class to access private methods of ModelDriver
class ModelDriverTestHelper {
 public:
  // Expose ConvertMultimodalEmbedding for testing
  static void ConvertMultimodalEmbedding(ModelDriver& driver, std::shared_ptr<Request>& req) {
    driver.ConvertMultimodalEmbedding(req);
  }
};

class ModelDriverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize Python interpreter if not already initialized
    if (!Py_IsInitialized()) {
      Py_Initialize();
    }
  }

  void TearDown() override {}

  // Import Python torch module and convert C++ tensor to Python object
  // This is only needed for tests that use py::object embeddings
  py::object tensor_to_pyobject(const torch::Tensor& tensor) {
    // Import torch module to ensure THPVariable_Wrap works correctly
    py::module::import("torch");
    PyObject* obj = THPVariable_Wrap(tensor);
    return py::reinterpret_steal<py::object>(obj);
  }
};

// Test case 1: embeddings, embedding_pyobjects, additional_pyobjects are all empty
TEST_F(ModelDriverTest, ConvertMultimodalEmbeddingAllEmpty) {
  Channel<std::pair<Status, std::shared_ptr<Request>>> request_queue;
  auto schedule_output_pool = std::make_shared<ScheduleOutputPool>();
  auto sampling_output_pool = std::make_shared<SamplingOutputPool>();
  auto context = std::make_shared<Context>();

  ModelDriver driver(request_queue, schedule_output_pool, sampling_output_pool, context);

  // Create request without any embeddings
  auto ksana_input = std::make_shared<KsanaPythonInput>();
  ksana_input->input_tokens = {1, 2, 3, 4, 5};
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto req = std::make_shared<Request>(ksana_input, req_ctx);

  // Call ConvertMultimodalEmbedding
  ModelDriverTestHelper::ConvertMultimodalEmbedding(driver, req);

  // Verify that embedding_tensors and additional_tensors are empty
  EXPECT_TRUE(req->input_refit_embedding.embedding_tensors.empty());
  EXPECT_TRUE(req->input_refit_embedding.additional_tensors.empty());
}

// Test case 2: embeddings is not empty, embedding_pyobjects and additional_pyobjects are empty
TEST_F(ModelDriverTest, ConvertMultimodalEmbeddingWithVectorFloat) {
  Channel<std::pair<Status, std::shared_ptr<Request>>> request_queue;
  auto schedule_output_pool = std::make_shared<ScheduleOutputPool>();
  auto sampling_output_pool = std::make_shared<SamplingOutputPool>();
  auto context = std::make_shared<Context>();

  ModelDriver driver(request_queue, schedule_output_pool, sampling_output_pool, context);

  // Create embeddings data
  std::vector<std::vector<float>> embeddings = {
      {1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}};

  // Create request with embeddings
  auto ksana_input = std::make_shared<KsanaPythonInput>();
  ksana_input->input_tokens = {1, 2, 3, 4, 5};
  ksana_input->input_refit_embedding.embeddings = embeddings;
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto req = std::make_shared<Request>(ksana_input, req_ctx);

  // Call ConvertMultimodalEmbedding
  ModelDriverTestHelper::ConvertMultimodalEmbedding(driver, req);

  // Verify that embedding_tensors has the correct size
  EXPECT_EQ(req->input_refit_embedding.embedding_tensors.size(), embeddings.size());
  EXPECT_TRUE(req->input_refit_embedding.additional_tensors.empty());

  // Verify each tensor
  for (size_t i = 0; i < embeddings.size(); i++) {
    const Tensor& tensor = req->input_refit_embedding.embedding_tensors[i];

    // Check tensor properties
    EXPECT_EQ(tensor.location, MemoryLocation::LOCATION_HOST);
    EXPECT_EQ(tensor.dtype, DataType::TYPE_FP32);
    EXPECT_EQ(tensor.shape.size(), 1);
    EXPECT_EQ(tensor.shape[0], embeddings[i].size());

    // Check tensor data
    const float* tensor_data = tensor.GetPtr<float>();
    for (size_t j = 0; j < embeddings[i].size(); j++) {
      EXPECT_FLOAT_EQ(tensor_data[j], embeddings[i][j]);
    }
  }
}

// Test case 3: embeddings is empty, embedding_pyobjects and additional_pyobjects are both not empty
TEST_F(ModelDriverTest, ConvertMultimodalEmbeddingWithPyObjects) {
  Channel<std::pair<Status, std::shared_ptr<Request>>> request_queue;
  auto schedule_output_pool = std::make_shared<ScheduleOutputPool>();
  auto sampling_output_pool = std::make_shared<SamplingOutputPool>();
  auto context = std::make_shared<Context>();

  ModelDriver driver(request_queue, schedule_output_pool, sampling_output_pool, context);

  // Create torch::Tensor embeddings
  std::vector<torch::Tensor> torch_embeddings;
  torch_embeddings.push_back(torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, torch::kFloat32));
  torch_embeddings.push_back(torch::tensor({5.0f, 6.0f, 7.0f, 8.0f}, torch::kFloat32));

  // Create additional tensors
  std::vector<torch::Tensor> torch_additional;
  torch_additional.push_back(torch::tensor({0, 1, 2, 3}, torch::kInt64));
  torch_additional.push_back(torch::tensor({100}, torch::kInt64));

  // Create request with py::object embeddings and additional tensors
  auto ksana_input = std::make_shared<KsanaPythonInput>();
  ksana_input->input_tokens = {1, 2, 3, 4, 5};

  {
    py::gil_scoped_acquire acquire;
    for (const auto& tensor : torch_embeddings) {
      ksana_input->input_refit_embedding.embedding_pyobjects.push_back(tensor_to_pyobject(tensor));
    }
    for (const auto& tensor : torch_additional) {
      ksana_input->input_refit_embedding.additional_pyobjects.push_back(tensor_to_pyobject(tensor));
    }
  }

  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto req = std::make_shared<Request>(ksana_input, req_ctx);

  // Call ConvertMultimodalEmbedding
  ModelDriverTestHelper::ConvertMultimodalEmbedding(driver, req);

  // Verify that embedding_tensors has the correct size
  EXPECT_EQ(req->input_refit_embedding.embedding_tensors.size(), torch_embeddings.size());

  // Verify that additional_tensors has the correct size
  EXPECT_EQ(req->input_refit_embedding.additional_tensors.size(), torch_additional.size());

  // Verify embedding tensors
  for (size_t i = 0; i < torch_embeddings.size(); i++) {
    const Tensor& tensor = req->input_refit_embedding.embedding_tensors[i];
    EXPECT_EQ(tensor.location, MemoryLocation::LOCATION_HOST);
    EXPECT_EQ(tensor.dtype, DataType::TYPE_FP32);
    EXPECT_EQ(tensor.GetElementNumber(), static_cast<size_t>(torch_embeddings[i].numel()));

    // Check tensor data
    const float* tensor_data = tensor.GetPtr<float>();
    const float* expected_data = torch_embeddings[i].data_ptr<float>();
    for (size_t j = 0; j < tensor.GetElementNumber(); j++) {
      EXPECT_FLOAT_EQ(tensor_data[j], expected_data[j]);
    }
  }

  // Verify additional tensors
  for (size_t i = 0; i < torch_additional.size(); i++) {
    const Tensor& tensor = req->input_refit_embedding.additional_tensors[i];
    EXPECT_EQ(tensor.location, MemoryLocation::LOCATION_HOST);
    EXPECT_EQ(tensor.dtype, DataType::TYPE_INT64);
    EXPECT_EQ(tensor.GetElementNumber(), static_cast<size_t>(torch_additional[i].numel()));

    // Check tensor data
    const int64_t* tensor_data = tensor.GetPtr<int64_t>();
    const int64_t* expected_data = torch_additional[i].data_ptr<int64_t>();
    for (size_t j = 0; j < tensor.GetElementNumber(); j++) {
      EXPECT_EQ(tensor_data[j], expected_data[j]);
    }
  }
}

// Test case 4: embedding_bytes is not empty
TEST_F(ModelDriverTest, ConvertMultimodalEmbeddingWithBytes) {
  Channel<std::pair<Status, std::shared_ptr<Request>>> request_queue;
  auto schedule_output_pool = std::make_shared<ScheduleOutputPool>();
  auto sampling_output_pool = std::make_shared<SamplingOutputPool>();
  auto context = std::make_shared<Context>();

  ModelDriver driver(request_queue, schedule_output_pool, sampling_output_pool, context);

  // Create embeddings data
  std::vector<std::vector<float>> embeddings = {
      {1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f}};
  std::vector<std::vector<uint8_t>> embedding_bytes(embeddings.size());
  for (size_t i = 0; i < embeddings.size(); i++) {
    embedding_bytes[i].resize(embeddings[i].size() * sizeof(float) / sizeof(uint8_t));
    std::memcpy(embedding_bytes[i].data(), embeddings[i].data(), embedding_bytes[i].size());
  }

  // Create request with embedding_bytes
  auto ksana_input = std::make_shared<KsanaPythonInput>();
  ksana_input->input_tokens = {1, 2, 3, 4, 5};
  ksana_input->input_refit_embedding.embedding_bytes = embedding_bytes;
  ksana_input->input_refit_embedding.embedding_dtype = "fp32";
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto req = std::make_shared<Request>(ksana_input, req_ctx);

  // Call ConvertMultimodalEmbedding
  ModelDriverTestHelper::ConvertMultimodalEmbedding(driver, req);

  // Verify that embedding_tensors has the correct size
  EXPECT_EQ(req->input_refit_embedding.embedding_tensors.size(), embedding_bytes.size());
  EXPECT_TRUE(req->input_refit_embedding.additional_tensors.empty());

  // Verify each tensor
  for (size_t i = 0; i < embedding_bytes.size(); i++) {
    const Tensor& tensor = req->input_refit_embedding.embedding_tensors[i];

    // Check tensor properties
    EXPECT_EQ(tensor.location, MemoryLocation::LOCATION_HOST);
    EXPECT_EQ(tensor.dtype, DataType::TYPE_FP32);
    EXPECT_EQ(tensor.shape.size(), 1);
    EXPECT_EQ(tensor.shape[0], embeddings[i].size());

    // Check tensor data
    const float* tensor_data = tensor.GetPtr<float>();
    for (size_t j = 0; j < embeddings[i].size(); j++) {
      EXPECT_FLOAT_EQ(tensor_data[j], embeddings[i][j]);
    }
  }
}

}  // namespace ksana_llm
