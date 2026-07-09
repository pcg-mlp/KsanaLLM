/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "engine/model_driver.h"

#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/torch.h>

#include <cstring>
#include <memory>
#include <vector>

#include "batch_scheduler/schedule_processor_interface.h"
#include "pd_v2/public/pd_v2_types.h"
#include "runtime/infer_request.h"
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

  // Expose Enqueue for testing
  static Status Enqueue(ModelDriver& driver, std::shared_ptr<Request>& req) { return driver.Enqueue(req); }
};

// Minimal stub: captures the infer_request_group handed to AddInferRequest so a
// test can inspect the InferRequest objects ModelDriver::Enqueue builds, without
// standing up a real scheduler.
class StubScheduleProcessor : public ScheduleProcessorInterface {
 public:
  void Initialize(std::shared_ptr<BatchSchedulerInterface> /*batch_scheduler*/) override {}
  Status AddInferRequest(std::vector<std::shared_ptr<InferRequest>>& infer_request_group) override {
    last_group_ = infer_request_group;
    return Status();
  }
  std::shared_ptr<ScheduleResult> GetNextScheduleResult() override { return nullptr; }
  void UpdateWithGenerationResult(const GenerationOutputGroup& /*generation_output*/) override {}
  void Lock() override {}
  void Unlock() override {}
  void Start() override {}
  void Stop() override {}

  std::vector<std::shared_ptr<InferRequest>> last_group_;
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

// Enqueue on a pd_v2 prefill Request must copy the pd_v2 decode-target fields
// onto each built InferRequest and register the on_destroy callback.
TEST_F(ModelDriverTest, EnqueuePdV2PrefillCopiesFields) {
  Channel<std::pair<Status, std::shared_ptr<Request>>> request_queue;
  auto schedule_output_pool = std::make_shared<ScheduleOutputPool>();
  auto sampling_output_pool = std::make_shared<SamplingOutputPool>();
  auto context = std::make_shared<Context>();

  ModelDriver driver(request_queue, schedule_output_pool, sampling_output_pool, context);
  auto stub = std::make_shared<StubScheduleProcessor>();
  driver.SetScheduleProcessor(stub);

  auto ksana_input = std::make_shared<KsanaPythonInput>();
  ksana_input->input_tokens = {1, 2, 3};
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto req = std::make_shared<Request>(ksana_input, req_ctx);

  req->is_pd_v2_prefill = true;
  pd_v2::DecodeTargetLite tgt;
  tgt.decode_attn_tp_rank_in_group = 2;
  tgt.kv_pool_base_va = 0xdeadbeefULL;
  req->pd_v2_decode_targets = {tgt};
  req->pd_v2_decode_peer_addr = "1.2.3.4:5000";
  req->pd_v2_decode_attn_dp_group_id = 7;
  auto destroyed = std::make_shared<bool>(false);
  req->pd_v2_on_destroy = [destroyed]() { *destroyed = true; };

  Status s = ModelDriverTestHelper::Enqueue(driver, req);
  EXPECT_TRUE(s.OK());

  ASSERT_EQ(stub->last_group_.size(), 1u);
  std::shared_ptr<InferRequest> ir = stub->last_group_[0];
  ASSERT_EQ(ir->pd_v2.decode_targets.size(), 1u);
  EXPECT_EQ(ir->pd_v2.decode_targets[0].decode_attn_tp_rank_in_group, 2u);
  EXPECT_EQ(ir->pd_v2.decode_targets[0].kv_pool_base_va, 0xdeadbeefULL);
  EXPECT_EQ(ir->pd_v2.decode_engine_inference_addr, "1.2.3.4:5000");
  EXPECT_EQ(ir->pd_v2.decode_attn_dp_group_id, 7u);

  // on_destroy fires once the last reference drops. The InferRequest holds a
  // shared_ptr to itself via req_group (SetReqGroup), so break that cycle first.
  stub->last_group_.clear();
  ir->ClearReqGroup();
  ir.reset();
  EXPECT_TRUE(*destroyed);
}

// Non-pd_v2 Request leaves the pd_v2 fields at their defaults.
TEST_F(ModelDriverTest, EnqueueNonPdV2LeavesFieldsDefault) {
  Channel<std::pair<Status, std::shared_ptr<Request>>> request_queue;
  auto schedule_output_pool = std::make_shared<ScheduleOutputPool>();
  auto sampling_output_pool = std::make_shared<SamplingOutputPool>();
  auto context = std::make_shared<Context>();

  ModelDriver driver(request_queue, schedule_output_pool, sampling_output_pool, context);
  auto stub = std::make_shared<StubScheduleProcessor>();
  driver.SetScheduleProcessor(stub);

  auto ksana_input = std::make_shared<KsanaPythonInput>();
  ksana_input->input_tokens = {1, 2, 3};
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto req = std::make_shared<Request>(ksana_input, req_ctx);
  // is_pd_v2_prefill defaults to false.

  Status s = ModelDriverTestHelper::Enqueue(driver, req);
  EXPECT_TRUE(s.OK());

  ASSERT_EQ(stub->last_group_.size(), 1u);
  std::shared_ptr<InferRequest> ir = stub->last_group_[0];
  EXPECT_TRUE(ir->pd_v2.decode_targets.empty());
  EXPECT_TRUE(ir->pd_v2.decode_engine_inference_addr.empty());
  EXPECT_EQ(ir->pd_v2.decode_attn_dp_group_id, 0u);

  stub->last_group_.clear();
  ir->ClearReqGroup();
  ir.reset();
}

}  // namespace ksana_llm
