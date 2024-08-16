/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/

#include "request_packer.h"
#include "base64.hpp"
#include "ksana_llm/utils/device_utils.h"
#include "logger.h"
#include "msgpack.hpp"
#include "request_serial.h"
#include "test.h"

namespace ksana_llm {

class RequestPackerTest : public testing::Test {
 protected:
  void SetUp() override {
    // Any tokenizer works here.
    request_packer_.InitTokenizer("/model/llama-hf/7B");
  }
  void TearDown() override {}

  RequestPacker request_packer_;

  // test input
  msgpack::sbuffer sbuf;
  std::vector<std::shared_ptr<KsanaPythonInput>> ksana_python_inputs;
  std::vector<KsanaPythonOutput> ksana_python_outputs;
  BatchRequestSerial batch_request;
  RequestSerial request;
  TargetRequestSerial target;
};

// Test for simple unpacking.
TEST_F(RequestPackerTest, SimpleUnpack) {
  request.prompt = "hello world";
  target.target_name = "logits";
  target.slice_pos.emplace_back(0, 0);  // [0, 0]
  target.token_reduce_mode = "GATHER_TOKEN_ID";
  request.request_target.push_back(target);
  batch_request.requests.push_back(request);

  msgpack::pack(sbuf, batch_request);

  // Parse request
  std::string request_bytes(sbuf.data(), sbuf.size());
  ASSERT_TRUE(request_packer_.Unpack(request_bytes, ksana_python_inputs).OK());
  ASSERT_EQ(ksana_python_inputs.size(), 1ul);
  ASSERT_EQ(ksana_python_inputs[0]->sampling_config.max_new_tokens, 1);  // forward interface
  ASSERT_EQ(ksana_python_inputs[0]->request_target.size(), 1);
  ASSERT_TRUE(ksana_python_inputs[0]->request_target.count(target.target_name));
  ASSERT_EQ(ksana_python_inputs[0]->request_target[target.target_name].slice_pos, target.slice_pos);
  ASSERT_EQ(ksana_python_inputs[0]->request_target[target.target_name].token_reduce_mode,
            TokenReduceMode::GATHER_TOKEN_ID);
}

// Test for complex unpacking: multiple requests and multiple targets.
TEST_F(RequestPackerTest, ComplexUnpack) {
  request.prompt = "Once upon a time";
  target.target_name = "logits";
  target.slice_pos.emplace_back(0, 1);    // [0, 1]
  target.slice_pos.emplace_back(-2, -2);  // [-2, -2]
  target.token_reduce_mode = "GATHER_TOKEN_ID";
  TargetRequestSerial target2;
  target2.target_name = "layernorm";
  target2.slice_pos.emplace_back(2, 3);  // [2, 3]
  target2.token_reduce_mode = "GATHER_ALL";
  request.request_target.push_back(target);
  request.request_target.push_back(target2);
  batch_request.requests.push_back(request);
  RequestSerial request2 = request;
  request2.prompt.clear();
  request2.input_tokens = std::vector<int>{10, 11, 12, 13, 14};
  request2.request_target[1].target_name = "layernorm";
  request2.request_target[1].slice_pos.clear();
  request2.request_target[1].token_id.push_back(12);
  batch_request.requests.push_back(request2);

  msgpack::pack(sbuf, batch_request);

  // Parse request
  std::string request_bytes(sbuf.data(), sbuf.size());
  ASSERT_TRUE(request_packer_.Unpack(request_bytes, ksana_python_inputs).OK());
  ASSERT_EQ(ksana_python_inputs.size(), 2ul);
  for (const auto& ksana_python_input : ksana_python_inputs) {
    ASSERT_EQ(ksana_python_input->sampling_config.max_new_tokens, 1);
    ASSERT_EQ(ksana_python_input->request_target.size(), 2ul);
  }
  // Some random checks
  ASSERT_TRUE(ksana_python_inputs[0]->request_target.count(target.target_name));
  ASSERT_TRUE(ksana_python_inputs[0]->request_target.count(target2.target_name));
  std::vector<std::pair<int, int>> real_slice_pos{{0, 1}, {3, 3}};  // 3 = -2 + 5
  ASSERT_EQ(ksana_python_inputs[0]->request_target[target.target_name].slice_pos, real_slice_pos);
  ASSERT_EQ(ksana_python_inputs[0]->request_target[target2.target_name].token_reduce_mode, TokenReduceMode::GATHER_ALL);
  ASSERT_TRUE(ksana_python_inputs[1]->request_target.count(target.target_name));
  ASSERT_TRUE(ksana_python_inputs[1]->request_target.count(target2.target_name));
  ASSERT_EQ(ksana_python_inputs[1]->request_target[target.target_name].token_id, request2.request_target[0].token_id);
  ASSERT_EQ(ksana_python_inputs[1]->request_target[target.target_name].token_reduce_mode,
            TokenReduceMode::GATHER_TOKEN_ID);
}

struct BatchRequestNewSerial {
  std::string id;
  std::vector<RequestSerial> requests;
  int error_code;

  MSGPACK_DEFINE_MAP(id, requests, error_code);
};

// Test redundant fields for api compatibility, expect to be ignored.
TEST_F(RequestPackerTest, RedundantFileds) {
  target.target_name = "transformer";
  target.token_reduce_mode = "GATHER_ALL";
  request.request_target.push_back(target);
  request.input_tokens = std::vector<int>{7, 6, 1};
  BatchRequestNewSerial batch_request;
  batch_request.requests.push_back(request);
  batch_request.id = "1";
  batch_request.error_code = 200;

  msgpack::pack(sbuf, batch_request);

  // Parse request
  std::string request_bytes(sbuf.data(), sbuf.size());
  ASSERT_TRUE(request_packer_.Unpack(request_bytes, ksana_python_inputs).OK());
  ASSERT_EQ(ksana_python_inputs.size(), 1ul);
  ASSERT_EQ(ksana_python_inputs[0]->input_tokens, request.input_tokens);
  ASSERT_EQ(ksana_python_inputs[0]->request_target.size(), 1);
  ASSERT_TRUE(ksana_python_inputs[0]->request_target.count(target.target_name));
  ASSERT_EQ(ksana_python_inputs[0]->request_target[target.target_name].token_reduce_mode, TokenReduceMode::GATHER_ALL);
}

// Test for various input errors.
TEST_F(RequestPackerTest, WrongInput) {
  BatchRequestSerial batch_request;
  RequestSerial& request = batch_request.requests.emplace_back();
  TargetRequestSerial& target = request.request_target.emplace_back();
  std::string request_bytes;

  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs)
                .GetMessage()
                .find("Missing 'target_name' in target description."),
            std::string::npos);

  target.target_name = "logits";
  target.token_reduce_mode = "GATHER_ALL";
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs)
                .GetMessage()
                .find("The output for logits does not support the 'GATHER_ALL' reduction mode."),
            std::string::npos);
  target.token_reduce_mode = "GATHER_TOKEN_ID";

  request.input_tokens = std::vector<int>{1};
  target.slice_pos.emplace_back(0, 1);  // [0, 1]
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs)
                .GetMessage()
                .find("Error: The end position of interval [0, 1] exceeds the total number of input tokens (1)."),
            std::string::npos);

  request.input_tokens = std::vector<int>{1, 2, 3, 4, 5};
  target.slice_pos.emplace_back(1, 2);  // [1, 2]
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs)
                .GetMessage()
                .find("Error: Interval [1, 2] overlaps with the previous interval ending at position 1."),
            std::string::npos);

  target.slice_pos.back().first = 2;  // [2, 2]
  target.slice_pos.emplace_back(-1, -2);
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs)
                .GetMessage()
                .find("Error: The end position of interval [4, 3] is less than its start position."),
            std::string::npos);

  target.slice_pos.back().second = -1;
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(
      request_packer_.Unpack(request_bytes, ksana_python_inputs)
          .GetMessage()
          .find("Get the last position is not supported for logits in the 'GATHER_TOKEN_ID' token reduction mode."),
      std::string::npos);
  target.slice_pos.pop_back();

  target.token_id = std::vector<int>{2};
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs)
                .GetMessage()
                .find("Unable to set both token_id and slice_pos at the same time."),
            std::string::npos);

  target.slice_pos.clear();
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs)
                .GetMessage()
                .find("Specifying token_id for logits output is not supported."),
            std::string::npos);

  target.token_reduce_mode = "asdfg";  // an invalid mode
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs)
                .GetMessage()
                .find(fmt::format("The specified token reduce mode in {} is invalid.", target.target_name)),
            std::string::npos);
  target.token_reduce_mode = "GATHER_TOKEN_ID";

  target.target_name = "layernorm";
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs)
                .GetMessage()
                .find(fmt::format("The output of the {} does not support 'GATHER_TOKEN_ID'.", target.target_name)),
            std::string::npos);

  target.target_name = "abcd";  // an invalid target
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs)
                .GetMessage()
                .find(fmt::format("Invalid target name {}.", target.target_name)),
            std::string::npos);

  request_bytes = "bad request";
  ASSERT_NE(
      request_packer_.Unpack(request_bytes, ksana_python_inputs).GetMessage().find("Failed to parse the request."),
      std::string::npos);
}

// Test for packing.
TEST_F(RequestPackerTest, NormalPack) {
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3, 4, 5};
  ksana_python_inputs.push_back(ksana_python_input);
  KsanaPythonOutput ksana_python_output;
  PythonTensor tensor;
  std::vector<float> probs_output = {-1.f, -2.f, .5f, 3.f, .7f};
  tensor.shape = {probs_output.size()};
  tensor.dtype = GetTypeString(TYPE_FP32);
  tensor.data.resize(probs_output.size() * sizeof(float));
  memcpy(tensor.data.data(), probs_output.data(), tensor.data.size());
  ksana_python_output.response["logits"] = tensor;
  ksana_python_outputs.push_back(ksana_python_output);

  std::string response_bytes;
  ASSERT_TRUE(request_packer_.Pack(ksana_python_inputs, ksana_python_outputs, response_bytes).OK());

  auto handle = msgpack::unpack(response_bytes.data(), response_bytes.size());
  auto object = handle.get();
  BatchResponseSerial batch_response = object.as<BatchResponseSerial>();
  ASSERT_EQ(batch_response.responses.size(), 1ul);
  ASSERT_EQ(batch_response.responses[0].response.size(), 1ul);
  ASSERT_EQ(batch_response.responses[0].response[0].target_name, "logits");
  ASSERT_EQ(batch_response.responses[0].response[0].tensor.shape[0], probs_output.size());
  auto probs_response_bytes =
      base64::decode_into<std::vector<uint8_t>>(batch_response.responses[0].response[0].tensor.data.begin(),
                                                batch_response.responses[0].response[0].tensor.data.end());
  std::vector<float> probs_response(probs_output.size());
  memcpy(probs_response.data(), probs_response_bytes.data(), probs_response_bytes.size());
  ASSERT_EQ(probs_response, probs_output);
}

}  // namespace ksana_llm
