/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "ipc/core/ipc_func_serialization.h"
#include "utils/ret_code.h"
#include "utils/status.h"

using namespace ksana_llm;

class IpcFuncSerializationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Allocate sufficient buffer for testing
    buffer_size = 1024 * 1024;  // 1MB
    buffer = new uint8_t[buffer_size];
    std::memset(buffer, 0, buffer_size);
  }

  void TearDown() override { delete[] buffer; }

  size_t buffer_size;
  uint8_t* buffer;
};

// Test basic POD type serialization/deserialization
TEST_F(IpcFuncSerializationTest, TestPodTypes) {
  size_t offset = 0;

  // Test various POD types
  int32_t int_val = 42;
  int64_t long_val = 123456789012345LL;
  float float_val = 3.14159f;
  double double_val = 2.718281828459045;
  bool bool_val = true;

  // Serialize
  SerializePod<int32_t>(int_val, buffer, offset);
  SerializePod<int64_t>(long_val, buffer, offset);
  SerializePod<float>(float_val, buffer, offset);
  SerializePod<double>(double_val, buffer, offset);
  SerializePod<bool>(bool_val, buffer, offset);

  // Deserialize
  offset = 0;
  int32_t int_val_des = DeserializePod<int32_t>(buffer, offset);
  int64_t long_val_des = DeserializePod<int64_t>(buffer, offset);
  float float_val_des = DeserializePod<float>(buffer, offset);
  double double_val_des = DeserializePod<double>(buffer, offset);
  bool bool_val_des = DeserializePod<bool>(buffer, offset);

  EXPECT_EQ(int_val, int_val_des);
  EXPECT_EQ(long_val, long_val_des);
  EXPECT_FLOAT_EQ(float_val, float_val_des);
  EXPECT_DOUBLE_EQ(double_val, double_val_des);
  EXPECT_EQ(bool_val, bool_val_des);
}

// Test enum type serialization/deserialization
TEST_F(IpcFuncSerializationTest, TestEnumTypes) {
  size_t offset = 0;

  // Test enum types
  RetCode ret_code = RET_SUCCESS;
  RetCode ret_code_error = RET_RUNTIME_FAILED;

  // Serialize
  SerializeEnum<RetCode>(ret_code, buffer, offset);
  SerializeEnum<RetCode>(ret_code_error, buffer, offset);

  // Deserialize
  offset = 0;
  RetCode ret_code_des = DeserializeEnum<RetCode>(buffer, offset);
  RetCode ret_code_error_des = DeserializeEnum<RetCode>(buffer, offset);

  EXPECT_EQ(ret_code, ret_code_des);
  EXPECT_EQ(ret_code_error, ret_code_error_des);
}

// Test string serialization/deserialization
TEST_F(IpcFuncSerializationTest, TestStringSerialization) {
  size_t offset = 0;

  // Test various strings
  std::string empty_str = "";
  std::string short_str = "Hello";
  std::string long_str = "This is a long string for testing serialization capabilities";
  std::string special_chars = "Special chars: \n\t\r\\\"'";

  // Serialize
  SerializeString(empty_str, buffer, offset);
  SerializeString(short_str, buffer, offset);
  SerializeString(long_str, buffer, offset);
  SerializeString(special_chars, buffer, offset);

  // Deserialize
  offset = 0;
  std::string empty_str_des = DeserializeString(buffer, offset);
  std::string short_str_des = DeserializeString(buffer, offset);
  std::string long_str_des = DeserializeString(buffer, offset);
  std::string special_chars_des = DeserializeString(buffer, offset);

  EXPECT_EQ(empty_str, empty_str_des);
  EXPECT_EQ(short_str, short_str_des);
  EXPECT_EQ(long_str, long_str_des);
  EXPECT_EQ(special_chars, special_chars_des);
}

// Test memory block serialization/deserialization
TEST_F(IpcFuncSerializationTest, TestMemorySerialization) {
  size_t offset = 0;

  // Prepare test data
  const size_t kDataSize = 256;
  uint8_t original_data[kDataSize];
  uint8_t deserialized_data[kDataSize];

  // Fill with sequential data
  for (size_t i = 0; i < kDataSize; ++i) {
    original_data[i] = static_cast<uint8_t>(i % 256);
  }

  // Serialize
  SerializeMemory(original_data, kDataSize, buffer, offset);

  // Deserialize
  offset = 0;
  DeserializeMemory(deserialized_data, buffer, offset);

  // Verify data consistency
  EXPECT_EQ(0, std::memcmp(original_data, deserialized_data, kDataSize));
}

// Test array serialization/deserialization
TEST_F(IpcFuncSerializationTest, TestArraySerialization) {
  size_t offset = 0;

  // Prepare test array
  const size_t kArraySize = 10;
  int32_t original_array[kArraySize];
  int32_t deserialized_array[kArraySize];

  for (size_t i = 0; i < kArraySize; ++i) {
    original_array[i] = static_cast<int32_t>(i * 10);
  }

  // Serialize
  SerializeArray<int32_t>(original_array, kArraySize, buffer, offset);

  // Deserialize
  offset = 0;
  DeserializeArray<int32_t>(deserialized_array, buffer, offset);

  // Verify data consistency
  for (size_t i = 0; i < kArraySize; ++i) {
    EXPECT_EQ(original_array[i], deserialized_array[i]);
  }
}

// Test vector serialization/deserialization
TEST_F(IpcFuncSerializationTest, TestVectorSerialization) {
  size_t offset = 0;

  // Test integer vectors
  std::vector<int32_t> int_vec = {1, 2, 3, 4, 5};
  std::vector<int32_t> empty_int_vec = {};
  std::vector<int32_t> large_int_vec;
  for (int i = 0; i < 100; ++i) {
    large_int_vec.push_back(i);
  }

  // Serialize
  SerializeVector<int32_t>(int_vec, buffer, offset);
  SerializeVector<int32_t>(empty_int_vec, buffer, offset);
  SerializeVector<int32_t>(large_int_vec, buffer, offset);

  // Deserialize
  offset = 0;
  std::vector<int32_t> int_vec_des;
  std::vector<int32_t> empty_int_vec_des;
  std::vector<int32_t> large_int_vec_des;

  DeserializeVector<int32_t>(int_vec_des, buffer, offset);
  DeserializeVector<int32_t>(empty_int_vec_des, buffer, offset);
  DeserializeVector<int32_t>(large_int_vec_des, buffer, offset);

  EXPECT_EQ(int_vec, int_vec_des);
  EXPECT_EQ(empty_int_vec, empty_int_vec_des);
  EXPECT_EQ(large_int_vec, large_int_vec_des);
}

// Test string vector serialization/deserialization
TEST_F(IpcFuncSerializationTest, TestStringVectorSerialization) {
  size_t offset = 0;

  // Test string vectors
  std::vector<std::string> str_vec = {"hello", "world", "test"};
  std::vector<std::string> empty_str_vec = {};
  std::vector<std::string> mixed_str_vec = {"", "single", "multiple words"};

  // Serialize
  SerializeVector<std::string>(str_vec, buffer, offset);
  SerializeVector<std::string>(empty_str_vec, buffer, offset);
  SerializeVector<std::string>(mixed_str_vec, buffer, offset);

  // Deserialize
  offset = 0;
  std::vector<std::string> str_vec_des;
  std::vector<std::string> empty_str_vec_des;
  std::vector<std::string> mixed_str_vec_des;

  DeserializeVector<std::string>(str_vec_des, buffer, offset);
  DeserializeVector<std::string>(empty_str_vec_des, buffer, offset);
  DeserializeVector<std::string>(mixed_str_vec_des, buffer, offset);

  EXPECT_EQ(str_vec, str_vec_des);
  EXPECT_EQ(empty_str_vec, empty_str_vec_des);
  EXPECT_EQ(mixed_str_vec, mixed_str_vec_des);
}

// Test dictionary to vector serialization/deserialization
TEST_F(IpcFuncSerializationTest, TestKeyToVectorSerialization) {
  size_t offset = 0;

  // Test dictionary
  std::unordered_map<int32_t, std::vector<std::string>> original_dict;
  original_dict[1] = {"one", "first"};
  original_dict[2] = {"two", "second"};
  original_dict[3] = {};  // Empty vector

  std::unordered_map<int32_t, std::vector<std::string>> deserialized_dict;

  // Serialize
  SerializeKeyToVector<int32_t, std::string>(original_dict, buffer, offset);

  // Deserialize
  offset = 0;
  DeserializeKeyToVector<int32_t, std::string>(deserialized_dict, buffer, offset);

  EXPECT_EQ(original_dict.size(), deserialized_dict.size());
  for (const auto& pair : original_dict) {
    EXPECT_EQ(pair.second, deserialized_dict[pair.first]);
  }
}

// Test vector of vectors serialization/deserialization
TEST_F(IpcFuncSerializationTest, TestVectorOfVectorSerialization) {
  size_t offset = 0;

  // Test vector of vectors
  std::vector<std::vector<int32_t>> original_vecs = {{1, 2, 3}, {4, 5}, {}, {6, 7, 8, 9}};

  std::vector<std::vector<int32_t>> deserialized_vecs;

  // Serialize
  SerializeVectorOfVector<int32_t>(original_vecs, buffer, offset);

  // Deserialize
  offset = 0;
  DeserializeVectorOfVector<int32_t>(deserialized_vecs, buffer, offset);

  EXPECT_EQ(original_vecs.size(), deserialized_vecs.size());
  for (size_t i = 0; i < original_vecs.size(); ++i) {
    EXPECT_EQ(original_vecs[i], deserialized_vecs[i]);
  }
}

// Test status serialization/deserialization
TEST_F(IpcFuncSerializationTest, TestStatusSerialization) {
  size_t offset = 0;

  // Test various statuses
  Status ok_status = Status();
  Status error_status = Status(RET_RUNTIME_FAILED, "Runtime error occurred");
  Status warning_status = Status(RET_INVALID_ARGUMENT, "Invalid argument provided");

  // Serialize
  SerializeStatus(ok_status, buffer, offset);
  SerializeStatus(error_status, buffer, offset);
  SerializeStatus(warning_status, buffer, offset);

  // Deserialize
  offset = 0;
  Status ok_status_des = DeserializeStatus(buffer, offset);
  Status error_status_des = DeserializeStatus(buffer, offset);
  Status warning_status_des = DeserializeStatus(buffer, offset);

  EXPECT_TRUE(ok_status_des.OK());
  EXPECT_EQ(error_status.GetCode(), error_status_des.GetCode());
  EXPECT_EQ(error_status.GetMessage(), error_status_des.GetMessage());
  EXPECT_EQ(warning_status.GetCode(), warning_status_des.GetCode());
  EXPECT_EQ(warning_status.GetMessage(), warning_status_des.GetMessage());
}

// Test edge cases
TEST_F(IpcFuncSerializationTest, TestEdgeCases) {
  size_t offset = 0;

  // Test empty string
  std::string empty_str = "";
  SerializeString(empty_str, buffer, offset);

  offset = 0;
  std::string empty_str_des = DeserializeString(buffer, offset);
  EXPECT_EQ(empty_str, empty_str_des);

  // Test zero-length array
  offset = 0;
  int32_t empty_array[0];
  SerializeArray<int32_t>(empty_array, 0, buffer, offset);

  offset = 0;
  int32_t deserialized_array[0];
  DeserializeArray<int32_t>(deserialized_array, buffer, offset);

  // Test zero-length memory block
  offset = 0;
  uint8_t empty_memory[0];
  SerializeMemory(empty_memory, 0, buffer, offset);

  offset = 0;
  uint8_t deserialized_memory[0];
  DeserializeMemory(deserialized_memory, buffer, offset);
}

// Test serialization order consistency
TEST_F(IpcFuncSerializationTest, TestSerializationOrder) {
  size_t offset = 0;

  // Mixed type serialization
  int32_t int_val = 42;
  std::string str_val = "test string";
  float float_val = 3.14f;
  std::vector<int32_t> vec_val = {1, 2, 3};

  // Serialize in order
  SerializePod<int32_t>(int_val, buffer, offset);
  SerializeString(str_val, buffer, offset);
  SerializePod<float>(float_val, buffer, offset);
  SerializeVector<int32_t>(vec_val, buffer, offset);

  // Deserialize in same order
  offset = 0;
  int32_t int_val_des = DeserializePod<int32_t>(buffer, offset);
  std::string str_val_des = DeserializeString(buffer, offset);
  float float_val_des = DeserializePod<float>(buffer, offset);
  std::vector<int32_t> vec_val_des;
  DeserializeVector<int32_t>(vec_val_des, buffer, offset);

  EXPECT_EQ(int_val, int_val_des);
  EXPECT_EQ(str_val, str_val_des);
  EXPECT_FLOAT_EQ(float_val, float_val_des);
  EXPECT_EQ(vec_val, vec_val_des);
}

// Test large data serialization
TEST_F(IpcFuncSerializationTest, TestLargeDataSerialization) {
  size_t offset = 0;

  // Create large capacity data
  const size_t large_size = 10000;
  std::vector<int32_t> large_vector(large_size);
  for (size_t i = 0; i < large_size; ++i) {
    large_vector[i] = static_cast<int32_t>(i);
  }

  // Serialize
  SerializeVector<int32_t>(large_vector, buffer, offset);

  // Deserialize
  offset = 0;
  std::vector<int32_t> large_vector_des;
  DeserializeVector<int32_t>(large_vector_des, buffer, offset);

  EXPECT_EQ(large_vector.size(), large_vector_des.size());
  EXPECT_EQ(large_vector, large_vector_des);
}

// Test special characters and Unicode strings
TEST_F(IpcFuncSerializationTest, TestSpecialCharacters) {
  size_t offset = 0;

  // Test various special characters
  std::string special_str = "Line1\nLine2\tTab\rCarriage\\Backslash\"Quote";
  std::string unicode_str = "中文测试 Español Français 日本語";
  std::string binary_str = "\x00\x01\x02\x03\x04\x05";

  // Serialize
  SerializeString(special_str, buffer, offset);
  SerializeString(unicode_str, buffer, offset);
  SerializeString(binary_str, buffer, offset);

  // Deserialize
  offset = 0;
  std::string special_str_des = DeserializeString(buffer, offset);
  std::string unicode_str_des = DeserializeString(buffer, offset);
  std::string binary_str_des = DeserializeString(buffer, offset);

  EXPECT_EQ(special_str, special_str_des);
  EXPECT_EQ(unicode_str, unicode_str_des);
  EXPECT_EQ(binary_str, binary_str_des);
}

// Test offset parameter correctness
TEST_F(IpcFuncSerializationTest, TestOffsetParameter) {
  size_t offset = 100;  // Start from non-zero position

  int32_t int_val = 123;
  std::string str_val = "offset test";

  // Serialize
  SerializePod<int32_t>(int_val, buffer, offset);
  SerializeString(str_val, buffer, offset);

  // Deserialize
  offset = 100;
  int32_t int_val_des = DeserializePod<int32_t>(buffer, offset);
  std::string str_val_des = DeserializeString(buffer, offset);

  EXPECT_EQ(int_val, int_val_des);
  EXPECT_EQ(str_val, str_val_des);
}
