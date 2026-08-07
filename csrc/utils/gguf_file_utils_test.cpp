/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "utils/gguf_file_utils.h"

#include <filesystem>
#include <fstream>

#include "device/device_types.h"
#include "gtest/gtest.h"

namespace ksana = ksana_llm;

class GGUFLoadTest : public ::testing::Test {
 protected:
  std::string test_file_name_ = "test_model.gguf";
  std::unique_ptr<ksana::GGUFFileTensorLoader> tensor_loader_;

  void SetUp() override {
    CreateTestGGUFFile();
    tensor_loader_ = std::make_unique<ksana::GGUFFileTensorLoader>(test_file_name_, true);
  }

  void TearDown() override {
    if (std::filesystem::exists(test_file_name_)) {
      std::filesystem::remove(test_file_name_);
    }
  }

  void CreateTestGGUFFile() {
    std::ofstream outfile(test_file_name_, std::ios::binary);

    uint32_t magic = GGUF_MAGIC;
    outfile.write(reinterpret_cast<char*>(&magic), sizeof(uint32_t));

    uint32_t version = GGUF_VERSION;
    outfile.write(reinterpret_cast<char*>(&version), sizeof(uint32_t));

    uint64_t tensor_count = 1;
    outfile.write(reinterpret_cast<char*>(&tensor_count), sizeof(uint64_t));

    uint64_t metadata_kv_count = 0;
    outfile.write(reinterpret_cast<char*>(&metadata_kv_count), sizeof(uint64_t));

    std::string tensor_name = "test_tensor";
    uint64_t tensor_name_length = tensor_name.size();
    outfile.write(reinterpret_cast<char*>(&tensor_name_length), sizeof(uint64_t));
    outfile.write(tensor_name.c_str(), tensor_name_length);

    uint32_t n_dims = 2;
    outfile.write(reinterpret_cast<char*>(&n_dims), sizeof(uint32_t));

    uint64_t dim0 = 3;
    uint64_t dim1 = 4;
    outfile.write(reinterpret_cast<char*>(&dim0), sizeof(uint64_t));
    outfile.write(reinterpret_cast<char*>(&dim1), sizeof(uint64_t));

    uint32_t data_type = ksana::GGMLType::GGML_TYPE_F32;
    outfile.write(reinterpret_cast<char*>(&data_type), sizeof(uint32_t));

    uint64_t offset = 0;
    outfile.write(reinterpret_cast<char*>(&offset), sizeof(uint64_t));

    size_t header_size = outfile.tellp();
    size_t alignment = GGUF_ALIGNMENT;
    size_t padding = (alignment - (header_size % alignment)) % alignment;
    for (size_t i = 0; i < padding; ++i) {
      outfile.put(0);
    }

    offset = header_size + padding;
    outfile.seekp(header_size - sizeof(uint64_t), std::ios::beg);
    outfile.write(reinterpret_cast<char*>(&offset), sizeof(uint64_t));
    outfile.seekp(0, std::ios::end);

    size_t tensor_size = dim0 * dim1 * sizeof(float);
    std::vector<float> tensor_data(dim0 * dim1, 0.0f);
    outfile.write(reinterpret_cast<char*>(tensor_data.data()), tensor_size);

    outfile.close();
  }
};

TEST_F(GGUFLoadTest, GetTensorNameListTest) {
  const std::vector<std::string>& tensor_names = tensor_loader_->GetTensorNameList();
  ASSERT_EQ(tensor_names.size(), 1);
  EXPECT_EQ(tensor_names[0], "test_tensor");
}

TEST_F(GGUFLoadTest, GetTensorDataTest) {
  void* tensor_ptr;
  size_t tensor_size;

  std::tie(tensor_ptr, tensor_size) = tensor_loader_->GetTensor("test_tensor");

  ASSERT_NE(tensor_ptr, nullptr);
  EXPECT_EQ(tensor_size, 3 * 4 * sizeof(float));
}

TEST_F(GGUFLoadTest, GetTensorShapeTest) {
  std::vector<size_t> shape = tensor_loader_->GetTensorShape("test_tensor");
  ASSERT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], 4);
  EXPECT_EQ(shape[1], 3);
}

TEST_F(GGUFLoadTest, GetTensorDataTypeTest) {
  ksana::DataType data_type = tensor_loader_->GetTensorDataType("test_tensor");
  EXPECT_EQ(data_type, ksana::TYPE_FP32);
}

TEST_F(GGUFLoadTest, GetInvalidTensorTest) {
  void* tensor_ptr;
  size_t tensor_size;

  std::tie(tensor_ptr, tensor_size) = tensor_loader_->GetTensor("invalid_tensor");

  EXPECT_EQ(tensor_ptr, nullptr);
  EXPECT_EQ(tensor_size, 0);
}

TEST_F(GGUFLoadTest, GetMetadataTest) {
  auto context = tensor_loader_->GetMetadata();
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(context->metadata_map.size(), 0);
}

TEST_F(GGUFLoadTest, SetTensorTest) {
  torch::Tensor tensor = torch::zeros({3, 4}, torch::dtype(torch::kFloat32));

  tensor_loader_->SetTensor("test_tensor", tensor);

  void* tensor_ptr;
  size_t tensor_size;

  std::tie(tensor_ptr, tensor_size) = tensor_loader_->GetTensor("test_tensor");

  ASSERT_NE(tensor_ptr, nullptr);
  EXPECT_EQ(tensor_size, 3 * 4 * sizeof(float));

  EXPECT_EQ(tensor_ptr, tensor.data_ptr());
}

TEST_F(GGUFLoadTest, GetTensorFileNameTest) {
  std::string file_name = tensor_loader_->GetTensorFileName();
  EXPECT_EQ(file_name, test_file_name_);
}

TEST_F(GGUFLoadTest, MissingFileTest) {
  std::filesystem::remove(test_file_name_);

  EXPECT_THROW({ ksana::GGUFFileTensorLoader loader("nonexistent_file.gguf", true); }, std::runtime_error);
}

TEST_F(GGUFLoadTest, AlignmentTest) {
  auto context = tensor_loader_->GetMetadata();
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(context->alignment, GGUF_ALIGNMENT);
}

TEST_F(GGUFLoadTest, ConvertGGMLTypeToDataTypeTest) {
  ksana::DataType data_type = tensor_loader_->GetTensorDataType("test_tensor");
  EXPECT_EQ(data_type, ksana::TYPE_FP32);
}

TEST_F(GGUFLoadTest, ZeroTensorCountTest) {
  std::string zero_tensor_file = "zero_tensor.gguf";
  {
    std::ofstream outfile(zero_tensor_file, std::ios::binary);

    uint32_t magic = GGUF_MAGIC;
    outfile.write(reinterpret_cast<char*>(&magic), sizeof(uint32_t));

    uint32_t version = GGUF_VERSION;
    outfile.write(reinterpret_cast<char*>(&version), sizeof(uint32_t));

    uint64_t tensor_count = 0;
    outfile.write(reinterpret_cast<char*>(&tensor_count), sizeof(uint64_t));

    uint64_t metadata_kv_count = 0;
    outfile.write(reinterpret_cast<char*>(&metadata_kv_count), sizeof(uint64_t));

    outfile.close();
  }

  ksana::GGUFFileTensorLoader loader(zero_tensor_file, true);

  const std::vector<std::string>& tensor_names = loader.GetTensorNameList();
  EXPECT_EQ(tensor_names.size(), 0);

  std::filesystem::remove(zero_tensor_file);
}
