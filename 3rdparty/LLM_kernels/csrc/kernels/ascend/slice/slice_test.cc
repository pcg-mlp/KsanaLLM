/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "3rdparty/half/include/half.hpp"
#include "csrc/kernels/ascend/slice/slice.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"

#ifdef ENABLE_ACL_ATB
#  include "csrc/utils/ascend/atb_executor.h"
#endif

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {
namespace test {

class LlamaAscendSliceTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::context;
  using AscendTestSuitBase::default_device;
  using AscendTestSuitBase::is_inited;
  using AscendTestSuitBase::stream;

#ifdef ENABLE_ACL_ATB
  template <typename DTYPE>
  void TestATBQKVSlice() {
    constexpr uint32_t ntokens{512};
    constexpr uint32_t kv_head_size{40};
    constexpr uint32_t head_size{40};
    constexpr uint32_t head_dim{128};
    aclDataType aclnn_dtype = aclDataType::ACL_FLOAT16;
    if (std::is_same<DTYPE, float>::value) {
      aclnn_dtype = aclDataType::ACL_FLOAT;
    } else if (std::is_same<DTYPE, aclFloat16>::value || std::is_same<DTYPE, half_float::half>::value) {
      aclnn_dtype = aclDataType::ACL_FLOAT16;
    } else {
      GTEST_SKIP_("This test is just supported float and float16.");
    }

    // for input
    void* qkv_tensor_device_ptr = nullptr;
    size_t qkv_elem_nums = ntokens * ((head_size + 2 * kv_head_size) * head_dim);
    size_t qkv_size = qkv_elem_nums * sizeof(DTYPE);
    ACL_CHECK_RET(aclrtMalloc(&qkv_tensor_device_ptr, qkv_size, ACL_MEM_MALLOC_HUGE_FIRST));
    std::vector<DTYPE> qkv_tensor_host_vec(qkv_elem_nums);
    for (size_t i = 0; i < qkv_elem_nums; ++i) {
      if (std::is_same<DTYPE, aclFloat16>::value) {
        qkv_tensor_host_vec[i] = aclFloatToFloat16(float(std::cos(i)));
      } else if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
        qkv_tensor_host_vec[i] = DTYPE(std::cos(i));
      } else {
        throw std::invalid_argument("Invalid QKVSlice compute type, only support float16 or float32.");
      }
    }
    ACL_CHECK_RET(
        aclrtMemcpy(qkv_tensor_device_ptr, qkv_size, qkv_tensor_host_vec.data(), qkv_size, ACL_MEMCPY_HOST_TO_DEVICE));

    // for output
    void* q_tensor_ptr = nullptr;
    size_t q_elem_nums = ntokens * (head_size * head_dim);
    size_t q_size = q_elem_nums * sizeof(DTYPE);
    ACL_CHECK_RET(aclrtMalloc(&q_tensor_ptr, q_size, ACL_MEM_MALLOC_HUGE_FIRST));
    void* k_tensor_ptr = nullptr;
    size_t k_elem_nums = ntokens * (kv_head_size * head_dim);
    size_t k_size = k_elem_nums * sizeof(DTYPE);
    ACL_CHECK_RET(aclrtMalloc(&k_tensor_ptr, q_size, ACL_MEM_MALLOC_HUGE_FIRST));
    void* v_tensor_ptr = nullptr;
    size_t v_elem_nums = ntokens * (kv_head_size * head_dim);
    size_t v_size = v_elem_nums * sizeof(DTYPE);
    ACL_CHECK_RET(aclrtMalloc(&v_tensor_ptr, v_size, ACL_MEM_MALLOC_HUGE_FIRST));
    std::vector<DTYPE> q_tensor_host_ptr(q_elem_nums);
    std::vector<DTYPE> k_tensor_host_ptr(k_elem_nums);
    std::vector<DTYPE> v_tensor_host_ptr(v_elem_nums);

    // for ref output
    std::vector<DTYPE> q_tensor_ref_host_ptr(q_elem_nums);
    std::vector<DTYPE> k_tensor_ref_host_ptr(k_elem_nums);
    std::vector<DTYPE> v_tensor_ref_host_ptr(v_elem_nums);

    // create run graph
    unsigned int tensor_idx = 0;
    unsigned int input_qkv_tensor = tensor_idx++;  // [ntokens, 3 * head_num * head_dim] or [ntokens,
                                               // (head_num + 2 * kv_head_num) * head_dim]
    unsigned int output_q_tensor = tensor_idx++;   // [ntokens, head_num * head_dim]
    unsigned int output_k_tensor = tensor_idx++;
    unsigned int output_v_tensor = tensor_idx++;
    atb::GraphParam op_graph;
    op_graph.name = "TestATBQKVSlice";
    op_graph.inTensorNum = 1;
    op_graph.outTensorNum = 3;
    op_graph.internalTensorNum = 0;
    op_graph.nodes.resize(1);
    unsigned int node_idx = 0;
    atb::Node& op_node = op_graph.nodes.at(node_idx++);
    CreateSplitQKVATBOperation(ntokens, head_size, kv_head_size, head_dim, &op_node.operation);
    op_node.inTensorIds = {input_qkv_tensor};
    op_node.outTensorIds = {output_q_tensor, output_k_tensor, output_v_tensor};
    op_graph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc>& input_tensor_descs,
                                  atb::SVector<atb::TensorDesc>& output_tensor_descs) {
      output_tensor_descs.resize(3);
      output_tensor_descs.at(0) = input_tensor_descs.at(0);
      output_tensor_descs.at(0).shape.dims[0] = input_tensor_descs.at(0).shape.dims[0];
      output_tensor_descs.at(0).shape.dims[1] = head_size * head_dim;
      output_tensor_descs.at(1) = input_tensor_descs.at(0);
      output_tensor_descs.at(1).shape.dims[0] = input_tensor_descs.at(0).shape.dims[0];
      output_tensor_descs.at(1).shape.dims[1] = kv_head_size * head_dim;
      output_tensor_descs.at(2) = input_tensor_descs.at(0);
      output_tensor_descs.at(2).shape.dims[0] = input_tensor_descs.at(0).shape.dims[0];
      output_tensor_descs.at(2).shape.dims[1] = kv_head_size * head_dim;
      return atb::NO_ERROR;
    };

    llm_kernels::utils::ATBOperationExecutor atb_op_executor;
    atb_op_executor.Init(default_device, op_graph);
    atb_op_executor.ResetVariantPack();
    atb_op_executor.SetInputTensor(qkv_tensor_device_ptr, {ntokens, (head_size + 2 * kv_head_size) * head_dim},
                                   aclnn_dtype);
    atb_op_executor.SetOutputTensor(q_tensor_ptr, {ntokens, head_size * head_dim}, aclnn_dtype);
    atb_op_executor.SetOutputTensor(k_tensor_ptr, {ntokens, kv_head_size * head_dim}, aclnn_dtype);
    atb_op_executor.SetOutputTensor(v_tensor_ptr, {ntokens, kv_head_size * head_dim}, aclnn_dtype);

    atb_op_executor.Run(atb_context, llm_kernels::utils::GetTestWorkSpaceFunc);
    ACL_CHECK_RET(aclrtSynchronizeStream(stream));

    ACL_CHECK_RET(aclrtMemcpy(q_tensor_host_ptr.data(), q_size, q_tensor_ptr, q_size, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK_RET(aclrtMemcpy(k_tensor_host_ptr.data(), k_size, k_tensor_ptr, k_size, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK_RET(aclrtMemcpy(v_tensor_host_ptr.data(), v_size, v_tensor_ptr, v_size, ACL_MEMCPY_DEVICE_TO_HOST));

    for (size_t token_idx = 0; token_idx < ntokens; ++token_idx) {
      float val = 0.0f;
      float ref_val = 0.0f;
      // for q
      for (size_t idx = 0; idx < (head_size * head_dim); ++idx) {
        if (std::is_same<DTYPE, aclFloat16>::value) {
          val = aclFloat16ToFloat(q_tensor_host_ptr[token_idx * (head_size * head_dim) + idx]);
          ref_val =
              aclFloat16ToFloat(qkv_tensor_host_vec[token_idx * ((head_size + 2 * kv_head_size) * head_dim) + idx]);
        } else if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
          val = float(q_tensor_host_ptr[token_idx * (head_size * head_dim) + idx]);
          ref_val = float(qkv_tensor_host_vec[token_idx * ((head_size + 2 * kv_head_size) * head_dim) + idx]);
        } else {
          throw std::invalid_argument("Invalid QKVSlice compute type, only support float16 or float32.");
        }

        EXPECT_EQ(val, ref_val) << "q tensor token id: " << token_idx << ", idx: " << idx << " is different: " << val
                                << " v.s " << ref_val;
      }
      // for k
      for (size_t idx = 0; idx < (kv_head_size * head_dim); ++idx) {
        if (std::is_same<DTYPE, aclFloat16>::value) {
          val = aclFloat16ToFloat(k_tensor_host_ptr[token_idx * (kv_head_size * head_dim) + idx]);
          ref_val = aclFloat16ToFloat(qkv_tensor_host_vec[token_idx * ((head_size + 2 * kv_head_size) * head_dim) +
                                                          (head_size * head_dim) + idx]);
        } else if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
          val = float(k_tensor_host_ptr[token_idx * (kv_head_size * head_dim) + idx]);
          ref_val = float(qkv_tensor_host_vec[token_idx * ((head_size + 2 * kv_head_size) * head_dim) +
                                              (head_size * head_dim) + idx]);
        } else {
          throw std::invalid_argument("Invalid QKVSlice compute type, only support float16 or float32.");
        }
        EXPECT_EQ(val, ref_val) << "k tensor token id: " << token_idx << ", idx: " << idx << " is different: " << val
                                << " v.s " << ref_val;
      }
      // for v
      for (size_t idx = 0; idx < (kv_head_size * head_dim); ++idx) {
        if (std::is_same<DTYPE, aclFloat16>::value) {
          val = aclFloat16ToFloat(v_tensor_host_ptr[token_idx * (kv_head_size * head_dim) + idx]);
          ref_val = aclFloat16ToFloat(qkv_tensor_host_vec[token_idx * ((head_size + 2 * kv_head_size) * head_dim) +
                                                          ((head_size + kv_head_size) * head_dim) + idx]);
        } else if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
          val = float(v_tensor_host_ptr[token_idx * (kv_head_size * head_dim) + idx]);
          ref_val = float(qkv_tensor_host_vec[token_idx * ((head_size + 2 * kv_head_size) * head_dim) +
                                              ((head_size + kv_head_size) * head_dim) + idx]);
        } else {
          throw std::invalid_argument("Invalid QKVSlice compute type, only support float16 or float32.");
        }
        EXPECT_EQ(val, ref_val) << "v tensor token id: " << token_idx << ", idx: " << idx << " is different: " << val
                                << " v.s " << ref_val;
      }
    }

    ACL_CHECK_RET(aclrtFree(v_tensor_ptr));
    ACL_CHECK_RET(aclrtFree(k_tensor_ptr));
    ACL_CHECK_RET(aclrtFree(q_tensor_ptr));
    ACL_CHECK_RET(aclrtFree(qkv_tensor_device_ptr));
  }
#endif
};

TEST_F(LlamaAscendSliceTestSuit, SliceTest) {
  const std::vector<int64_t> input_shape = {4, 8};
  aclTensor* input_tensor = nullptr;
  void* input_workspace = nullptr;

  const std::vector<int64_t> output_shape = {4, 4};
  aclTensor* output_tensor = nullptr;
  void* output_workspace = nullptr;

  CreateAclTensor(input_shape, &input_workspace, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &input_tensor);
  CreateAclTensor(output_shape, &output_workspace, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &output_tensor);
  std::vector<half_float::half> input_vec_host(GetShapeSize(input_shape));
  std::vector<half_float::half> out_vec_host(GetShapeSize(output_shape));
  for (size_t i = 0; i < input_vec_host.size(); ++i) {
    input_vec_host[i] = (half_float::half)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
  }
  ACL_CHECK_RET(aclrtMemcpyAsync(input_workspace, GetShapeSize(input_shape) * sizeof(half_float::half),
                                 input_vec_host.data(), GetShapeSize(input_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  Slice(input_tensor, -1, 4, 8, 1, &output_tensor, stream, llm_kernels::utils::GetTestWorkSpaceFunc);
  ACL_CHECK_RET(aclrtMemcpyAsync(out_vec_host.data(), GetShapeSize(output_shape) * sizeof(half_float::half),
                                 output_workspace, GetShapeSize(output_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  for (size_t i = 0; i < input_shape[0]; ++i) {
    for (size_t j = 4; j < input_shape[1]; ++j) {
      EXPECT_NEAR(float(input_vec_host[i * input_shape[1] + j]), float(out_vec_host[i * output_shape[1] + j - 4]),
                  1e-5);
    }
  }

  ACL_CHECK_RET(aclDestroyTensor(output_tensor));
  ACL_CHECK_RET(aclDestroyTensor(input_tensor));
  ACL_CHECK_RET(aclrtFree(input_workspace));
  ACL_CHECK_RET(aclrtFree(output_workspace));
}

TEST_F(LlamaAscendSliceTestSuit, SliceKernelTest) {
  // A [10, 10] matrix.
  int row = 10;
  int col = 10;

  std::vector<float> data(row * col, 0.0);
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j) {
      data[i * row + j] = i + (0.001 * j);
    }
  }

  void* input_data_dev;
  size_t input_size = row * col * sizeof(float);
  ACL_CHECK_RET(aclrtMalloc(&input_data_dev, input_size + 32, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMemcpy(input_data_dev, input_size, data.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE));

  // A [10, 3] matrix
  int output_row = 10;
  int output_col = 3;

  void* output_data_dev;
  size_t output_size = output_row * output_col * sizeof(float);
  ACL_CHECK_RET(aclrtMalloc(&output_data_dev, output_size + 32, ACL_MEM_MALLOC_HUGE_FIRST));

  Slice2<float> slice;
  uint32_t start_offset = 3;
  uint32_t slice_length = 3;
  uint32_t slice_step = 10;
  uint32_t slice_times = 10;
  slice.Forward(output_data_dev, input_data_dev, start_offset, slice_length, slice_step, slice_times, stream);
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  std::vector<float> result(output_row * output_col, 0);
  ACL_CHECK_RET(aclrtMemcpy(result.data(), result.size() * sizeof(float), output_data_dev,
                            output_row * output_col * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

  size_t idx = 0;
  for (int i = 0; i < output_row; ++i) {
    for (int j = 0; j < output_col; ++j) {
      EXPECT_FLOAT_EQ(i + (0.001 * (3 + j)), result[idx++]);
    }
  }
}

#ifdef ENABLE_ACL_ATB
TEST_F(LlamaAscendSliceTestSuit, ATBSliceTest) { TestATBQKVSlice<half_float::half>(); }
#endif

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels
