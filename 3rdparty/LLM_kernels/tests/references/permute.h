/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <vector>

namespace llm_kernels {
namespace ascend {
namespace test {

constexpr int MAX_DIM_NUMS = 6;

size_t GetInputIndexPos(size_t i, size_t j, size_t k, size_t x, size_t y, size_t z,
                        const std::vector<uint64_t>& input_strides) {
  return (i * input_strides[0] + j * input_strides[1] + k * input_strides[2] + x * input_strides[3] +
          y * input_strides[4] + z * input_strides[5]);
}

size_t GetNewIndexPos(size_t i, size_t j, size_t k, size_t x, size_t y, size_t z,
                      const std::vector<uint64_t>& output_strides, const std::vector<uint64_t>& new_idxes) {
  size_t indexes[MAX_DIM_NUMS] = {i, j, k, x, y, z};
  return (indexes[new_idxes[0]] * output_strides[0] + indexes[new_idxes[1]] * output_strides[1] +
          indexes[new_idxes[2]] * output_strides[2] + indexes[new_idxes[3]] * output_strides[3] +
          indexes[new_idxes[4]] * output_strides[4] + indexes[new_idxes[5]] * output_strides[5]);
}

template <typename T>
void RunPermuteRef(void* input, void* output, const std::vector<uint64_t>& input_shape,
                   const std::vector<uint64_t>& output_shape, const std::vector<uint64_t>& permute_dims) {
  std::vector<uint64_t> unify_input_shape = input_shape;
  while (unify_input_shape.size() < MAX_DIM_NUMS) {
    unify_input_shape.insert(unify_input_shape.begin(), 1);
  }
  size_t padding_idx = MAX_DIM_NUMS - unify_input_shape.size();
  std::vector<uint64_t> input_strides;
  input_strides.resize(unify_input_shape.size(), 1);
  for (int64_t i = unify_input_shape.size() - 2; i >= 0; i--) {
    input_strides[i] = unify_input_shape[i + 1] * input_strides[i + 1];
  }

  std::vector<uint64_t> output_new_indexes = permute_dims;
  int fill_dim = MAX_DIM_NUMS - output_new_indexes.size();
  for (size_t i = 0; i < output_new_indexes.size(); ++i) {
    output_new_indexes[i] = output_new_indexes[i] + fill_dim;
  }
  fill_dim = fill_dim - 1;
  while (fill_dim >= 0) {
    output_new_indexes.insert(output_new_indexes.begin(), fill_dim);
    fill_dim = fill_dim - 1;
  }

  std::vector<int64_t> new_shape;
  for (auto i : output_new_indexes) {
    new_shape.push_back(unify_input_shape[i]);
  }

  std::vector<uint64_t> output_strides;
  output_strides.resize(new_shape.size(), 1);
  for (int64_t i = new_shape.size() - 2; i >= 0; i--) {
    output_strides[i] = new_shape[i + 1] * output_strides[i + 1];
  }

  for (size_t i = 0; i < unify_input_shape[0]; ++i) {
    for (size_t j = 0; j < unify_input_shape[1]; ++j) {
      for (size_t k = 0; k < unify_input_shape[2]; ++k) {
        for (size_t x = 0; x < unify_input_shape[3]; ++x) {
          for (size_t y = 0; y < unify_input_shape[4]; ++y) {
            for (size_t z = 0; z < unify_input_shape[5]; ++z) {
              size_t src_pos = GetInputIndexPos(i, j, k, x, y, z, input_strides);
              size_t dst_pos = GetNewIndexPos(i, j, k, x, y, z, output_strides, output_new_indexes);
              ((T*)output)[dst_pos] = ((T*)input)[src_pos];
            }
          }
        }
      }
    }
  }
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels