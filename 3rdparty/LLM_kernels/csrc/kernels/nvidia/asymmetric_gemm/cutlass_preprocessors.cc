/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "csrc/kernels/nvidia/asymmetric_gemm/cutlass_preprocessors.h"

#include "csrc/kernels/nvidia/cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h"
#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

int32_t GetBitsInQuantType(QuantType quant_type) {
  switch (quant_type) {
    case QuantType::INT8_WEIGHT_ONLY:
      return 8;
    case QuantType::PACKED_INT4_WEIGHT_ONLY:
      return 4;
    default:
      throw std::invalid_argument("Invalid quant_type");
      return -1;
  }
}

struct LayoutDetails {
  enum class Layout { UNKNOWN, ROW_MAJOR, COLUMN_MAJOR };

  Layout layoutB = Layout::UNKNOWN;
  int32_t rows_per_column_tile = 1;
  int32_t columns_interleaved = 1;

  bool uses_imma_ldsm = false;
};

template <typename Layout>
struct GetLayoutDetails {};

template <>
struct GetLayoutDetails<cutlass::layout::RowMajor> {
  LayoutDetails operator()() {
    LayoutDetails layout_details;
    layout_details.layoutB = LayoutDetails::Layout::ROW_MAJOR;
    return layout_details;
  }
};

template <>
struct GetLayoutDetails<cutlass::layout::ColumnMajor> {
  LayoutDetails operator()() {
    LayoutDetails layout_details;
    layout_details.layoutB = LayoutDetails::Layout::COLUMN_MAJOR;
    return layout_details;
  }
};

template <int32_t RowsPerTile, int32_t ColumnsInterleaved>
struct GetLayoutDetails<cutlass::layout::ColumnMajorTileInterleave<RowsPerTile, ColumnsInterleaved>> {
  LayoutDetails operator()() {
    LayoutDetails layout_details;
    layout_details.layoutB = LayoutDetails::Layout::COLUMN_MAJOR;
    layout_details.rows_per_column_tile = RowsPerTile;
    layout_details.columns_interleaved = ColumnsInterleaved;
    return layout_details;
  }
};

template <typename cutlassArch, typename TypeB>
LayoutDetails GetLayoutDetailsForArchAndQuantType() {
  using CompileTraits = cutlass::gemm::kernel::LayoutDetailsB<TypeB, cutlassArch>;
  using LayoutB = typename CompileTraits::Layout;
  using MmaOperator = typename CompileTraits::Operator;
  LayoutDetails details = GetLayoutDetails<LayoutB>()();
  details.uses_imma_ldsm = std::is_same<MmaOperator, cutlass::arch::OpMultiplyAddDequantizeInterleavedBToA>::value;
  return details;
}

template <typename cutlassArch>
LayoutDetails GetLayoutDetailsForArch(QuantType quant_type) {
  LayoutDetails details;
  if (quant_type == QuantType::INT8_WEIGHT_ONLY) {
    details = GetLayoutDetailsForArchAndQuantType<cutlassArch, uint8_t>();
  } else if (quant_type == QuantType::PACKED_INT4_WEIGHT_ONLY) {
    details = GetLayoutDetailsForArchAndQuantType<cutlassArch, cutlass::uint4b_t>();
  } else {
    throw std::invalid_argument("Unsupported quantization type");
  }
  return details;
}

LayoutDetails GetLayoutDetailsForTransform(QuantType quant_type) {
  const int32_t arch = GetSMVersion();
  if (arch >= NVIDIA_VOLTA_GPU_COMPUTE_CAPABILITY && arch < NVIDIA_TURING_GPU_COMPUTE_CAPABILITY) {
    return GetLayoutDetailsForArch<cutlass::arch::Sm70>(quant_type);
  } else if (arch >= NVIDIA_TURING_GPU_COMPUTE_CAPABILITY && arch < NVIDIA_AMPERE_GPU_COMPUTE_CAPABILITY) {
    return GetLayoutDetailsForArch<cutlass::arch::Sm75>(quant_type);
  } else if (arch >= NVIDIA_AMPERE_GPU_COMPUTE_CAPABILITY && arch <= NVIDIA_HOPPER_GPU_COMPUTE_CAPABILITY) {
    return GetLayoutDetailsForArch<cutlass::arch::Sm80>(quant_type);
  } else {
    throw std::invalid_argument("Unsupported Nvidia GPU Arch");
    return LayoutDetails();
  }
}

// Permutes the rows of B for Turing and Ampere. Throws an error for other architectures.
// The data is permuted such that:
// For int8, each group of 16 rows is permuted using the map below:
//  0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15
// For int4, each group of 32 rows is permuted using the map below:
//  0 1 8 9 16 17 24 25 2 3 10 11 18 19 26 27 4 5 12 13 20 21 28 29 6 7 14 15 22 23 30 31
void PermuteBRowsForMixedGemm(int8_t* permuted_quantized_tensor, const int8_t* quantized_tensor,
                              const std::vector<size_t>& shape, QuantType quant_type, const int64_t arch_version) {
  // We only want to run this step for weight only quant.
  if (quant_type != QuantType::PACKED_INT4_WEIGHT_ONLY && quant_type != QuantType::INT8_WEIGHT_ONLY) {
    throw std::invalid_argument("Invalid quant type, only support packed int4 weight only or int8 weight only.");
  }

  if (shape.size() != 2 && shape.size() != 3) {
    throw std::invalid_argument("Shape must be 2-D or 3-D");
  }
  const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
  const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
  const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

  const int32_t BITS_PER_ELT = GetBitsInQuantType(quant_type);
  const int32_t K = 16 / BITS_PER_ELT;
  const int32_t ELTS_PER_BYTE = 8 / BITS_PER_ELT;
  const int32_t ELTS_PER_REG = 32 / BITS_PER_ELT;

  const uint32_t* input_byte_ptr = reinterpret_cast<const uint32_t*>(quantized_tensor);
  uint32_t* output_byte_ptr = reinterpret_cast<uint32_t*>(permuted_quantized_tensor);

  int32_t MMA_SHAPE_N = 8;
  int32_t B_ROWS_PER_MMA = 8 * K;
  const int32_t elts_in_int32 = 32 / BITS_PER_ELT;

  const int32_t num_vec_cols = num_cols / elts_in_int32;

  if (arch_version < NVIDIA_TURING_GPU_COMPUTE_CAPABILITY) {
    throw std::invalid_argument("Unsupported Arch. Pre-volta not supported. Column interleave not needed on Volta.");
  }
  if (num_rows % B_ROWS_PER_MMA != 0) {
    throw std::invalid_argument(
        "Invalid shape for quantized tensor. Number of rows of quantized matrix must be a multiple of " +
        std::to_string(B_ROWS_PER_MMA));
  }
  if (num_cols % MMA_SHAPE_N != 0) {
    throw std::invalid_argument(
        "Invalid shape for quantized tensor. On turing/Ampere, the number of cols must be a multiple of " +
        std::to_string(MMA_SHAPE_N));
  }

  // The code is written as below so it works for both int8 and packed int4.
  for (int32_t expert = 0; expert < num_experts; ++expert) {
    const int64_t matrix_offset = expert * int64_t(num_rows) * int64_t(num_vec_cols);
    for (int32_t base_row = 0; base_row < num_rows; base_row += B_ROWS_PER_MMA) {
      for (int32_t tile_row = 0; tile_row < B_ROWS_PER_MMA; ++tile_row) {
        for (int32_t write_col = 0; write_col < num_vec_cols; ++write_col) {
          const int32_t write_row = base_row + tile_row;
          const int32_t tile_read_row =
              8 * (((tile_row % ELTS_PER_REG) / 2)) + tile_row % 2 + 2 * (tile_row / ELTS_PER_REG);
          const int32_t read_row = base_row + tile_read_row;
          const int32_t read_col = write_col;

          const int64_t read_offset = matrix_offset + int64_t(read_row) * num_vec_cols + read_col;
          const int64_t write_offset = matrix_offset + int64_t(write_row) * num_vec_cols + write_col;

          output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
        }
      }
    }
  }
}

// We need to use this transpose to correctly handle packed int4 and int8 data
// The reason this code is relatively complex is that the "trivial" loops took a substantial
// amount of time to transpose leading to long preprocessing times. This seemed to be a big
// issue for relatively large models.
template <QuantType quant_type>
void SubbyteTranspose_impl(int8_t* transposed_quantized_tensor, const int8_t* quantized_tensor,
                           const std::vector<size_t>& shape) {
  const int32_t bits_per_elt = GetBitsInQuantType(quant_type);

  if (shape.size() != 2 && shape.size() != 3) {
    throw std::invalid_argument("Shape must be 2-D or 3-D");
  }
  const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
  const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
  const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

  const size_t col_bytes = num_cols * bits_per_elt / 8;
  const size_t col_bytes_trans = num_rows * bits_per_elt / 8;
  const size_t num_bytes = size_t(num_experts) * num_rows * col_bytes;

  const uint8_t* input_byte_ptr = reinterpret_cast<const uint8_t*>(quantized_tensor);
  uint8_t* output_byte_ptr = reinterpret_cast<uint8_t*>(transposed_quantized_tensor);

  static_assert(quant_type == QuantType::INT8_WEIGHT_ONLY || quant_type == QuantType::PACKED_INT4_WEIGHT_ONLY, "");
  static constexpr int32_t ELTS_PER_BYTE = quant_type == QuantType::INT8_WEIGHT_ONLY ? 1 : 2;

  static constexpr int32_t M_TILE_L1 = 64;
  static constexpr int32_t N_TILE_L1 = M_TILE_L1 / ELTS_PER_BYTE;
  uint8_t cache_buf[M_TILE_L1][N_TILE_L1];

  static constexpr int32_t VECTOR_WIDTH = std::min(32, N_TILE_L1);

  // We assume the dims are a multiple of vector width. Our kernels only handle dims which are multiples
  // of 64 for weight-only quantization. As a result, this seemed like a reasonable tradeoff because it
  // allows GCC to emit vector instructions.
  if (!(!(col_bytes_trans % VECTOR_WIDTH) && !(col_bytes % VECTOR_WIDTH))) {
    throw std::invalid_argument("Number of bytes for rows and cols must be a multiple of " +
                                std::to_string(VECTOR_WIDTH) + ". However, num_rows_bytes = " +
                                std::to_string(col_bytes_trans) + " and num_col_bytes = " + std::to_string(col_bytes));
  }

  const int32_t num_m_tiles = (num_rows + M_TILE_L1 - 1) / M_TILE_L1;
  const int32_t num_n_tiles = (col_bytes + N_TILE_L1 - 1) / N_TILE_L1;

  for (size_t expert = 0; expert < num_experts; ++expert) {
    const size_t matrix_offset = expert * num_rows * col_bytes;
    for (size_t row_tile_start = 0; row_tile_start < num_rows; row_tile_start += M_TILE_L1) {
      for (size_t col_tile_start_byte = 0; col_tile_start_byte < col_bytes; col_tile_start_byte += N_TILE_L1) {
        const int32_t row_limit = std::min(row_tile_start + M_TILE_L1, num_rows);
        const int32_t col_limit = std::min(col_tile_start_byte + N_TILE_L1, col_bytes);

        for (int32_t ii = 0; ii < M_TILE_L1; ++ii) {
          const int32_t row = row_tile_start + ii;

          for (int32_t jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH) {
            const int32_t col = col_tile_start_byte + jj;

            const size_t logical_src_offset = matrix_offset + row * col_bytes + col;

            if (row < row_limit && col < col_limit) {
              for (int32_t v = 0; v < VECTOR_WIDTH; ++v) {
                cache_buf[ii][jj + v] = input_byte_ptr[logical_src_offset + v];
              }
            }
          }
        }

        if (quant_type == QuantType::INT8_WEIGHT_ONLY) {
          for (int32_t ii = 0; ii < M_TILE_L1; ++ii) {
            for (int32_t jj = ii + 1; jj < N_TILE_L1; ++jj) {
              std::swap(cache_buf[ii][jj], cache_buf[jj][ii]);
            }
          }
        } else if (quant_type == QuantType::PACKED_INT4_WEIGHT_ONLY) {
          for (int32_t ii = 0; ii < M_TILE_L1; ++ii) {
            // Using M_TILE_L1 here is deliberate since we assume that the cache tile
            // is square in the number of elements (not necessarily the number of bytes).
            for (int32_t jj = ii + 1; jj < M_TILE_L1; ++jj) {
              const int32_t ii_byte = ii / ELTS_PER_BYTE;
              const int32_t ii_bit_offset = ii % ELTS_PER_BYTE;

              const int32_t jj_byte = jj / ELTS_PER_BYTE;
              const int32_t jj_bit_offset = jj % ELTS_PER_BYTE;

              uint8_t src_elt = 0xF & (cache_buf[ii][jj_byte] >> (4 * jj_bit_offset));
              uint8_t tgt_elt = 0xF & (cache_buf[jj][ii_byte] >> (4 * ii_bit_offset));

              cache_buf[ii][jj_byte] &= (0xF0 >> (4 * jj_bit_offset));
              cache_buf[jj][ii_byte] &= (0xF0 >> (4 * ii_bit_offset));

              cache_buf[ii][jj_byte] |= (tgt_elt << (4 * jj_bit_offset));
              cache_buf[jj][ii_byte] |= (src_elt << (4 * ii_bit_offset));
            }
          }
        } else {
          throw std::invalid_argument("Unsupported quantization type.");
        }

        const size_t row_tile_start_trans = col_tile_start_byte * ELTS_PER_BYTE;
        const size_t col_tile_start_byte_trans = row_tile_start / ELTS_PER_BYTE;

        const int32_t row_limit_trans = std::min(row_tile_start_trans + M_TILE_L1, num_cols);
        const int32_t col_limit_trans = std::min(col_tile_start_byte_trans + N_TILE_L1, col_bytes_trans);

        for (int32_t ii = 0; ii < M_TILE_L1; ++ii) {
          const int32_t row = row_tile_start_trans + ii;
          for (int32_t jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH) {
            const int32_t col = col_tile_start_byte_trans + jj;

            const size_t logical_tgt_offset = matrix_offset + row * col_bytes_trans + col;

            if (row < row_limit_trans && col < col_limit_trans) {
              for (int32_t v = 0; v < VECTOR_WIDTH; ++v) {
                output_byte_ptr[logical_tgt_offset + v] = cache_buf[ii][jj + v];
              }
            }
          }
        }
      }
    }
  }
}

void SubbyteTranspose(int8_t* transposed_quantized_tensor, const int8_t* quantized_tensor,
                      const std::vector<size_t>& shape, QuantType quant_type) {
  if (quant_type == QuantType::INT8_WEIGHT_ONLY) {
    SubbyteTranspose_impl<QuantType::INT8_WEIGHT_ONLY>(transposed_quantized_tensor, quantized_tensor, shape);
  } else if (quant_type == QuantType::PACKED_INT4_WEIGHT_ONLY) {
    SubbyteTranspose_impl<QuantType::PACKED_INT4_WEIGHT_ONLY>(transposed_quantized_tensor, quantized_tensor, shape);
  } else {
    throw std::invalid_argument("Invalid quant_type");
  }
}

void AddBiasAndInterleaveInt8sInplace(int8_t* int8_tensor, const size_t num_elts) {
  for (int32_t ii = 0; ii < num_elts; ++ii) {
    int8_tensor[ii] = int8_t(int32_t(int8_tensor[ii]) + 128);
  }

  // Step 2 will transform the layout of a 32-bit register in CUDA in order to match the int4 layout. This has no
  // performance benefit and is purely so that int4 and int8 have the same layout.
  // Pictorially, this does the following:
  // bit 32                                                      0
  //      [elt_3  elt_2  elt_1  elt_0] (each elt occupies 8 bits)
  //
  // And it will rearrange the output 32 bit register to be the following:
  // bit 32                                                      0
  //      [elt_3  elt_1  elt_2  elt_0] (each elt occupies 8 bits)

  if (num_elts % 4 != 0) {
    throw std::invalid_argument("Dimensions of int8 tensor must be a multiple of 4 for register relayout");
  }
  for (size_t base = 0; base < num_elts; base += 4) {
    std::swap(int8_tensor[base + 1], int8_tensor[base + 2]);
  }
}

void AddBiasAndInterleaveInt4sInplace(int8_t* packed_int4_tensor, const size_t num_elts) {
  const int32_t num_bytes = num_elts / 2;

  // Step 1 will be to transform all the int4s to unsigned in order to make the dequantize take as little
  // instructions as possible in the CUDA code.
  for (size_t ii = 0; ii < num_bytes; ++ii) {
    int8_t transformed_packed_int4s = 0;
    int8_t transformed_first_elt =
        (int8_t(packed_int4_tensor[ii] << 4) >> 4) + 8;  // The double shift here is to ensure sign extension
    int8_t transformed_second_elt = (packed_int4_tensor[ii] >> 4) + 8;

    if (!(transformed_first_elt >= 0 && transformed_first_elt <= 15)) {
      throw std::invalid_argument("Illegal result for int4 transform (first elt)");
    }
    if (!(transformed_second_elt >= 0 && transformed_second_elt <= 15)) {
      throw std::invalid_argument("Illegal result for int4 transform (second elt)");
    }

    // We don't need to mask in these ops since everything should be in the range 0-15
    transformed_packed_int4s |= transformed_first_elt;
    transformed_packed_int4s |= (transformed_second_elt << 4);
    packed_int4_tensor[ii] = transformed_packed_int4s;
  }

  // Step 2 will transform the layout of a 32-bit register in CUDA in order to minimize the number of shift & logical
  // instructions That are needed to extract the int4s in the GEMM main loop. Pictorially, the loop below will do the
  // following: Take as input a 32 bit register with layout: bit 32 0
  //      [elt_7  elt_6  elt_5  elt_4  elt_3  elt_2  elt_1  elt_0] (each elt occupies 4 bits)
  //
  // And it will rearrange the output 32 bit register to be the following:
  // bit 32                                                      0
  //      [elt_7  elt_5  elt_3  elt_1  elt_6  elt_4  elt_2  elt_0] (each elt occupies 4 bits)

  if (num_bytes % 4 != 0) {
    throw std::invalid_argument("Dimensions of int4 tensor must be a multiple of 8 for register relayout");
  }
  const size_t num_registers = num_bytes / 4;

  uint32_t* register_ptr = reinterpret_cast<uint32_t*>(packed_int4_tensor);
  for (size_t ii = 0; ii < num_registers; ++ii) {
    const uint32_t current_register = register_ptr[ii];
    uint32_t transformed_register = 0;

    for (int32_t dest_idx = 0; dest_idx < 8; ++dest_idx) {
      const int32_t src_idx = dest_idx < 4 ? 2 * dest_idx : 2 * (dest_idx - 4) + 1;
      const int32_t src_shift = 4 * src_idx;
      const int32_t dest_shift = 4 * dest_idx;

      const uint32_t src_bits = (current_register >> src_shift) & 0xF;
      transformed_register |= (src_bits << dest_shift);
    }
    register_ptr[ii] = transformed_register;
  }
}

void AddBiasAndInterleaveQuantizedTensorInplace(int8_t* tensor, const size_t num_elts, QuantType quant_type) {
  if (quant_type == QuantType::INT8_WEIGHT_ONLY) {
    AddBiasAndInterleaveInt8sInplace(tensor, num_elts);
  } else if (quant_type == QuantType::PACKED_INT4_WEIGHT_ONLY) {
    AddBiasAndInterleaveInt4sInplace(tensor, num_elts);
  } else {
    throw std::invalid_argument("Invalid quantization type for interleaving.");
  }
}

void InterleaveColumnMajorTensor(int8_t* interleaved_quantized_tensor, const int8_t* quantized_tensor,
                                 const std::vector<size_t>& shape, QuantType quant_type, LayoutDetails details) {
  // We only want to run this step for weight only quant.
  if (quant_type != QuantType::PACKED_INT4_WEIGHT_ONLY && quant_type != QuantType::INT8_WEIGHT_ONLY) {
    throw std::invalid_argument("Invalid quant type, only support packed int4 weight only or int8 weight only.");
  }

  if (shape.size() != 2 && shape.size() != 3) {
    throw std::invalid_argument("Shape must be 2-D or 3-D");
  }
  const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
  const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
  const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

  const int32_t BITS_PER_ELT = GetBitsInQuantType(quant_type);
  const int32_t elts_in_int32 = 32 / BITS_PER_ELT;

  const int32_t rows_per_tile = details.rows_per_column_tile;

  if (num_rows % elts_in_int32 != 0) {
    throw std::invalid_argument("The number of rows must be a multiple of " + std::to_string(elts_in_int32) +
                                " but the number of rows is " + std::to_string(num_rows));
  }

  const uint32_t* input_byte_ptr = reinterpret_cast<const uint32_t*>(quantized_tensor);
  uint32_t* output_byte_ptr = reinterpret_cast<uint32_t*>(interleaved_quantized_tensor);

  if (num_rows % rows_per_tile != 0) {
    throw std::invalid_argument("The number of rows must be a multiple of " + std::to_string(rows_per_tile) +
                                " but the number of rows is " + std::to_string(num_rows));
  }

  const int32_t num_vec_rows = num_rows / elts_in_int32;
  const int32_t vec_rows_per_tile = rows_per_tile / elts_in_int32;
  const int32_t interleave = details.columns_interleaved;

  for (int32_t expert = 0; expert < num_experts; ++expert) {
    const int64_t matrix_offset = expert * int64_t(num_vec_rows) * int64_t(num_cols);
    for (int32_t read_col = 0; read_col < num_cols; ++read_col) {
      const int64_t write_col = read_col / interleave;
      for (int32_t base_vec_row = 0; base_vec_row < num_vec_rows; base_vec_row += vec_rows_per_tile) {
        for (int32_t vec_read_row = base_vec_row;
             vec_read_row < std::min(num_vec_rows, base_vec_row + vec_rows_per_tile); ++vec_read_row) {
          const int64_t vec_write_row = interleave * base_vec_row + vec_rows_per_tile * (read_col % interleave) +
                                        vec_read_row % vec_rows_per_tile;

          const int64_t read_offset = matrix_offset + int64_t(read_col) * num_vec_rows + vec_read_row;
          const int64_t write_offset = matrix_offset + int64_t(write_col) * num_vec_rows * interleave + vec_write_row;
          output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
        }
      }
    }
  }
}

void PreprocessWeightsForMixedGemm(int8_t* preprocessed_quantized_weight, const int8_t* row_major_quantized_weight,
                                   const std::vector<size_t>& shape, QuantType quant_type) {
  LayoutDetails details = GetLayoutDetailsForTransform(quant_type);

  if (shape.size() != 2 && shape.size() != 3) {
    throw std::invalid_argument("Shape must be 2-D or 3-D");
  }

  size_t num_elts = 1;
  for (const auto& dim : shape) {
    num_elts *= dim;
  }

  const size_t num_bytes = num_elts * GetBitsInQuantType(quant_type) / 8;

  std::vector<int8_t> src_buf(num_bytes);
  std::vector<int8_t> dst_buf(num_bytes);
  std::copy(row_major_quantized_weight, row_major_quantized_weight + num_bytes, src_buf.begin());

  // Works on row major data, so issue this permutation first.
  if (details.uses_imma_ldsm) {
    const int32_t arch = GetSMVersion();
    PermuteBRowsForMixedGemm(dst_buf.data(), src_buf.data(), shape, quant_type, arch);
    src_buf.swap(dst_buf);
  }

  if (details.layoutB == LayoutDetails::Layout::COLUMN_MAJOR) {
    SubbyteTranspose(dst_buf.data(), src_buf.data(), shape, quant_type);
    src_buf.swap(dst_buf);
  }

  if (details.columns_interleaved > 1) {
    InterleaveColumnMajorTensor(dst_buf.data(), src_buf.data(), shape, quant_type, details);
    src_buf.swap(dst_buf);
  }

  AddBiasAndInterleaveQuantizedTensorInplace(src_buf.data(), num_elts, quant_type);
  std::copy(src_buf.begin(), src_buf.end(), preprocessed_quantized_weight);
}

/*
    Arguments:
      input_weight_ptr - the weight tensor to be quantized. Must be 2-D or 3-D and of type FP16.

      quant_type - the type of the output quantization weight.

    This function does symmetric quantization on 2-D or 3-D tensors. It uses the full int32_t range and assumes the
    zero-point is zero and will automatically construct the scales.

    It always quantizes the last axis of the tensor. For 3-D tensors, it operates in "batched" mode where the tensor is
    viewed as a stack of matrices and a scale is produced for each column of every matrix.

Outputs
    processed_quantized_weight - quantized AND processed weight for GEMM. This MUST be used with the CUTLASS GEMM
    unprocessed_quantized_weight - quantized but unprocessed weights. Useful for reference checking.
    scale_ptr - scales for the quantized weight.

    Note that the returned quantized_weights will be preprocessed in a way to accelerate the mixed type GEMM. The data
    layout may not make sense if printed.

    Shapes:
      quant_type == int8:
        If weight is a [m,n] matrix, quantized_weights will have shape [m,n] and scales of shape [n]
        If weight is a [b,m,n] tensor, unprocessed_quantized_weight will have shape [b,m,n] and scales of shape [b,n]
      quant_type == int4:
        If weight is a [m,n] matrix, quantized_weights will have shape [m, ceil(n/2)] and scales of shape [n]
        If weight is a [b,m,n] tensor, unprocessed_quantized_weight will have shape [b,m, ceil(n/2)] and scales of shape
          [b,n]

      The quantized_weight will be of type torch.int8 and have two int4 values packed in a single byte. This is the
      reason for halving the shape. At the time of writing this code, there was not an elegant way to handle this kind
      of batched quantization using torch's quantized tensors (to the best of the author's knowledge). Scale tensors
      must have a dimension of 1, which breaks the semantics we need for batched weights.
  */

template <typename ComputeType, typename WeightType>
void SymmetricQuantize(int8_t* processed_quantized_weight, int8_t* unprocessed_quantized_weight, ComputeType* scale_ptr,
                       const WeightType* input_weight_ptr, const std::vector<size_t>& shape, QuantType quant_type) {
  if (processed_quantized_weight == nullptr) {
    throw std::invalid_argument("Processed quantized tensor is NULL");
  }
  if (scale_ptr == nullptr) {
    throw std::invalid_argument("Scale output pointer is NULL");
  }
  if (input_weight_ptr == nullptr) {
    throw std::invalid_argument("Input weight pointer is NULL");
  }

  if (shape.size() != 2 && shape.size() != 3) {
    throw std::invalid_argument("Shape must be 2-D or 3-D");
  }
  const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
  const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
  const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

  const int32_t bits_in_type = GetBitsInQuantType(quant_type);
  const int32_t bytes_per_out_col = num_cols * bits_in_type / 8;

  std::vector<int8_t> weight_buf;
  if (unprocessed_quantized_weight == nullptr) {
    weight_buf.resize(num_experts * num_rows * num_cols);
    unprocessed_quantized_weight = weight_buf.data();
  }

  const int32_t input_mat_size = num_rows * num_cols;
  const int32_t quantized_mat_size = num_rows * bytes_per_out_col;
  const float quant_range_scale = 1.f / float(1 << (bits_in_type - 1));

  std::vector<float> per_col_max(num_cols);

  for (int32_t expert = 0; expert < num_experts; ++expert) {
    const WeightType* current_weight = input_weight_ptr + expert * input_mat_size;
    int8_t* current_quantized_weight = unprocessed_quantized_weight + expert * quantized_mat_size;

    // First we find the per column max for this expert weight.
    for (int32_t jj = 0; jj < num_cols; ++jj) {
      per_col_max[jj] = 0.f;
    }

    for (int32_t ii = 0; ii < num_rows; ++ii) {
      const WeightType* current_weight_row = current_weight + ii * num_cols;
      for (int32_t jj = 0; jj < num_cols; ++jj) {
        per_col_max[jj] = std::max(per_col_max[jj], std::abs(float(current_weight_row[jj])));
      }
    }

    // Then, we construct the scales
    ComputeType* current_scales = scale_ptr + expert * num_cols;
    for (int32_t jj = 0; jj < num_cols; ++jj) {
      per_col_max[jj] *= quant_range_scale;
      current_scales[jj] = ComputeType(per_col_max[jj]);
    }

    // Finally, construct the weights.
    for (int32_t ii = 0; ii < num_rows; ++ii) {
      int8_t* current_quantized_weight_row = current_quantized_weight + ii * bytes_per_out_col;
      const WeightType* current_weight_row = current_weight + ii * num_cols;
      for (int32_t jj = 0; jj < bytes_per_out_col; ++jj) {
        if (quant_type == QuantType::INT8_WEIGHT_ONLY) {
          const float col_scale = per_col_max[jj];
          const float weight_elt = float(current_weight_row[jj]);
          const float scaled_weight = round(weight_elt / col_scale);
          const int8_t clipped_weight = int8_t(std::max(-128.f, std::min(127.f, scaled_weight)));
          current_quantized_weight_row[jj] = clipped_weight;
        } else if (quant_type == QuantType::PACKED_INT4_WEIGHT_ONLY) {
          // We will pack two int4 elements per iteration of the inner loop.
          int8_t packed_int4s = 0;
          for (int32_t packed_idx = 0; packed_idx < 2; ++packed_idx) {
            const int32_t input_idx = 2 * jj + packed_idx;
            if (input_idx < num_cols) {
              const float col_scale = per_col_max[input_idx];
              const float weight_elt = float(current_weight_row[input_idx]);
              const float scaled_weight = round(weight_elt / col_scale);
              int32_t int_weight = int32_t(scaled_weight);
              const int8_t clipped_weight = std::max(-8, std::min(7, int_weight));

              // Kill the sign extension bits (hence 0x0F mask) then shift to upper bits
              // if packing the second int4 and or the bits into the final result.
              packed_int4s |= ((clipped_weight & 0x0F) << (4 * packed_idx));
            }
          }
          current_quantized_weight_row[jj] = packed_int4s;
        } else {
          throw std::invalid_argument("Unsupported quantization type");
        }
      }
    }
  }

  PreprocessWeightsForMixedGemm(processed_quantized_weight, unprocessed_quantized_weight, shape, quant_type);
}

template void SymmetricQuantize<half, float>(int8_t*, int8_t*, half*, const float*, const std::vector<size_t>&,
                                             QuantType);

template void SymmetricQuantize<half, half>(int8_t*, int8_t*, half*, const half*, const std::vector<size_t>&,
                                            QuantType);

#ifdef ENABLE_BF16
template void SymmetricQuantize<__nv_bfloat16, __nv_bfloat16>(int8_t*, int8_t*, __nv_bfloat16*, const __nv_bfloat16*,
                                                              const std::vector<size_t>&, QuantType);

template void SymmetricQuantize<__nv_bfloat16, float>(int8_t*, int8_t*, __nv_bfloat16*, const float*,
                                                      const std::vector<size_t>&, QuantType);
#endif

template <typename ComputeType, typename WeightType>
void SymmetricQuantize(int8_t* processed_quantized_weight, ComputeType* scale_ptr, const WeightType* input_weight_ptr,
                       const std::vector<size_t>& shape, QuantType quant_type) {
  SymmetricQuantize(processed_quantized_weight, nullptr, scale_ptr, input_weight_ptr, shape, quant_type);
}

template void SymmetricQuantize<float, float>(int8_t*, float*, const float*, const std::vector<size_t>&, QuantType);

template void SymmetricQuantize<half, float>(int8_t*, half*, const float*, const std::vector<size_t>&, QuantType);

template void SymmetricQuantize<half, half>(int8_t*, half*, const half*, const std::vector<size_t>&, QuantType);

#ifdef ENABLE_BF16
template void SymmetricQuantize<__nv_bfloat16, __nv_bfloat16>(int8_t*, __nv_bfloat16*, const __nv_bfloat16*,
                                                              const std::vector<size_t>&, QuantType);

template void SymmetricQuantize<__nv_bfloat16, half>(int8_t*, __nv_bfloat16*, const half*, const std::vector<size_t>&,
                                                     QuantType);

template void SymmetricQuantize<half, __nv_bfloat16>(int8_t*, half*, const __nv_bfloat16*, const std::vector<size_t>&,
                                                     QuantType);

template void SymmetricQuantize<__nv_bfloat16, float>(int8_t*, __nv_bfloat16*, const float*, const std::vector<size_t>&,
                                                      QuantType);
#endif

}  // namespace nvidia
}  // namespace llm_kernels
