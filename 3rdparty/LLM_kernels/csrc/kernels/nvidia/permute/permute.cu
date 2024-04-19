/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "permute.h"
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

template<size_t num_dims, typename IndexType>
PermuteKernelParams<num_dims, IndexType>
generatePermuteParams(const size_t* src_dims, const void* src, const size_t* permutation, void* dst, size_t count)
{
    PermuteKernelParams<num_dims, IndexType> params;
    params.src_index_helper = NdIndexOffsetHelper<IndexType, num_dims>(src_dims);
    size_t dst_dims[num_dims];
    for (size_t i = 0; i < num_dims; ++i) {
        dst_dims[i] = src_dims[permutation[i]];
    }
    params.dst_index_helper = NdIndexOffsetHelper<IndexType, num_dims>(dst_dims);
    for (size_t i = 0; i < num_dims; ++i) {
        params.permutation[i] = permutation[i];
    }
    params.src   = src;
    params.dst   = dst;
    params.count = static_cast<IndexType>(count);
    return params;
}

template<size_t num_dims, size_t movement_size, typename IndexType>
__global__ void PermuteKernel(PermuteKernelParams<num_dims, IndexType> params)
{
    using T       = typename std::aligned_storage<movement_size, movement_size>::type;
    const T*  src = reinterpret_cast<const T*>(params.src);
    T*        dst = reinterpret_cast<T*>(params.dst);
    IndexType src_index[num_dims];
    IndexType dst_index[num_dims];
    CUDA_1D_KERNEL_LOOP_T(IndexType, i, params.count)
    {
        params.dst_index_helper.OffsetToNdIndex(i, dst_index);
#pragma unroll
        for (size_t dim = 0; dim < num_dims; ++dim) {
            src_index[params.permutation[dim]] = dst_index[dim];
        }
        IndexType src_offset = params.src_index_helper.NdIndexToOffset(src_index);
        dst[i]               = src[src_offset];
    }
}

template <size_t num_dims, size_t movement_size>
void InvokePermute(void* input, void* output, std::vector<size_t> input_shape, std::vector<size_t> permutation,
                   cudaStream_t& stream) {
    size_t total_size = 1;
    for (size_t& dim : input_shape) {
        total_size *= dim;
    }
    PermuteKernelParams<num_dims, size_t> permute_params = generatePermuteParams<num_dims, size_t>(
        const_cast<const size_t*>(input_shape.data()), const_cast<const void*>(input),
        const_cast<const size_t*>(permutation.data()), output, total_size);

    size_t kDefaultCudaThreadsNumPerBlock = 512;
    PermuteKernel<num_dims, movement_size, size_t>
        <<<BlocksNum4ThreadsNum(permute_params.count), kDefaultCudaThreadsNumPerBlock, 0, stream>>>(permute_params);
}

template
void InvokePermute<4ul, 4ul>(void*, void*, std::vector<size_t>, std::vector<size_t>, cudaStream_t&);
template
void InvokePermute<3ul, 4ul>(void*, void*, std::vector<size_t>, std::vector<size_t>, cudaStream_t&);
template
void InvokePermute<4ul, 2ul>(void*, void*, std::vector<size_t>, std::vector<size_t>, cudaStream_t&);
template
void InvokePermute<3ul, 2ul>(void*, void*, std::vector<size_t>, std::vector<size_t>, cudaStream_t&);

}  // namespace nvidia
}  // namespace llm_kernels
