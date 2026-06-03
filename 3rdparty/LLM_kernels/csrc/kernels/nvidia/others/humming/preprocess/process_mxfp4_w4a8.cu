#include "humming_preprocess.h"

#include <cuda_runtime.h>
#include <torch/torch.h>

// Include the process_mxfp4_w4a8 CUDA kernel from humming package headers
#include <humming/kernel/process_mxfp4.cuh>

namespace llm_kernels::nvidia::humming {

void process_mxfp4_w4a8_weight(cudaStream_t stream, void* weight, void* weight_scale, float* global_scale_output,
                               int64_t num_experts, int64_t size_n, int64_t size_k, int64_t group_size) {
  // Number of scale groups per expert
  int64_t groups_per_expert = (size_n * size_k) / group_size;
  int64_t total_groups = num_experts * groups_per_expert;

  // Get the CUDA device from the stream
  int device_id;
  cudaGetDevice(&device_id);
  auto options_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device_id);

  // Wrap weight_scale as a [num_experts, groups_per_expert] uint8 tensor
  auto scale_tensor = torch::from_blob(weight_scale, {num_experts, groups_per_expert}, options_u8);

  // Step 1: Compute per-expert scale statistics
  auto scale_max = std::get<0>(scale_tensor.max(/*dim=*/1, /*keepdim=*/true));  // [E, 1]
  auto scale_min = std::get<0>(scale_tensor.min(/*dim=*/1, /*keepdim=*/true));  // [E, 1]
  auto scale_range = (scale_max - scale_min).clamp_max(11);  // max_range_val=11 for float8e4m3 (matches humming 0.1.3)
  auto scale_min_new = scale_max - scale_range;              // [E, 1]

  // Step 2: Compute delta_scale_offsets
  auto delta_scale_offsets = (scale_tensor.maximum(scale_min_new) - scale_tensor).to(torch::kUInt8);

  // Step 3: Normalize weight_scale in-place
  auto normalized_scale = (scale_tensor.maximum(scale_min_new) - scale_min_new).to(torch::kUInt8);

  // W4A8 float8e4m3 path: add 1 to weight_scale (matches humming may_process_fused_e8m0_scale)
  normalized_scale = normalized_scale + 1;

  // CRITICAL: Synchronize PyTorch's default stream before using results on our stream.
  // Libtorch operations above run on the default CUDA stream, but our kernel/memcpy
  // use the passed-in `stream`. Without sync, we'd have a data race.
  cudaDeviceSynchronize();

  cudaMemcpyAsync(weight_scale, normalized_scale.data_ptr(), total_groups * sizeof(uint8_t), cudaMemcpyDeviceToDevice,
                  stream);

  // Step 4: Launch process_mxfp4_w4a8 CUDA kernel to adjust FP4 weight values
  int grid_size = (total_groups + 127) / 128;
  process_mxfp4_w4a8<<<grid_size, 128, 0, stream>>>(reinterpret_cast<uint4*>(weight), reinterpret_cast<uint4*>(weight),
                                                    delta_scale_offsets.data_ptr<uint8_t>(),
                                                    static_cast<uint32_t>(total_groups));

  // Step 5: Compute global_scale = 2^(scale_min_new.float() - 127) / 2
  // (matches humming may_process_fused_e8m0_scale: always divide by 2)
  auto global_scale = torch::pow(2.0f, scale_min_new.squeeze(1).to(torch::kFloat32) - 128.0f);

  // Sync again: torch::pow runs on default stream, copy is on our stream
  cudaDeviceSynchronize();

  // Copy global_scale to output
  cudaMemcpyAsync(global_scale_output, global_scale.data_ptr<float>(), num_experts * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
}

}  // namespace llm_kernels::nvidia::humming
