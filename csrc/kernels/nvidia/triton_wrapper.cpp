/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "kernels/nvidia/triton_wrapper.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "device/nvidia/cuda_utils.h"
#include "utils/logger.h"
#include "utils/string_utils.h"

namespace fs = std::filesystem;

namespace ksana_llm {

#define CUDA_CHECK_WITH_LOG(call, info)                                \
  do {                                                                 \
    CUresult err = call;                                               \
    if (err != CUDA_SUCCESS) {                                         \
      const char* errStr;                                              \
      cuGetErrorString(err, &errStr);                                  \
      KLLM_LOG_ERROR << "CUDA Error: " << errStr << ", info:" << info; \
      abort();                                                         \
    }                                                                  \
  } while (0)

inline size_t CeilDiv(size_t a, size_t b) { return (a + b - 1) / b; }

template <typename T>
inline std::string GetComputeType() {
  if (std::is_same<T, float>::value) {
    return "FP32";
  } else if (std::is_same<T, half>::value) {
    return "FP16";
  } else if (std::is_same<T, nv_bfloat16>::value) {
    return "BF16";
  } else {
    return "Unknown";
  }
}

template <typename T>
inline std::string ConvertToString(const T& value) {
  if (std::is_same<T, int>::value) {
    return std::to_string(value);
  } else if (std::is_same<T, bool>::value) {
    return value ? "True" : "False";
  } else if (std::is_same<T, size_t>::value) {
    return std::to_string(value);
  } else if (std::is_same<T, float>::value) {
    return fmt::format("{:g}", value);
  } else {
    return "Unknown";
  }
}

TritonWrapper::TritonWrapper() {
  InitKernelsDir();

  // 初始化,声明kernel关键key的组合顺序
  kernel_key_map_["fused_moe_kernel"] = {"group_n",      "group_k",      "BLOCK_SIZE_M",      "BLOCK_SIZE_N",
                                         "BLOCK_SIZE_K", "GROUP_SIZE_M", "MUL_ROUTED_WEIGHT", "top_k",
                                         "compute_type", "use_fp8_w8a8", "use_int8_w8a16",    "even_Ks"};
  kernel_key_map_["fused_moe_gptq_awq_kernel"] = {"BLOCK_SIZE_M",      "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
                                                  "MUL_ROUTED_WEIGHT", "top_k",        "compute_type", "has_zp",
                                                  "weight_bits",       "group_size"};
  kernel_key_map_["fused_moe_gptq_int4_fp8_kernel"] = {"BLOCK_SIZE_M", "BLOCK_SIZE_N",      "BLOCK_SIZE_K",
                                                       "GROUP_SIZE_M", "MUL_ROUTED_WEIGHT", "top_k",
                                                       "compute_type", "group_size",        "quant_a_per_tensor"};
  kernel_key_map_["fused_moe_wfp4afp8_kernel"] = {"BLOCK_SIZE_M", "BLOCK_SIZE_N",      "BLOCK_SIZE_K",
                                                  "GROUP_SIZE_M", "MUL_ROUTED_WEIGHT", "top_k",
                                                  "compute_type", "group_size",        "quant_a_per_tensor"};
  kernel_key_map_["fused_batched_moe_kernel"] = {"group_n",      "group_k",       "BLOCK_SIZE_M", "BLOCK_SIZE_N",
                                                 "BLOCK_SIZE_K", "GROUP_SIZE_M",  "top_k",        "compute_type",
                                                 "use_fp8_w8a8", "use_int8_w8a16"};
  kernel_key_map_["mhc_pre_big_fuse_kernel"] = {
      "hidden_size",         "rms_eps",         "mhc_pre_eps", "mhc_sinkhorn_eps",
      "mhc_post_mult_value", "sinkhorn_repeat", "n_splits",    "mhc_mult"};
  kernel_key_map_["mhc_pre_big_fuse_with_norm_kernel"] = {
      "hidden_size",         "rms_eps",         "mhc_pre_eps", "mhc_sinkhorn_eps",
      "mhc_post_mult_value", "sinkhorn_repeat", "n_splits",    "mhc_mult"};
  kernel_key_map_["mhc_head_big_fuse_kernel"] = {"hidden_size", "rms_eps", "mhc_pre_eps", "n_splits", "mhc_mult"};
}

void TritonWrapper::InitKernelsDir() {
  // If KSANA_TRITON_KERNEL_PATH is set, KsanaLLM will load the kernel from
  // ${KSANA_TRITON_KERNEL_PATH}/triton_kernel_files. Otherwise, it will load the kernel from the default path
  // ${HOME}/.cache/KsanaLLM/triton_kernel_files, or from the relative path ./triton_kernel_files.
  const char* env_path = std::getenv("KSANA_TRITON_KERNEL_PATH");
  if (env_path != nullptr) {
    kernels_dir_ = fmt::format("{}/triton_kernel_files/", env_path);
  } else {
    const char* home_dir = std::getenv("HOME");
    if (home_dir != nullptr) {
      kernels_dir_ = std::string(home_dir) + "/.cache/KsanaLLM/triton_kernel_files";
    } else {
      kernels_dir_ = "./triton_kernel_files";
      KLLM_LOG_WARNING << "Cannot get HOME environment variable, using relative path: " << kernels_dir_;
    }
  }
}

void TritonWrapper::SetKernelsDir(const std::string& kernels_dir) { kernels_dir_ = kernels_dir; }

std::optional<std::string> TritonWrapper::ConstructKernelName(const std::string& kernel_base_name,
                                                              const std::unordered_map<std::string, std::string>& map) {
  // 根据关键 key 的组合顺序,获取本地文件的文件名
  if (kernel_key_map_.find(kernel_base_name) == kernel_key_map_.end()) {
    KLLM_LOG_ERROR << fmt::format("Could not get triton kernel [{}]'s config from local file!", kernel_base_name);
    return std::nullopt;
  }
  std::string kernel_name = kernel_base_name;
  for (auto& key : kernel_key_map_[kernel_base_name]) {
    if (map.find(key) == map.end()) {
      KLLM_LOG_ERROR << fmt::format("The necessary key {} is missing from the user's input.", key);
      return std::nullopt;
    }
    kernel_name += fmt::format("_{}_{}", key, map.at(key));
  }
  return kernel_name;
}

std::optional<std::shared_ptr<TritonKernel>> TritonWrapper::LoadTritonKernelJsonFile(const std::string& file_path,
                                                                                     const std::string& kernel_name) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    KLLM_LOG_ERROR << fmt::format("File {} does not exist.", file_path);
    return std::nullopt;
  }
  nlohmann::json json_data;
  file >> json_data;
  std::shared_ptr<TritonKernel> kernel = std::make_shared<TritonKernel>();
  try {
    kernel->shm_size = json_data.at("shm_size");
  } catch (const nlohmann::json::out_of_range& e) {
    std::cout << "Error: Missing key in JSON file." << std::endl;
    return std::nullopt;
  }
  kernel->num_warps = json_data.value("num_warps", 4);
  kernel->num_stages = json_data.value("num_stages", 2);
  kernel->block_x = kernel->num_warps * 32;
  kernel->block_y = 1;
  kernel->block_z = 1;
  kernel->kernel_name = kernel_name;
  return kernel;
}

CUresult TritonWrapper::LoadTritonKernelFromFile(const std::string& file_path,
                                                 std::shared_ptr<TritonKernel>& triton_kernel) {
  fs::path file_full_path = fs::absolute(fs::path(file_path));
  KLLM_LOG_DEBUG << fmt::format("Try to load kernel {} from {}", triton_kernel->kernel_name, file_full_path.c_str());
  CUDA_CHECK_WITH_LOG(cuModuleLoad(&triton_kernel->module, file_full_path.c_str()),
                      fmt::format("cuModuleLoad failed. filename: {} ", file_full_path.c_str()));
  CUDA_CHECK_WITH_LOG(
      cuModuleGetFunction(&triton_kernel->kernel, triton_kernel->module, triton_kernel->kernel_name.c_str()),
      fmt::format("cuModuleGetFunction failed. kernel name: {} ", triton_kernel->kernel_name.c_str()));
  CUDA_CHECK_WITH_LOG(cuFuncSetAttribute(triton_kernel->kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                         triton_kernel->shm_size),
                      fmt::format("cuFuncSetAttribute failed. shm_size: {} ", triton_kernel->shm_size));
  KLLM_LOG_DEBUG << fmt::format("Load triton kernel from {} successd.", file_full_path.c_str());
  return CUDA_SUCCESS;
}

void TritonWrapper::InvokeTritonKernel(const std::string& kernel_base_name, void* args[],
                                       const std::unordered_map<std::string, std::string>& map, size_t grid_x,
                                       size_t grid_y, size_t grid_z, cudaStream_t stream, int block_x, int block_y,
                                       int block_z) {
  const auto kernel_name = ConstructKernelName(kernel_base_name, map);
  if (!kernel_name) {
    KLLM_THROW(fmt::format("Failed to get kernel file name by kernel name {}.", kernel_base_name));
    return;
  }

  const std::string kernel_stream_name = fmt::format("{}_Stream_{}", *kernel_name, *reinterpret_cast<int64_t*>(stream));
  std::shared_ptr<TritonKernel> kernel;
  if (kernel_map_.find(kernel_stream_name) == kernel_map_.end()) {
    std::string config_path = fmt::format("{}/{}.json", kernels_dir_, *kernel_name);
    auto triton_kernel = LoadTritonKernelJsonFile(config_path, kernel_base_name);
    if (!triton_kernel) {
      // 之前这里只 log error 然后 return，会让上层 wrapper 静默跳过 kernel 发射 ——
      // 输出 buffer 不会被改写，模型推理结果错乱但服务能起来，问题极难定位。
      // 改为直接 throw，让缺失 cubin/json 的配置在第一次触发时就明确暴露。
      KLLM_THROW(
          fmt::format("Failed to load triton kernel json file: {}. The triton kernel for "
                      "this configuration was not pre-generated; please re-run the kernel "
                      "creator.sh / re-configure cmake to produce the missing artifact.",
                      config_path));
    }

    std::string cubin_path = fmt::format("{}/{}.cubin", kernels_dir_, *kernel_name);
    CUresult result = LoadTritonKernelFromFile(cubin_path, *triton_kernel);
    if (result != CUDA_SUCCESS) {
      KLLM_THROW(
          fmt::format("Could not load triton kernel cubin: {}. Re-run the kernel "
                      "creator.sh to regenerate it.",
                      cubin_path));
    }
    kernel_map_[kernel_stream_name] = *triton_kernel;
  }
  kernel = kernel_map_[kernel_stream_name];
  if (block_x <= 0) {
    block_x = kernel->block_x;
  }
  if (block_y <= 0) {
    block_y = kernel->block_y;
  }
  if (block_z <= 0) {
    block_z = kernel->block_z;
  }
  CUlaunchAttribute attrs[1];
  attrs[0].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
  attrs[0].value.programmaticStreamSerializationAllowed = llm_kernels::utils::GetEnablePDL() ? 1 : 0;

  CUlaunchConfig config = {};
  config.gridDimX = grid_x;
  config.gridDimY = grid_y;
  config.gridDimZ = grid_z;
  config.blockDimX = block_x;
  config.blockDimY = block_y;
  config.blockDimZ = block_z;
  config.sharedMemBytes = kernel->shm_size;
  config.hStream = stream;
  config.numAttrs = 1;
  config.attrs = attrs;

  CUDA_CHECK_WITH_LOG(
      cuLaunchKernelEx(&config, kernel->kernel, args, nullptr),
      fmt::format("cuLaunchKernelEx failed. kernel_stream_name: {}, grid_x={}, grid_y={}, grid_z={}, "
                  "block_x={}, block_y={}, block_z={}, kernel->shm_size={}",
                  kernel_stream_name, grid_x, grid_y, grid_z, block_x, block_y, block_z, kernel->shm_size));
}

template <typename T>
void TritonWrapper::InvokeFusedMoeKernel(void* a, void* b, void* c, void* a_scale, void* b_scale, void* topk_weights,
                                         void* sorted_token_ids, void* expert_ids, void* num_tokens_post_padded, int n,
                                         int k, int max_num_tokens_padded, int numel, int a_stride0, int a_stride1,
                                         int b_stride0, int b_stride2, int b_stride1, int c_stride1, int c_stride2,
                                         int a_scale_stride0, int a_scale_stride1, int b_scale_stride0,
                                         int b_scale_stride2, int b_scale_stride1, int block_shape0, int block_shape1,
                                         bool mul_routed_weight, int top_k, bool use_fp8_w8a8, bool use_int8_w8a16,
                                         std::unordered_map<std::string, int> config, cudaStream_t stream) {
  std::string kernel_name = "fused_moe_kernel";

  // Triton >= 3.x (NVIDIA backend) appends two implicit CUdeviceptr params at the tail of every
  // compiled kernel's .param list: `global_scratch` and `profile_scratch`. See
  // triton/backends/nvidia/driver.py: _launch(... CUdeviceptr global_scratch, CUdeviceptr profile_scratch, ...).
  // If the C++ wrapper does not supply these, cuLaunchKernel reads past args[] on the stack and
  // libcuda segfaults (observed as CUDA_ERROR_INVALID_VALUE followed by GPF in dmesg). Pass NULL
  // for both; the current fused_moe_kernel does not reference either scratch buffer.
  CUdeviceptr global_scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void* args[] = {
      &a,
      &b,
      &c,
      &a_scale,
      &b_scale,
      &topk_weights,
      &sorted_token_ids,
      &expert_ids,
      &num_tokens_post_padded,
      &n,
      &k,
      &max_num_tokens_padded,
      &numel,
      &global_scratch,
      &profile_scratch,
  };
  std::unordered_map<std::string, std::string> map = {{"group_n", ConvertToString(block_shape0)},
                                                      {"group_k", ConvertToString(block_shape1)},
                                                      {"BLOCK_SIZE_M", ConvertToString(config["block_size_m"])},
                                                      {"BLOCK_SIZE_N", ConvertToString(config["block_size_n"])},
                                                      {"BLOCK_SIZE_K", ConvertToString(config["block_size_k"])},
                                                      {"GROUP_SIZE_M", ConvertToString(config["group_size_m"])},
                                                      {"MUL_ROUTED_WEIGHT", ConvertToString(mul_routed_weight)},
                                                      {"top_k", ConvertToString(top_k)},
                                                      {"compute_type", GetComputeType<T>()},
                                                      {"use_fp8_w8a8", ConvertToString(use_fp8_w8a8)},
                                                      {"use_int8_w8a16", ConvertToString(use_int8_w8a16)}};
  map["even_Ks"] = (k % config["block_size_k"] == 0) ? "True" : "False";
  size_t grid_x = CeilDiv(max_num_tokens_padded, config.at("block_size_m")) * CeilDiv(n, config.at("block_size_n"));
  InvokeTritonKernel(kernel_name, args, map, grid_x, 1, 1, stream);
}
#define INVOKE_FUSED_MOE_KERNEL(T)                                                                                  \
  template void TritonWrapper::InvokeFusedMoeKernel<T>(                                                             \
      void* a, void* b, void* c, void* a_scale, void* b_scale, void* topk_weights, void* sorted_token_ids,          \
      void* expert_ids, void* num_tokens_post_padded, int n, int k, int max_num_tokens_padded, int numel,           \
      int a_stride0, int a_stride1, int b_stride0, int b_stride2, int b_stride1, int c_stride1, int c_stride2,      \
      int a_scale_stride0, int a_scale_stride1, int b_scale_stride0, int b_scale_stride2, int b_scale_stride1,      \
      int block_shape0, int block_shape1, bool mul_routed_weight, int topk, bool use_fp8_w8a8, bool use_int8_w8a16, \
      std::unordered_map<std::string, int> config, cudaStream_t stream)
INVOKE_FUSED_MOE_KERNEL(float);
INVOKE_FUSED_MOE_KERNEL(half);
INVOKE_FUSED_MOE_KERNEL(__nv_bfloat16);
#undef INVOKE_FUSED_MOE_KERNEL

template <typename T>
void TritonWrapper::InvokeFusedMoeGptqAwqKernel(
    void* a, void* b, void* c, void* b_scale, void* b_zp, void* topk_weights, void* sorted_token_ids, void* expert_ids,
    void* num_tokens_post_padded, int n, int k, int max_num_tokens_padded, int numel, int a_stride0, int a_stride1,
    int b_stride0, int b_stride2, int b_stride1, int c_stride1, int c_stride2, int b_scale_stride0, int b_scale_stride2,
    int b_scale_stride1, int b_zp_stride0, int b_zp_stride2, int b_zp_stride1, bool mul_routed_weight, int top_k,
    bool has_zp, int weight_bits, int group_size, std::unordered_map<std::string, int> config, cudaStream_t stream) {
  std::string kernel_name = "fused_moe_gptq_awq_kernel";

  // Triton >= 3.x appends two implicit CUdeviceptr params at the tail of the
  // compiled kernel's parameter list. The kernel does not reference them, so
  // NULL scratch buffers are sufficient.
  CUdeviceptr global_scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void* args[] = {
      &a,
      &b,
      &c,
      &b_scale,
      &b_zp,
      &topk_weights,
      &sorted_token_ids,
      &expert_ids,
      &num_tokens_post_padded,
      &n,
      &k,
      &max_num_tokens_padded,
      &numel,
      &global_scratch,
      &profile_scratch,
  };
  std::unordered_map<std::string, std::string> map = {{"BLOCK_SIZE_M", ConvertToString(config["block_size_m"])},
                                                      {"BLOCK_SIZE_N", ConvertToString(config["block_size_n"])},
                                                      {"BLOCK_SIZE_K", ConvertToString(config["block_size_k"])},
                                                      {"GROUP_SIZE_M", ConvertToString(config["group_size_m"])},
                                                      {"MUL_ROUTED_WEIGHT", ConvertToString(mul_routed_weight)},
                                                      {"top_k", ConvertToString(top_k)},
                                                      {"compute_type", GetComputeType<T>()},
                                                      {"has_zp", ConvertToString(has_zp)},
                                                      {"weight_bits", ConvertToString(weight_bits)},
                                                      {"group_size", ConvertToString(group_size)}};
  size_t grid_x = CeilDiv(max_num_tokens_padded, config.at("block_size_m")) * CeilDiv(n, config.at("block_size_n"));
  InvokeTritonKernel(kernel_name, args, map, grid_x, 1, 1, stream);
}
#define INVOKE_FUSED_MOE_GPTQ_AWQ_KERNEL(T)                                                                    \
  template void TritonWrapper::InvokeFusedMoeGptqAwqKernel<T>(                                                 \
      void* a, void* b, void* c, void* b_scale, void* b_zp, void* topk_weights, void* sorted_token_ids,        \
      void* expert_ids, void* num_tokens_post_padded, int n, int k, int max_num_tokens_padded, int numel,      \
      int a_stride0, int a_stride1, int b_stride0, int b_stride2, int b_stride1, int c_stride1, int c_stride2, \
      int b_scale_stride0, int b_scale_stride2, int b_scale_stride1, int b_zp_stride0, int b_zp_stride2,       \
      int b_zp_stride1, bool mul_routed_weight, int top_k, bool has_zp, int weight_bits, int group_size,       \
      std::unordered_map<std::string, int> config, cudaStream_t stream)
INVOKE_FUSED_MOE_GPTQ_AWQ_KERNEL(float);
INVOKE_FUSED_MOE_GPTQ_AWQ_KERNEL(half);
INVOKE_FUSED_MOE_GPTQ_AWQ_KERNEL(__nv_bfloat16);
#undef INVOKE_FUSED_MOE_GPTQ_AWQ_KERNEL

template <typename T>
void TritonWrapper::InvokeFusedMoeGptqInt4Fp8Kernel(void* a, void* b, void* c, void* a_scale, void* b_scale,
                                                    void* topk_weights, void* sorted_token_ids, void* expert_ids,
                                                    void* num_tokens_post_padded, int n, int k,
                                                    int max_num_tokens_padded, int numel, bool mul_routed_weight,
                                                    int top_k, int group_size, bool quant_a_per_tensor, int num_experts,
                                                    std::unordered_map<std::string, int> config, cudaStream_t stream) {
  std::string kernel_name = "fused_moe_gptq_int4_fp8_kernel";
  (void)num_experts;

  // Triton >= 3.x appends two implicit CUdeviceptr params at the tail of the
  // compiled kernel's parameter list. The kernel does not reference them, so
  // NULL scratch buffers are sufficient.
  CUdeviceptr global_scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void* args[] = {
      &a,
      &b,
      &c,
      &a_scale,
      &b_scale,
      &topk_weights,
      &sorted_token_ids,
      &expert_ids,
      &num_tokens_post_padded,
      &n,
      &k,
      &max_num_tokens_padded,
      &numel,
      &global_scratch,
      &profile_scratch,
  };
  std::unordered_map<std::string, std::string> map = {{"BLOCK_SIZE_M", ConvertToString(config["block_size_m"])},
                                                      {"BLOCK_SIZE_N", ConvertToString(config["block_size_n"])},
                                                      {"BLOCK_SIZE_K", ConvertToString(config["block_size_k"])},
                                                      {"GROUP_SIZE_M", ConvertToString(config["group_size_m"])},
                                                      {"MUL_ROUTED_WEIGHT", ConvertToString(mul_routed_weight)},
                                                      {"top_k", ConvertToString(top_k)},
                                                      {"compute_type", GetComputeType<T>()},
                                                      {"group_size", ConvertToString(group_size)},
                                                      {"quant_a_per_tensor", ConvertToString(quant_a_per_tensor)}};
  size_t grid_x = CeilDiv(max_num_tokens_padded, config.at("block_size_m")) * CeilDiv(n, config.at("block_size_n"));
  InvokeTritonKernel(kernel_name, args, map, grid_x, 1, 1, stream);
}
#define INVOKE_FUSED_MOE_GPTQ_INT4_FP8_KERNEL(T)                                                           \
  template void TritonWrapper::InvokeFusedMoeGptqInt4Fp8Kernel<T>(                                         \
      void* a, void* b, void* c, void* a_scale, void* b_scale, void* topk_weights, void* sorted_token_ids, \
      void* expert_ids, void* num_tokens_post_padded, int n, int k, int max_num_tokens_padded, int numel,  \
      bool mul_routed_weight, int top_k, int group_size, bool quant_a_per_tensor, int num_experts,         \
      std::unordered_map<std::string, int> config, cudaStream_t stream)
INVOKE_FUSED_MOE_GPTQ_INT4_FP8_KERNEL(float);
INVOKE_FUSED_MOE_GPTQ_INT4_FP8_KERNEL(half);
INVOKE_FUSED_MOE_GPTQ_INT4_FP8_KERNEL(__nv_bfloat16);
#undef INVOKE_FUSED_MOE_GPTQ_INT4_FP8_KERNEL

template <typename T>
void TritonWrapper::InvokeFusedMoeWfp4Afp8Kernel(void* a, void* b, void* c, void* a_scale, void* b_scale,
                                                 void* topk_weights, void* sorted_token_ids, void* expert_ids,
                                                 void* num_tokens_post_padded, int n, int k, int max_num_tokens_padded,
                                                 int numel, bool mul_routed_weight, int top_k, int group_size,
                                                 bool quant_a_per_tensor, int num_experts,
                                                 std::unordered_map<std::string, int> config, cudaStream_t stream) {
  std::string kernel_name = "fused_moe_wfp4afp8_kernel";

  // Triton >= 3.x (NVIDIA backend) appends two implicit CUdeviceptr params at the tail of every
  // compiled kernel's .param list: `global_scratch` and `profile_scratch`. The wfp4afp8 cubin is
  // built with newer Triton; without supplying these tails, cuLaunchKernelEx reads past args[]
  // and libcuda returns CUDA_ERROR_INVALID_VALUE. The kernel itself does not reference either
  // scratch buffer, so passing NULL is safe.
  CUdeviceptr global_scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void* args[] = {
      &a,
      &b,
      &c,
      &a_scale,
      &b_scale,
      &topk_weights,
      &sorted_token_ids,
      &expert_ids,
      &num_tokens_post_padded,
      &n,
      &k,
      &max_num_tokens_padded,
      &numel,
      &global_scratch,
      &profile_scratch,
  };
  std::unordered_map<std::string, std::string> map = {{"BLOCK_SIZE_M", ConvertToString(config["block_size_m"])},
                                                      {"BLOCK_SIZE_N", ConvertToString(config["block_size_n"])},
                                                      {"BLOCK_SIZE_K", ConvertToString(config["block_size_k"])},
                                                      {"GROUP_SIZE_M", ConvertToString(config["group_size_m"])},
                                                      {"MUL_ROUTED_WEIGHT", ConvertToString(mul_routed_weight)},
                                                      {"top_k", ConvertToString(top_k)},
                                                      {"compute_type", GetComputeType<T>()},
                                                      {"group_size", ConvertToString(group_size)},
                                                      {"quant_a_per_tensor", ConvertToString(quant_a_per_tensor)}};
  size_t grid_x = CeilDiv(max_num_tokens_padded, config.at("block_size_m")) * CeilDiv(n, config.at("block_size_n"));
  InvokeTritonKernel(kernel_name, args, map, grid_x, 1, 1, stream);
}
#define INVOKE_FUSED_MOE_WFP4AFP8_KERNEL(T)                                                                \
  template void TritonWrapper::InvokeFusedMoeWfp4Afp8Kernel<T>(                                            \
      void* a, void* b, void* c, void* a_scale, void* b_scale, void* topk_weights, void* sorted_token_ids, \
      void* expert_ids, void* num_tokens_post_padded, int n, int k, int max_num_tokens_padded, int numel,  \
      bool mul_routed_weight, int top_k, int group_size, bool quant_a_per_tensor, int num_experts,         \
      std::unordered_map<std::string, int> config, cudaStream_t stream)
INVOKE_FUSED_MOE_WFP4AFP8_KERNEL(float);
INVOKE_FUSED_MOE_WFP4AFP8_KERNEL(half);
INVOKE_FUSED_MOE_WFP4AFP8_KERNEL(__nv_bfloat16);
#undef INVOKE_FUSED_MOE_WFP4AFP8_KERNEL

template <typename T>
void TritonWrapper::InvokeFusedBatchedMoeKernel(void* a, void* b, void* c, void* expert_num_tokens, void* a_scale,
                                                void* b_scale, void* b_zp, int max_num_tokens, int n, int k,
                                                int num_experts, int a_stride0, int a_stride1, int a_stride2,
                                                int b_stride0, int b_stride1, int b_stride2, int c_stride0,
                                                int c_stride1, int c_stride2, int block_shape0, int block_shape1,
                                                int top_k, bool use_fp8_w8a8, bool use_int8_w8a16,
                                                bool per_act_token_quant, std::unordered_map<std::string, int> config,
                                                cudaStream_t stream) {
  std::string kernel_name = "fused_batched_moe_kernel";

  // Compute strides for scale tensors
  int a_scale_stride0 = 0, a_scale_stride1 = 0, a_scale_stride2 = 0;
  int b_scale_stride0 = 0, b_scale_stride1 = 0, b_scale_stride2 = 0;

  KLLM_LOG_DEBUG << fmt::format("Triton Kernel args ptrs: a={:p}, b={:p}, c={:p}, expert_num_tokens={:p}", a, b, c,
                                expert_num_tokens);
  KLLM_LOG_DEBUG << fmt::format("Triton Kernel args int: max_num_tokens={}, k={}, n={}", max_num_tokens, k, n);
  KLLM_LOG_DEBUG << fmt::format("Triton Kernel scales ptrs: as={:p}, bs={:p}, bzp={:p}", a_scale, b_scale, b_zp);
  KLLM_LOG_DEBUG << fmt::format("Triton Kernel strides: a=({},{},{}), b=({},{},{}), c=({},{},{})", a_stride0, a_stride1,
                                a_stride2, b_stride0, b_stride1, b_stride2, c_stride0, c_stride1, c_stride2);
  KLLM_LOG_DEBUG << fmt::format("Triton Kernel scale strides: as=({},{},{}), bs=({},{},{})", a_scale_stride0,
                                a_scale_stride1, a_scale_stride2, b_scale_stride0, b_scale_stride1, b_scale_stride2);

  // TODO(zezhao): 折叠不必要的多余参数，释放寄存器
  void* args[] = {
      &a,                  // param_0: a_ptr
      &b,                  // param_1: b_ptr
      &c,                  // param_2: c_ptr
      &expert_num_tokens,  // param_3: expert_num_tokens
      &max_num_tokens,     // param_4: max_num_tokens (u32)
      &k,                  // param_5: K (u32)
      &n,                  // param_6: N (u32)
      &a_scale,            // param_7: a_scale_ptr
      &b_scale,            // param_8: b_scale_ptr
      &b_zp,               // param_9: b_zp_ptr
      &a_stride0,          // param_10: stride_ae (u64)
      &a_stride1,          // param_11: stride_am (u64)
      &a_stride2,          // param_12: stride_ak (u64)
      &b_stride0,          // param_13: stride_be (u64)
      &b_stride2,          // param_14: stride_bk (u64)
      &b_stride1,          // param_15: stride_bn (u64)
      &c_stride0,          // param_16: stride_ce (u64)
      &c_stride1,          // param_17: stride_cm (u64)
      &c_stride2,          // param_18: stride_cn (u64)
      &a_scale_stride0,    // param_19: stride_ase (u64)
      &a_scale_stride1,    // param_20: stride_asm (u64)
      &a_scale_stride2,    // param_21: stride_ask (u64)
      &b_scale_stride0,    // param_22: stride_bse (u64)
      &b_scale_stride1,    // param_23: stride_bsk (u64)
      &b_scale_stride2,    // param_24: stride_bsn (u64)
  };

  std::unordered_map<std::string, std::string> map = {{"group_n", ConvertToString(block_shape0)},
                                                      {"group_k", ConvertToString(block_shape1)},
                                                      {"BLOCK_SIZE_M", ConvertToString(config["block_size_m"])},
                                                      {"BLOCK_SIZE_N", ConvertToString(config["block_size_n"])},
                                                      {"BLOCK_SIZE_K", ConvertToString(config["block_size_k"])},
                                                      {"GROUP_SIZE_M", ConvertToString(config["group_size_m"])},
                                                      {"top_k", ConvertToString(top_k)},
                                                      {"compute_type", GetComputeType<T>()},
                                                      {"use_fp8_w8a8", ConvertToString(use_fp8_w8a8)},
                                                      {"use_int8_w8a16", ConvertToString(use_int8_w8a16)}};

  size_t grid_x = num_experts;
  size_t grid_y = CeilDiv(max_num_tokens, config.at("block_size_m")) * CeilDiv(n, config.at("block_size_n"));
  InvokeTritonKernel(kernel_name, args, map, grid_x, grid_y, 1, stream);
}

#define INVOKE_FUSED_BATCHED_MOE_KERNEL(T)                                                                           \
  template void TritonWrapper::InvokeFusedBatchedMoeKernel<T>(                                                       \
      void* a, void* b, void* c, void* expert_num_tokens, void* a_scale, void* b_scale, void* b_zp,                  \
      int max_num_tokens, int n, int k, int num_experts, int a_stride0, int a_stride1, int a_stride2, int b_stride0, \
      int b_stride1, int b_stride2, int c_stride0, int c_stride1, int c_stride2, int block_shape0, int block_shape1, \
      int top_k, bool use_fp8_w8a8, bool use_int8_w8a16, bool per_act_token_quant,                                   \
      std::unordered_map<std::string, int> config, cudaStream_t stream)
INVOKE_FUSED_BATCHED_MOE_KERNEL(float);
INVOKE_FUSED_BATCHED_MOE_KERNEL(half);
INVOKE_FUSED_BATCHED_MOE_KERNEL(__nv_bfloat16);
#undef INVOKE_FUSED_BATCHED_MOE_KERNEL

template <typename T>
void TritonWrapper::InvokeMhcPreBigFuseKernel(void* gemm_out_mul, void* gemm_out_sqrsum, void* mhc_scale,
                                              void* mhc_base, void* residual, void* post_mix, void* comb_mix,
                                              void* layer_input, int num_tokens, int hidden_size, float rms_eps,
                                              float mhc_pre_eps, float mhc_sinkhorn_eps, float mhc_post_mult_value,
                                              int sinkhorn_repeat, int n_splits, int mhc_mult, cudaStream_t stream) {
  std::string kernel_name = "mhc_pre_big_fuse_kernel";

  void* args[] = {
      &comb_mix,  &gemm_out_mul, &gemm_out_sqrsum, &layer_input, &mhc_base,
      &mhc_scale, &post_mix,     &residual,        &num_tokens,
  };
  std::unordered_map<std::string, std::string> map = {{"hidden_size", ConvertToString(hidden_size)},
                                                      {"rms_eps", ConvertToString(rms_eps)},
                                                      {"mhc_pre_eps", ConvertToString(mhc_pre_eps)},
                                                      {"mhc_sinkhorn_eps", ConvertToString(mhc_sinkhorn_eps)},
                                                      {"mhc_post_mult_value", ConvertToString(mhc_post_mult_value)},
                                                      {"sinkhorn_repeat", ConvertToString(sinkhorn_repeat)},
                                                      {"n_splits", ConvertToString(n_splits)},
                                                      {"mhc_mult", ConvertToString(mhc_mult)}};
  InvokeTritonKernel(kernel_name, args, map, num_tokens, 1, 1, stream, 96, 1, 1);
}
#define INVOKE_MHC_PRE_BIG_FUSE_KERNEL(T)                                                                           \
  template void TritonWrapper::InvokeMhcPreBigFuseKernel<T>(void*, void*, void*, void*, void*, void*, void*, void*, \
                                                            int, int, float, float, float, float, int, int, int,    \
                                                            cudaStream_t)
INVOKE_MHC_PRE_BIG_FUSE_KERNEL(float);
INVOKE_MHC_PRE_BIG_FUSE_KERNEL(half);
INVOKE_MHC_PRE_BIG_FUSE_KERNEL(__nv_bfloat16);
#undef INVOKE_MHC_PRE_BIG_FUSE_KERNEL

template <typename T>
void TritonWrapper::InvokeMhcPreBigFuseWithNormKernel(void* gemm_out_mul, void* gemm_out_sqrsum, void* mhc_scale,
                                                      void* mhc_base, void* residual, void* norm_weight, void* post_mix,
                                                      void* comb_mix, void* layer_input, int num_tokens,
                                                      int hidden_size, float rms_eps, float mhc_pre_eps,
                                                      float mhc_sinkhorn_eps, float mhc_post_mult_value,
                                                      int sinkhorn_repeat, int n_splits, int mhc_mult,
                                                      cudaStream_t stream) {
  std::string kernel_name = "mhc_pre_big_fuse_with_norm_kernel";

  // TileLang lowers `T.prim_func` params in alphabetical order; mirror that.
  void* args[] = {
      &comb_mix,  &gemm_out_mul, &gemm_out_sqrsum, &layer_input, &mhc_base,
      &mhc_scale, &norm_weight,  &post_mix,        &residual,    &num_tokens,
  };
  std::unordered_map<std::string, std::string> map = {{"hidden_size", ConvertToString(hidden_size)},
                                                      {"rms_eps", ConvertToString(rms_eps)},
                                                      {"mhc_pre_eps", ConvertToString(mhc_pre_eps)},
                                                      {"mhc_sinkhorn_eps", ConvertToString(mhc_sinkhorn_eps)},
                                                      {"mhc_post_mult_value", ConvertToString(mhc_post_mult_value)},
                                                      {"sinkhorn_repeat", ConvertToString(sinkhorn_repeat)},
                                                      {"n_splits", ConvertToString(n_splits)},
                                                      {"mhc_mult", ConvertToString(mhc_mult)}};
  InvokeTritonKernel(kernel_name, args, map, num_tokens, 1, 1, stream, 96, 1, 1);
}
#define INVOKE_MHC_PRE_BIG_FUSE_WITH_NORM_KERNEL(T)                                                                  \
  template void TritonWrapper::InvokeMhcPreBigFuseWithNormKernel<T>(void*, void*, void*, void*, void*, void*, void*, \
                                                                    void*, void*, int, int, float, float, float,     \
                                                                    float, int, int, int, cudaStream_t)
INVOKE_MHC_PRE_BIG_FUSE_WITH_NORM_KERNEL(float);
INVOKE_MHC_PRE_BIG_FUSE_WITH_NORM_KERNEL(half);
INVOKE_MHC_PRE_BIG_FUSE_WITH_NORM_KERNEL(__nv_bfloat16);
#undef INVOKE_MHC_PRE_BIG_FUSE_WITH_NORM_KERNEL

template <typename T>
void TritonWrapper::InvokeMhcHeadBigFuseKernel(void* gemm_out_mul, void* gemm_out_sqrsum, void* mhc_scale,
                                               void* mhc_base, void* residual, void* layer_input, int num_tokens,
                                               int hidden_size, float rms_eps, float mhc_pre_eps, int n_splits,
                                               int mhc_mult, cudaStream_t stream) {
  std::string kernel_name = "mhc_head_big_fuse_kernel";

  // TileLang lowers `T.prim_func` params in alphabetical order; mirror that.
  void* args[] = {
      &gemm_out_mul, &gemm_out_sqrsum, &layer_input, &mhc_base, &mhc_scale, &residual, &num_tokens,
  };
  std::unordered_map<std::string, std::string> map = {{"hidden_size", ConvertToString(hidden_size)},
                                                      {"rms_eps", ConvertToString(rms_eps)},
                                                      {"mhc_pre_eps", ConvertToString(mhc_pre_eps)},
                                                      {"n_splits", ConvertToString(n_splits)},
                                                      {"mhc_mult", ConvertToString(mhc_mult)}};
  InvokeTritonKernel(kernel_name, args, map, num_tokens, 1, 1, stream, 64, 1, 1);
}
#define INVOKE_MHC_HEAD_BIG_FUSE_KERNEL(T)                                                                       \
  template void TritonWrapper::InvokeMhcHeadBigFuseKernel<T>(void*, void*, void*, void*, void*, void*, int, int, \
                                                             float, float, int, int, cudaStream_t)
INVOKE_MHC_HEAD_BIG_FUSE_KERNEL(float);
INVOKE_MHC_HEAD_BIG_FUSE_KERNEL(half);
INVOKE_MHC_HEAD_BIG_FUSE_KERNEL(__nv_bfloat16);
#undef INVOKE_MHC_HEAD_BIG_FUSE_KERNEL

}  // namespace ksana_llm
