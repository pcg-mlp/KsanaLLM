/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "device/common_device.h"
#include "utils/status.h"

namespace ksana_llm {

struct TritonKernel {
  int shm_size;
  int block_x;
  int block_y;
  int block_z;
  int num_warps;
  int num_stages;
  std::string kernel_name;
  CUmodule module;
  CUfunction kernel;
};

static thread_local std::unordered_map<std::string, std::shared_ptr<TritonKernel>> kernel_map_;

class TritonWrapper {
 public:
  TritonWrapper();
  ~TritonWrapper() = default;

  void InitKernelsDir();

  void SetKernelsDir(const std::string& kernels_dir);

  template <typename T>
  void InvokeFusedMoeKernel(void* a, void* b, void* c, void* a_scale, void* b_scale, void* topk_weights,
                            void* sorted_token_ids, void* expert_ids, void* num_tokens_post_padded, int n, int k,
                            int max_num_tokens_padded, int numel, int a_stride0, int a_stride1, int b_stride0,
                            int b_stride2, int b_stride1, int c_stride1, int c_stride2, int a_scale_stride0,
                            int a_scale_stride1, int b_scale_stride0, int b_scale_stride2, int b_scale_stride1,
                            int block_shape0, int block_shape1, bool mul_routed_weight, int topk, bool use_fp8_w8a8,
                            bool use_int8_w8a16, std::unordered_map<std::string, int> config, cudaStream_t stream);

  /**
   * @brief 执行融合的MoE GPTQ AWQ内核计算
   *
   * @tparam T 模板类型参数
   * @param a 输入激活值，类型为T
   * @param b 量化权重数据，类型为uint8_t
   * @param c 输出结果，类型为T
   * @param b_scale 权重缩放因子，类型为T
   * @param b_zp 权重零点值，类型为uint8_t
   * @param topk_weights 顶部k个专家的路由权重，类型为float32
   * @param sorted_token_ids 排序后的token ID，类型为int32_t
   * @param expert_ids 专家ID，类型为int32_t
   * @param num_tokens_post_padded 填充后的token数量，类型为int32_t
   */
  template <typename T>
  void InvokeFusedMoeGptqAwqKernel(void* a, void* b, void* c, void* b_scale, void* b_zp, void* topk_weights,
                                   void* sorted_token_ids, void* expert_ids, void* num_tokens_post_padded, int n, int k,
                                   int max_num_tokens_padded, int numel, int a_stride0, int a_stride1, int b_stride0,
                                   int b_stride2, int b_stride1, int c_stride1, int c_stride2, int b_scale_stride0,
                                   int b_scale_stride2, int b_scale_stride1, int b_zp_stride0, int b_zp_stride2,
                                   int b_zp_stride1, bool mul_routed_weight, int top_k, bool has_zp, int weight_bits,
                                   int group_size, std::unordered_map<std::string, int> config, cudaStream_t stream);

  // num_experts 仅用于 profile 日志（打印每专家权重大小 / 总权重大小），不影响 kernel 启动。
  template <typename T>
  void InvokeFusedMoeGptqInt4Fp8Kernel(void* a, void* b, void* c, void* a_scale, void* b_scale, void* topk_weights,
                                       void* sorted_token_ids, void* expert_ids, void* num_tokens_post_padded, int n,
                                       int k, int max_num_tokens_padded, int numel, bool mul_routed_weight, int top_k,
                                       int group_size, bool quant_a_per_tensor, int num_experts,
                                       std::unordered_map<std::string, int> config, cudaStream_t stream);

  // 执行 W=MXFP4 / A=FP8 的 fused MoE kernel：
  //   - b 为 MXFP4 packed uint8，shape [E, N, K/2]
  //   - b_scale 为 E8M0 uint8，shape [E, N, K/group_size] (group_size = 32, MXFP4 标准)
  //   - a 为 FP8 E4M3，a_scale 为 fp32 (per-tensor 标量 或 per-token-group [m, K/128])
  // kernel 内部把 MXFP4 on-the-fly 转 FP8 再做 GEMM。
  // num_experts 仅用于 profile 日志（打印每专家权重大小 / 总权重大小），不影响 kernel 启动。
  template <typename T>
  void InvokeFusedMoeWfp4Afp8Kernel(void* a, void* b, void* c, void* a_scale, void* b_scale, void* topk_weights,
                                    void* sorted_token_ids, void* expert_ids, void* num_tokens_post_padded, int n,
                                    int k, int max_num_tokens_padded, int numel, bool mul_routed_weight, int top_k,
                                    int group_size, bool quant_a_per_tensor, int num_experts,
                                    std::unordered_map<std::string, int> config, cudaStream_t stream);

  /**
   * @brief 执行融合的批量MoE内核计算
   *
   * @tparam T 模板类型参数
   * @param a 输入激活值 [E, max_num_tokens, K]
   * @param b 权重数据 [E, K, N]
   * @param c 输出结果 [E, max_num_tokens, N]
   * @param expert_num_tokens 每个专家的token数量 [E]
   * @param a_scale 激活值缩放因子
   * @param b_scale 权重缩放因子
   * @param max_num_tokens 最大token数量
   * @param n 输出维度
   * @param k 输入维度
   * @param num_experts 专家数量
   * @param top_k 顶部k个专家
   * @param use_fp8_w8a8 是否使用FP8 W8A8量化
   * @param use_int8_w8a16 是否使用INT8 W8A16量化
   * @param per_act_token_quant 是否使用per-token量化
   * @param config 内核配置参数
   * @param stream CUDA流
   */
  template <typename T>
  void InvokeFusedBatchedMoeKernel(void* a, void* b, void* c, void* expert_num_tokens, void* a_scale, void* b_scale,
                                   void* b_zp, int max_num_tokens, int n, int k, int num_experts, int a_stride0,
                                   int a_stride1, int a_stride2, int b_stride0, int b_stride1, int b_stride2,
                                   int c_stride0, int c_stride1, int c_stride2, int block_shape0, int block_shape1,
                                   int top_k, bool use_fp8_w8a8, bool use_int8_w8a16, bool per_act_token_quant,
                                   std::unordered_map<std::string, int> config, cudaStream_t stream);

  template <typename T>
  void InvokeMhcPreBigFuseKernel(void* gemm_out_mul, void* gemm_out_sqrsum, void* mhc_scale, void* mhc_base,
                                 void* residual, void* post_mix, void* comb_mix, void* layer_input, int num_tokens,
                                 int hidden_size, float rms_eps, float mhc_pre_eps, float mhc_sinkhorn_eps,
                                 float mhc_post_mult_value, int sinkhorn_repeat, int n_splits, int mhc_mult,
                                 cudaStream_t stream);

  template <typename T>
  void InvokeMhcPreBigFuseWithNormKernel(void* gemm_out_mul, void* gemm_out_sqrsum, void* mhc_scale, void* mhc_base,
                                         void* residual, void* norm_weight, void* post_mix, void* comb_mix,
                                         void* layer_input, int num_tokens, int hidden_size, float rms_eps,
                                         float mhc_pre_eps, float mhc_sinkhorn_eps, float mhc_post_mult_value,
                                         int sinkhorn_repeat, int n_splits, int mhc_mult, cudaStream_t stream);

  template <typename T>
  void InvokeMhcHeadBigFuseKernel(void* gemm_out_mul, void* gemm_out_sqrsum, void* mhc_scale, void* mhc_base,
                                  void* residual, void* layer_input, int num_tokens, int hidden_size, float rms_eps,
                                  float mhc_pre_eps, int n_splits, int mhc_mult, cudaStream_t stream);

  template <typename T>
  void InvokeHummingFp8Quant(const T* input, float* scale, void* output, int hidden_size, int num_tokens,
                             cudaStream_t stream);

 private:
  std::optional<std::string> ConstructKernelName(const std::string& kernel_base_name,
                                                 const std::unordered_map<std::string, std::string>& map);

  std::optional<std::shared_ptr<TritonKernel>> LoadTritonKernelJsonFile(const std::string& file_path,
                                                                        const std::string& kernel_name);

  CUresult LoadTritonKernelFromFile(const std::string& ptx_file_path, std::shared_ptr<TritonKernel>& triton_kernel);

  void InvokeTritonKernel(const std::string& kernel_base_name, void* args[],
                          const std::unordered_map<std::string, std::string>& map, size_t grid_x, size_t grid_y,
                          size_t grid_z, cudaStream_t stream, int block_x = -1, int block_y = -1, int block_z = -1);

  std::unordered_map<std::string, std::vector<std::string>> kernel_key_map_;

  std::string kernels_dir_;
};

}  // namespace ksana_llm
