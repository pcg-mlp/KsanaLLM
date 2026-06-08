/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"
#ifdef ENABLE_ACL
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/atb_executor.h"
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/common.h"
#endif

namespace ksana_llm {

class MhcLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  // PreForward: hidden + 3 weights -> 4 outputs (out, mix_h, norm, h_post, h_res)
  Status ForwardPre(const Tensor* hidden, const Tensor* fn_w, const Tensor* scale_w, const Tensor* base_w, Tensor* out,
                    Tensor* mix_h, Tensor* norm_buf, Tensor* h_post, Tensor* h_res,
                    const LayerExecContext& exec_ctx = {});
  // PreForwardWithNorm: same as ForwardPre but with additional norm weight.
  Status ForwardPreWithNorm(const Tensor* hidden, const Tensor* fn_w, const Tensor* scale_w, const Tensor* base_w,
                            const Tensor* norm_w, Tensor* out, Tensor* mix_h, Tensor* norm_buf, Tensor* h_post,
                            Tensor* h_res, const LayerExecContext& exec_ctx = {});
  // PostForward: hidden + h_post + h_res -> output
  Status ForwardPost(const Tensor* hidden, const Tensor* h_post, const Tensor* h_res, Tensor* output,
                     const LayerExecContext& exec_ctx = {});
  // InitialForward: hidden -> output (up-project after embedding)
  Status ForwardInitial(const Tensor* hidden, Tensor* output, const LayerExecContext& exec_ctx = {});
  // FinalForward: hidden + 3 weights -> 3 outputs (out, mix_h, norm)
  Status ForwardFinal(const Tensor* hidden, const Tensor* fn_w, const Tensor* scale_w, const Tensor* base_w,
                      Tensor* out, Tensor* mix_h, Tensor* norm_buf, const LayerExecContext& exec_ctx = {});

  // The num_splits used by mhc_pre kernel
  size_t GetMhcPreNumSplits(int token_num, int hc_hidden_dim);

  static constexpr size_t kMaxNumSplits = 64;

 private:
  template <typename T>
  Status PreForwardT(const Tensor& hidden, const Tensor& fn_w, const Tensor& scale_w, const Tensor& base_w, Tensor& out,
                     Tensor& mix_h, Tensor& norm_buf, Tensor& h_post, Tensor& h_res);
  template <typename T>
  Status PreForwardWithNormT(const Tensor& hidden, const Tensor& fn_w, const Tensor& scale_w, const Tensor& base_w,
                             const Tensor& norm_w, Tensor& out, Tensor& mix_h, Tensor& norm_buf, Tensor& h_post,
                             Tensor& h_res);
  template <typename T>
  Status PostForwardT(const Tensor& hidden, const Tensor& h_post, const Tensor& h_res, Tensor& output);
  template <typename T>
  Status InitialForwardT(const Tensor& hidden, Tensor& output);
  template <typename T>
  Status FinalForwardT(const Tensor& hidden, const Tensor& fn_w, const Tensor& scale_w, const Tensor& base_w,
                       Tensor& out, Tensor& mix_h, Tensor& norm_buf);

 private:
  float rms_norm_eps_;
  uint32_t hc_mult_;
  uint32_t hc_sinkhorn_iters_;
  float hc_eps_;
  size_t hidden_dim_;
  size_t max_token_num_;

#ifdef ENABLE_ACL
  llm_kernels::utils::ATBOperationExecutor atb_op_executor_;
#endif  // ENABLE_ACL
};

}  // namespace ksana_llm
