/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "models/base/layer_creation_context.h"

namespace ksana_llm {

// Buffers used in mHC.
struct MhcBuffers {
  TensorBuffer* norm_buffer;    // sqr_sum ([t])
  TensorBuffer* mix_h_buffer;   // H': [H'^pre, H'^post, H'^res] ([t, (2 + hc) * hc])
  TensorBuffer* h_post_buffer;  // H^post ([t, hc])
  TensorBuffer* h_res_buffer;   // H^res ([t, hc * hc])
};

class Mhc {
 public:
  Mhc(const std::string& weight_name, const LayerCreationContext& creation_context, MhcBuffers& mhc_buffers,
      const std::string& norm_weight_name = "");

  ~Mhc() = default;

  static Status CreateBuffers(BufferManager* buffer_mgr, const MhcConfig& mhc_config,
                              const RuntimeConfig& runtime_config, MhcBuffers& mhc_buffers);

  // Applied before attn/mlp
  // 1. Compute [H'^pre, H'^post, H'^res] = input[0] * fn
  // 2. Compute [H^pre, H^post, H^res] = [sigmoid(scale * H'^pre / rms + base),
  //                                      2 * sigmoid(scale * H'^post / rms + base),
  //                                      Sinkhorn-Knopp(scale * H'^res / rms + base)]
  // 3. Compute output[0] = H^pre * input[0]
  // input[0]: [t, hc, d]
  // output[0]: [t, d]
  Status PreForward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  // Like PreForward, but fuses the subsequent RMSNorm.
  // output[0] is already normed after this call.
  // Requires norm_weight_name to be set in the constructor.
  Status PreForwardWithNorm(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  // Applied after attn/mlp
  // Compute output[0] = H^res * output[0] + H^post * input[0]
  // input[0]: [t, d]
  // output[0]: [t, hc, d]
  Status PostForward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  // Applied after embedding lookup
  // Expand input[0] along the hc dimention to get output[0]
  // input[0]: [t, d]
  // output[0]: [t, hc, d]
  Status InitialForward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  // Applied before lmhead
  // 1. Compute output[1] = input[0] * fn
  // 2. Compute output[1] = sigmoid(scale * output[1] / rms + base)
  // 3. Compute output[0] = output[1] * input[0]
  // input[0]: [t, hc, d]
  // output[0]: [t, hc]
  // output[1]: [t, hc]
  Status FinalForward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                      Tensor* workspace_tensor = nullptr);

 private:
  std::shared_ptr<Context> context_;
  const MhcConfig& mhc_config_;

  MhcBuffers& mhc_buffers_;

  Tensor fn_weight_;     // [(2 + hc) * hc, hc * d] FP32
  Tensor scale_weight_;  // [3] FP32
  Tensor base_weight_;   // [(2 + hc) * hc] FP32
  Tensor norm_weight_;   // [d] T (RMSNorm weight, only for fused path)

  std::unique_ptr<BaseLayer> mhc_layer_;
};

}  // namespace ksana_llm
