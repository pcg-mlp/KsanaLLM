/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/kda_attention_layer.h"

#include <cstdint>

#include "kernels/nvidia/basic_kernel_wrapper.h"
#include "kernels/nvidia/kda_kernel_wrapper.h"

namespace ksana_llm {
namespace {

struct KdaShape {
  int num_tokens{0};
  int num_sequences{0};
  int num_chunks{0};
  int num_heads{0};
  int token_head_dim{0};
  int qkv_dim{0};
  int max_seq_len{0};
  int num_conv_state_slots{0};
  int conv_state_slot_stride{0};
  int ssm_state_slot_stride{0};
};

struct KdaQkvViews {
  const __nv_bfloat16* q{nullptr};
  const __nv_bfloat16* k{nullptr};
  const __nv_bfloat16* v{nullptr};
};

KdaShape BuildShape(const KdaAttentionInputs& inputs, size_t num_sequences, uint32_t num_heads,
                    uint32_t head_dim, size_t num_chunks = 0, int max_seq_len = 0) {
  KdaShape shape;
  shape.num_tokens = static_cast<int>(inputs.qkv->shape[0]);
  shape.num_sequences = static_cast<int>(num_sequences);
  shape.num_chunks = static_cast<int>(num_chunks);
  shape.num_heads = static_cast<int>(num_heads);
  shape.token_head_dim = static_cast<int>(num_heads * head_dim);
  shape.qkv_dim = 3 * shape.token_head_dim;
  shape.max_seq_len = max_seq_len;
  shape.num_conv_state_slots = static_cast<int>(inputs.conv_state->shape[0]);
  shape.conv_state_slot_stride = inputs.conv_state_slot_stride;
  shape.ssm_state_slot_stride = inputs.ssm_state_slot_stride;
  return shape;
}

KdaConvStrides GetConvStateStrides(const KdaShape& shape) {
  const int slot_stride = shape.conv_state_slot_stride > 0
                              ? shape.conv_state_slot_stride
                              : kKdaConvStateLen * shape.qkv_dim;
  return {slot_stride, 1, shape.qkv_dim};
}

KdaQkvViews RunPrefillConv(const KdaPrefillInputs& inputs, const KdaShape& shape,
                           __nv_bfloat16* conv_qkv, cudaStream_t stream) {
  const KdaAttentionInputs& common = inputs.common;
  const __nv_bfloat16* qkv = common.qkv->GetPtr<__nv_bfloat16>();
  const float* weight = common.conv_weight->GetPtr<float>();
  __nv_bfloat16* conv_state = common.conv_state->GetPtr<__nv_bfloat16>();
  for (int component = 0; component < 3; ++component) {
    KdaCausalConv1dParams conv;
    conv.x = qkv + component * shape.token_head_dim;
    conv.weight = weight + component * shape.token_head_dim * kKdaConvWidth;
    conv.conv_state = conv_state + component * shape.token_head_dim;
    conv.state_indices = common.state_indices->GetPtr<int32_t>();
    conv.has_initial_state = inputs.has_initial_state->GetPtr<bool>();
    conv.cu_seqlens = inputs.cu_seqlens->GetPtr<int32_t>();
    conv.output = conv_qkv + component * shape.num_tokens * shape.token_head_dim;
    conv.num_cache_lines = shape.num_conv_state_slots;
    conv.num_sequences = shape.num_sequences;
    conv.max_seq_len = shape.max_seq_len;
    conv.num_channels = shape.token_head_dim;
    // Packed sequences are located through cu_seqlens; Q/K/V remain
    // interleaved in the projection output and become contiguous here.
    conv.x_strides = {0, 1, shape.qkv_dim};
    conv.state_strides = GetConvStateStrides(shape);
    conv.output_strides = {0, 1, shape.token_head_dim};
    InvokeCausalConv1dFn(conv, stream);
  }
  return {conv_qkv, conv_qkv + shape.num_tokens * shape.token_head_dim,
          conv_qkv + 2 * shape.num_tokens * shape.token_head_dim};
}

void RunDecodeConv(const KdaAttentionInputs& inputs, const KdaShape& shape, __nv_bfloat16* conv_qkv,
                   cudaStream_t stream) {
  KdaCausalConv1dUpdateParams conv;
  conv.mode = KdaForwardMode::kDecode;
  conv.x = inputs.qkv->GetPtr<__nv_bfloat16>();
  conv.weight = inputs.conv_weight->GetPtr<float>();
  conv.conv_state = inputs.conv_state->GetPtr<__nv_bfloat16>();
  conv.state_indices = inputs.state_indices->GetPtr<int32_t>();
  conv.output = conv_qkv;
  conv.batch_size = shape.num_sequences;
  conv.seqlen = 1;
  conv.num_cache_lines = shape.num_conv_state_slots;
  conv.num_channels = shape.qkv_dim;
  conv.x_strides = {shape.qkv_dim, 1, shape.qkv_dim};
  conv.output_strides = conv.x_strides;
  conv.state_strides = GetConvStateStrides(shape);
  InvokeCausalConv1dUpdate(conv, stream);
}

void RunPrefillCore(const KdaPrefillInputs& inputs, const KdaShape& shape, const KdaQkvViews& views,
                    const KdaWorkspaceLayout& scratch, __nv_bfloat16* output,
                    const std::optional<float>& gate_lower_bound, cudaStream_t stream) {
  const KdaAttentionInputs& common = inputs.common;
  KdaChunkParams chunk;
  chunk.q = views.q;
  chunk.k = views.k;
  chunk.v = views.v;
  chunk.forget_gate = common.forget_gate->GetPtr<__nv_bfloat16>();
  chunk.beta = scratch.beta_fp32;
  chunk.a_log = common.a_log->GetPtr<float>();
  chunk.dt_bias = common.dt_bias->GetPtr<float>();
  chunk.gate_lower_bound = gate_lower_bound;
  chunk.cu_seqlens = inputs.cu_seqlens->GetPtr<int32_t>();
  chunk.chunk_indices = inputs.chunk_indices->GetPtr<int32_t>();
  chunk.chunk_offsets = inputs.chunk_offsets->GetPtr<int32_t>();
  chunk.state = common.ssm_state->GetPtr<void>();
  chunk.ssm_dtype = common.ssm_state->dtype;
  chunk.state_indices = common.state_indices->GetPtr<int32_t>();
  chunk.output = output;
  chunk.scratch = scratch;
  chunk.num_tokens = shape.num_tokens;
  chunk.num_sequences = shape.num_sequences;
  chunk.num_chunks = shape.num_chunks;
  chunk.num_heads = shape.num_heads;
  chunk.state_stride = shape.ssm_state_slot_stride;
  InvokeChunkKda(chunk, stream);
}

void RunDecodeCore(const KdaAttentionInputs& inputs, const KdaShape& shape,
                   const __nv_bfloat16* conv_qkv, __nv_bfloat16* output,
                   const std::optional<float>& gate_lower_bound, cudaStream_t stream) {
  KdaPackedDecodeParams decode;
  decode.mixed_qkv = conv_qkv;
  decode.forget_gate = inputs.forget_gate->GetPtr<__nv_bfloat16>();
  decode.beta = inputs.beta->GetPtr<__nv_bfloat16>();
  decode.a_log = inputs.a_log->GetPtr<float>();
  decode.dt_bias = inputs.dt_bias->GetPtr<float>();
  decode.gate_lower_bound = gate_lower_bound;
  decode.output = output;
  decode.state = inputs.ssm_state->GetPtr<void>();
  decode.ssm_dtype = inputs.ssm_state->dtype;
  decode.state_indices = inputs.state_indices->GetPtr<int32_t>();
  decode.batch_size = shape.num_sequences;
  decode.num_heads = shape.num_heads;
  decode.initial_state_stride = shape.ssm_state_slot_stride;
  decode.final_state_stride = shape.ssm_state_slot_stride;
  InvokeFusedRecurrentKdaPackedDecode(decode, stream);
}

void RunOutputNorm(const KdaAttentionInputs& inputs, const KdaShape& shape,
                   const KdaWorkspaceLayout& scratch, __nv_bfloat16* output, float norm_eps,
                   cudaStream_t stream) {
  KdaGatedRmsNormParams norm;
  norm.x = output;
  norm.gate = inputs.output_gate->GetPtr<__nv_bfloat16>();
  norm.weight = inputs.norm_weight->GetPtr<__nv_bfloat16>();
  norm.output = output;
  norm.rstd = scratch.rstd;
  norm.num_rows = shape.num_tokens * shape.num_heads;
  norm.eps = norm_eps;
  InvokeKdaGatedRmsNorm(norm, stream);
}

}  // namespace

Status KdaAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                               std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  const auto params = std::any_cast<const KdaAttentionLayerParameters>(parameters.front());
  num_heads_ = params.num_heads;
  head_dim_ = params.head_dim;
  gate_lower_bound_ = params.gate_lower_bound;
  norm_eps_ = params.norm_eps;
  max_step_token_num_ = runtime_config.max_step_token_num;
  max_batch_size_ = static_cast<size_t>(runtime_config.max_batch_size);
  return Status();
}

size_t KdaAttentionLayer::GetWorkspaceSize() {
  const size_t max_chunks =
      max_batch_size_ + (max_step_token_num_ - max_batch_size_) / static_cast<size_t>(kKdaChunkSize);
  const size_t workspace_size =
      GetKdaWorkspaceSize(max_step_token_num_, num_heads_, max_chunks);
  KLLM_LOG_DEBUG << fmt::format("KdaAttentionLayer requires workspace: {}", workspace_size);
  return workspace_size;
}

Status KdaAttentionLayer::ForwardPrefill(const KdaPrefillInputs& inputs, Tensor* output,
                                         const LayerExecContext& exec_ctx) {
  const size_t num_tokens = inputs.common.qkv->shape[0];
  const size_t num_sequences = inputs.cu_seqlens->shape[0] - 1;
  const size_t num_chunks = inputs.chunk_indices->shape[0];
  output->shape = {num_tokens, num_heads_, head_dim_};
  output->dtype = DataType::TYPE_BF16;
  if (num_tokens == 0) {
    return Status();
  }
  const KdaWorkspaceLayout scratch = BuildKdaWorkspaceLayout(
      /*base=*/workspace_buffer_->GetPtr<void>(), /*num_tokens=*/num_tokens,
      /*num_heads=*/num_heads_, /*num_chunks=*/num_chunks);
  const KdaShape shape =
      BuildShape(/*inputs=*/inputs.common, /*num_sequences=*/num_sequences, /*num_heads=*/num_heads_,
                 /*head_dim=*/head_dim_, /*num_chunks=*/num_chunks, /*max_seq_len=*/inputs.max_seq_len);
  cudaStream_t stream = GetComputeStream(exec_ctx);

  const size_t beta_elements = num_tokens * num_heads_;
  DataToFloat<__nv_bfloat16>(inputs.common.beta->GetPtr<void>(), static_cast<int>(beta_elements),
                             beta_elements, beta_elements, scratch.beta_fp32, stream);
  InvokeSigmoidActivation<float>(scratch.beta_fp32, beta_elements, 1.0f, stream);
  const KdaQkvViews views = RunPrefillConv(inputs, shape, scratch.conv_qkv, stream);
  RunPrefillCore(inputs, shape, views, scratch, output->GetPtr<__nv_bfloat16>(), gate_lower_bound_,
                 stream);
  RunOutputNorm(inputs.common, shape, scratch, output->GetPtr<__nv_bfloat16>(), norm_eps_, stream);
  return Status();
}

Status KdaAttentionLayer::ForwardDecode(const KdaDecodeInputs& inputs, Tensor* output,
                                        const LayerExecContext& exec_ctx) {
  // Decode is one token per sequence, so qkv's leading dimension is both the
  // batch size and the token count. InvokeChunkKda is not on this path, so
  // chunked-state scratch stays empty (num_chunks defaults to 0).
  const size_t num_sequences = inputs.common.qkv->shape[0];
  output->shape = {num_sequences, num_heads_, head_dim_};
  output->dtype = DataType::TYPE_BF16;
  if (num_sequences == 0) {
    return Status();
  }
  const KdaWorkspaceLayout scratch = BuildKdaWorkspaceLayout(
      /*base=*/workspace_buffer_->GetPtr<void>(), /*num_tokens=*/num_sequences,
      /*num_heads=*/num_heads_);
  const KdaShape shape =
      BuildShape(/*inputs=*/inputs.common, /*num_sequences=*/num_sequences, /*num_heads=*/num_heads_,
                 /*head_dim=*/head_dim_);
  cudaStream_t stream = GetComputeStream(exec_ctx);

  RunDecodeConv(inputs.common, shape, scratch.conv_qkv, stream);
  RunDecodeCore(inputs.common, shape, scratch.conv_qkv, output->GetPtr<__nv_bfloat16>(),
                gate_lower_bound_, stream);
  RunOutputNorm(inputs.common, shape, scratch, output->GetPtr<__nv_bfloat16>(), norm_eps_, stream);
  return Status();
}

}  // namespace ksana_llm
