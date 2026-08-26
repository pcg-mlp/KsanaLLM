/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>

#include "device/device_types.h"

namespace ksana_llm {

// KDA dtype contract: activations are __nv_bfloat16; gate parameters (A_log,
// dt_bias), the causal-conv filter and RMSNorm statistics stay in float;
// index/metadata tensors are int32_t, except cu_seqlens which is uint64_t.
// Boolean masks are one byte per element, matching torch.bool.
// The recurrent state is the one exception: it is typed at
// runtime by the state tensor's DataType, so its buffers are passed as void*.
// FP32 is the delta-rule reference; BF16 halves the state pool. Each storage
// dtype is a separate AOT artifact selected by the SSM specialization key.

// Structural constants baked into the AOT Triton artifacts. They are part of
// the call contract and must stay in sync with
// 3rdparty/LLM_kernels/csrc/kernels/nvidia/kda/generate_kda.py.
constexpr int kKdaHeadDim = 128;
constexpr int kKdaChunkSize = 64;
constexpr int kKdaConvWidth = 4;
// A depthwise conv of width W keeps W-1 past tokens per channel in its state.
constexpr int kKdaConvStateLen = kKdaConvWidth - 1;

// Expected gated RMSNorm epsilon. Callers still pass ``eps``; a value other
// than ``kKdaNormEps`` is accepted but logs a warning for now.
constexpr float kKdaNormEps = 1e-5f;

enum class KdaForwardMode : uint8_t {
  kPrefill,
  kDecode,
  kTargetVerifyChain,
  kTargetVerifyTree,
};

// Element strides of a causal-conv tensor viewed as [sequence, channel, token].
// A zero ``seq`` stride marks a packed-varlen tensor where sequences are
// addressed through cu_seqlens instead of a leading batch dimension.
struct KdaConvStrides {
  int seq{0};
  int dim{0};
  int token{0};
};

// Element strides of the depthwise filter viewed as [channel, width]. KDA always
// stores it contiguously as [3*H*D, kKdaConvWidth] and therefore never overrides
// these defaults; the field is kept so that a caller with a non-contiguous filter
// can reuse the causal-conv entry points without changing their signature.
struct KdaConvWeightStrides {
  int dim{kKdaConvWidth};
  int width{1};
};

// Element strides of the target-verify conv checkpoint viewed as
// [cache slot, step, channel, window position].
struct KdaConvWindowStrides {
  int seq{0};
  int step{0};
  int dim{0};
  int window{0};
};

// Element strides of a per-sequence metadata matrix viewed as [sequence, token].
struct KdaRowStrides {
  int seq{0};
  int token{0};
};

// Scratch slices carved out of the caller's KDA workspace buffer. The caller
// resolves it once per forward call with BuildKdaWorkspaceLayout and passes the
// slices it needs to each kernel entry point, so the buffer partitioning is an
// explicit contract instead of an assumption shared between call sites.
// ``conv_qkv`` and ``rstd`` belong to the causal-conv and output-RMSNorm
// stages and are never touched by InvokeChunkKda; the remaining slices are the chunked
// core's scratch. Slices are pairwise non-overlapping except for the
// time-multiplexed solve region (k_norm, akkd, akk) and state region (h, v_new),
// which share one base offset by design.
struct KdaWorkspaceLayout {
  __nv_bfloat16* conv_qkv{nullptr};
  // Chunked prefill consumes sigmoid(beta) in FP32 while the model projection
  // produces BF16. The NVIDIA layer owns this conversion buffer so model code
  // does not depend on CUDA scalar types or backend preprocessing helpers.
  float* beta_fp32{nullptr};
  float* rstd{nullptr};
  __nv_bfloat16* q_norm{nullptr};
  __nv_bfloat16* k_norm{nullptr};
  float* gate_cumsum{nullptr};
  __nv_bfloat16* aqk{nullptr};
  float* akkd{nullptr};
  __nv_bfloat16* akk{nullptr};
  __nv_bfloat16* w{nullptr};
  __nv_bfloat16* u{nullptr};
  __nv_bfloat16* kg{nullptr};
  __nv_bfloat16* h{nullptr};
  __nv_bfloat16* v_new{nullptr};
  size_t bytes{0};
};

// Packed-varlen causal-conv forward (prefill). All strides are in elements.
struct KdaCausalConv1dParams {
  const __nv_bfloat16* x{nullptr};
  const float* weight{nullptr};
  __nv_bfloat16* conv_state{nullptr};
  const int32_t* state_indices{nullptr};
  const bool* has_initial_state{nullptr};
  const uint64_t* cu_seqlens{nullptr};
  __nv_bfloat16* output{nullptr};

  // Number of cache slots backing conv_state, i.e. its leading dimension.
  int num_cache_lines{0};
  int num_sequences{0};
  // Longest sequence in the packed batch; sizes the token grid dimension.
  int max_seq_len{0};
  // Uniform sequence length for the non-varlen ABI slot. Packed-varlen callers
  // leave this at 0 and rely on cu_seqlens instead.
  int seqlen{0};
  int num_channels{0};

  KdaConvStrides x_strides{};
  KdaConvWeightStrides weight_strides{};
  KdaConvStrides state_strides{};
  KdaConvStrides output_strides{};
};

// Causal-conv decode/verify update. ``mode`` selects the decode, verify-chain
// or verify-tree artifact; the verify-only members stay null in decode mode.
struct KdaCausalConv1dUpdateParams {
  KdaForwardMode mode{KdaForwardMode::kDecode};

  const __nv_bfloat16* x{nullptr};
  const float* weight{nullptr};
  __nv_bfloat16* conv_state{nullptr};
  // Upstream speculative-decoding circular-buffer metadata. Unused by the
  // current KDA artifacts; kept so a reused launcher can pass them without
  // changing the Params shape.
  const int32_t* cache_seqlens{nullptr};
  const int32_t* state_indices{nullptr};
  __nv_bfloat16* output{nullptr};

  // Target-verify checkpoint of the conv window, replayed when a draft token is
  // rejected. Null in decode mode.
  __nv_bfloat16* intermediate_conv_window{nullptr};
  const int32_t* intermediate_state_indices{nullptr};
  // Eagle tree topology; only the verify-tree mode reads them. next/sibling are
  // inputs, parent_token is written by the kernel for the recurrent stage.
  const int32_t* retrieve_next_token{nullptr};
  const int32_t* retrieve_next_sibling{nullptr};
  int32_t* retrieve_parent_token{nullptr};

  int batch_size{0};
  // Tokens per sequence: 1 in decode mode, the draft length when verifying.
  int seqlen{0};
  int num_cache_lines{0};
  int num_channels{0};

  KdaConvStrides x_strides{};
  KdaConvWeightStrides weight_strides{};
  KdaConvStrides state_strides{};
  // Cache-slot index vectors are contiguous 1-D tensors.
  int stride_state_indices{1};
  KdaConvWindowStrides intermediate_strides{};
  int stride_intermediate_state_indices{1};
  KdaRowStrides retrieve_next_token_strides{};
  KdaRowStrides retrieve_next_sibling_strides{};
  KdaRowStrides retrieve_parent_token_strides{};
  KdaConvStrides output_strides{};
};

// Chunked prefill core. ``beta`` is FP32 here, unlike the BF16 beta consumed by
// the decode and verify kernels.
struct KdaChunkParams {
  const __nv_bfloat16* q{nullptr};
  const __nv_bfloat16* k{nullptr};
  const __nv_bfloat16* v{nullptr};
  const __nv_bfloat16* forget_gate{nullptr};
  const float* beta{nullptr};
  const float* a_log{nullptr};
  const float* dt_bias{nullptr};
  // Forget-gate activation selected by model configuration. A configured lower
  // bound uses ``g = gate_lower_bound * sigmoid(exp(A_log) * x)``; an unset
  // bound uses ``g = -exp(A_log) * softplus(x)``. The variants are separate AOT
  // artifacts, so an unset optional is not the same thing as a bound of zero.
  std::optional<float> gate_lower_bound{std::nullopt};

  const uint64_t* cu_seqlens{nullptr};
  const int32_t* chunk_indices{nullptr};
  const int32_t* chunk_offsets{nullptr};
  // Recurrent state pool, typed by ``ssm_dtype``.
  void* state{nullptr};
  DataType ssm_dtype{DataType::TYPE_FP32};
  const int32_t* state_indices{nullptr};
  __nv_bfloat16* output{nullptr};

  // Scratch resolved by BuildKdaWorkspaceLayout for this call's shape. Only the
  // chunked-core slices are read; conv_qkv and rstd may stay null.
  KdaWorkspaceLayout scratch{};

  int num_tokens{0};
  int num_sequences{0};
  int num_chunks{0};
  int num_heads{0};
  // Element pitch between request slots. Zero keeps the dense legacy layout.
  int state_stride{0};
};

// One-token-per-sequence decode reading Q/K/V straight out of the packed
// [batch, 3*H*D] conv output, without splitting it first.
struct KdaPackedDecodeParams {
  const __nv_bfloat16* mixed_qkv{nullptr};
  const __nv_bfloat16* forget_gate{nullptr};
  const __nv_bfloat16* beta{nullptr};
  const float* a_log{nullptr};
  const float* dt_bias{nullptr};
  // Safe gate when set, softplus gate otherwise; see KdaChunkParams.
  std::optional<float> gate_lower_bound{std::nullopt};

  __nv_bfloat16* output{nullptr};
  // Recurrent state pool, typed by ``ssm_dtype``. The launcher passes this
  // buffer as both the initial and final state so the decode step updates it in
  // place.
  void* state{nullptr};
  DataType ssm_dtype{DataType::TYPE_FP32};
  const int32_t* state_indices{nullptr};

  int batch_size{0};
  int num_heads{0};
  // Element stride of state_indices across the batch. Contiguous by default.
  int stride_indices{1};
  // Initial/final state may share one block-strided Direct Cache buffer.
  // Zero keeps the dense legacy layout.
  int initial_state_stride{0};
  int final_state_stride{0};
};

// Recurrent decode/verify update with fused sigmoid gating.
struct KdaRecurrentUpdateParams {
  KdaForwardMode mode{KdaForwardMode::kDecode};

  const float* a_log{nullptr};
  const __nv_bfloat16* forget_gate{nullptr};
  const float* dt_bias{nullptr};
  // Safe gate when set, softplus gate otherwise; see KdaChunkParams.
  std::optional<float> gate_lower_bound{std::nullopt};
  const __nv_bfloat16* q{nullptr};
  const __nv_bfloat16* k{nullptr};
  const __nv_bfloat16* v{nullptr};
  const __nv_bfloat16* beta{nullptr};

  __nv_bfloat16* output{nullptr};
  // Recurrent state pool and its verify checkpoints, both typed by ``ssm_dtype``.
  void* state{nullptr};
  DataType ssm_dtype{DataType::TYPE_FP32};
  const int32_t* state_indices{nullptr};
  const uint64_t* cu_seqlens{nullptr};

  // Verify-only per-step state checkpoints and Eagle tree parent links.
  void* intermediate_states{nullptr};
  const int32_t* intermediate_state_indices{nullptr};
  const int32_t* retrieve_parent_token{nullptr};

  // Checkpoint slots per cache line; zero in decode mode.
  int cache_steps{0};
  // Row pitch of the tree metadata matrices, in elements.
  int tree_metadata_stride{0};
  // Token pitch within a tree-metadata row. Contiguous rows use 1.
  int stride_parent_token{1};
  // Token pitch of q/k/v. Zero falls back to the tightly packed H*D pitch.
  int qkv_token_stride{0};

  int num_tokens{0};
  int num_sequences{0};
  int num_heads{0};
};

// Output RMSNorm followed by a sigmoid gate, over [num_rows, kKdaHeadDim].
struct KdaGatedRmsNormParams {
  const __nv_bfloat16* x{nullptr};
  const __nv_bfloat16* gate{nullptr};
  const __nv_bfloat16* weight{nullptr};
  __nv_bfloat16* output{nullptr};
  // Optional bias / residual branch / LayerNorm mean. Null by default: the
  // current AOT artifacts compile with HAS_BIAS=False, HAS_RESIDUAL=False and
  // IS_RMS_NORM=True, so these slots stay unused until those constexprs change.
  const __nv_bfloat16* bias{nullptr};
  const __nv_bfloat16* residual{nullptr};
  __nv_bfloat16* residual_output{nullptr};
  float* mean{nullptr};
  float* rstd{nullptr};

  int num_rows{0};
  // Expected to be ``kKdaNormEps`` (1e-5); other values are accepted with a warning.
  float eps{kKdaNormEps};
};

// Resolve the KDA scratch layout for a given shape. ``base`` points at a buffer
// of at least GetKdaWorkspaceSize() bytes and may be null to compute sizes only.
// ``num_chunks`` sizes the chunked-state scratch (layout.h) consumed by
// InvokeChunkKda. Recurrent decode omits it: that path never reads the slice,
// so the default of 0 leaves it empty.
KdaWorkspaceLayout BuildKdaWorkspaceLayout(void* base, size_t num_tokens, int num_heads, size_t num_chunks = 0);

// Size in bytes of the contiguous scratch buffer backing KdaWorkspaceLayout.
// The returned size covers the whole causal-conv -> KDA core -> gated RMSNorm
// pipeline, not only the chunked core. Callers that also run InvokeChunkKda
// must pass the prefill chunk count; omit it only for a decode-only buffer.
size_t GetKdaWorkspaceSize(size_t num_tokens, int num_heads, size_t num_chunks = 0);

// Standalone causal-conv forward.
void InvokeCausalConv1dFn(KdaCausalConv1dParams& params, cudaStream_t stream);

// Standalone causal-conv decode/verify update.
void InvokeCausalConv1dUpdate(KdaCausalConv1dUpdateParams& params, cudaStream_t stream);

void InvokeChunkKda(KdaChunkParams& params, cudaStream_t stream);

void InvokeFusedRecurrentKdaPackedDecode(KdaPackedDecodeParams& params, cudaStream_t stream);

void InvokeFusedSigmoidGatingDeltaRuleUpdate(KdaRecurrentUpdateParams& params, cudaStream_t stream);

// Accepts ``params.eps``; warns when it is not ``kKdaNormEps``.
void InvokeKdaGatedRmsNorm(KdaGatedRmsNormParams& params, cudaStream_t stream);

}  // namespace ksana_llm
