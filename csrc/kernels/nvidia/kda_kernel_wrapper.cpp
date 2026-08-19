/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "kernels/nvidia/kda_kernel_wrapper.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "kernels/nvidia/triton_wrapper.h"
#include "utils/logger.h"
#include "utils/singleton.h"

namespace ksana_llm {
namespace {

// Sub-chunk width of the intra-chunk solve, i.e. the second tile level inside a
// kKdaChunkSize chunk. Fixed by the AOT artifacts (generate_kda.py SUB_CHUNK_SIZE).
constexpr int kSubChunkSize = 16;

// Structural tile sizes; they select the AOT artifact through its file name and
// must match the values in generate_kda.py.
constexpr int kConvBlockM = 8;
constexpr int kConvBlockN = 256;
constexpr int kNormBlockT = 32;
constexpr int kL2BlockT = 16;
constexpr int kGateBlockS = 32;
constexpr int kIntraBlockH = 1;
constexpr int kInterBlockK = 64;
constexpr int kInterBlockV = 64;
constexpr int kStateBlockV = 32;
constexpr int kRecurrentBlockK = 128;
constexpr int kRecurrentBlockV = 32;

// Every workspace slice starts on a 256-byte boundary so that each scratch
// tensor is aligned for the widest vectorized access the kernels emit.
constexpr size_t kWorkspaceAlignment = 256;

// Chunk-count buckets of the state kernel's NTB specialization; the artifact is
// generated for NTB in {0, 1, 2} and picked by the chunk count at launch time.
constexpr int kStateSmallChunkLimit = 32;
constexpr int kStateMediumChunkLimit = 128;

// Above this many chunk-head blocks the fully fused inter/solve kernel loses to
// the token-parallel intra kernel followed by the split solve and recompute.
constexpr int kFusedInterSolveBlockLimit = 256;

// KDA AOT artifacts are generated with USE_GDC=False, so they do not contain
// the device-side protocol required by programmatic dependent launch.
constexpr bool kKdaAllowPdl = false;

// L2 normalization epsilon of the Q/K pre-normalization. Unlike the output
// RMSNorm epsilon this one is an internal detail of the KDA core and is fixed by
// the reference implementation (generate_kda.py L2_EPS).
float kL2Eps = 1e-6f;

// log2(e). The gate cumsum kernel accumulates in base 2 (USE_EXP2) so that the
// state kernel can use the faster exp2 instruction; scaling the gate by log2(e)
// keeps the result identical to a natural-log accumulation.
float kLog2E = 1.4426950216293335f;

// 1/sqrt(kKdaHeadDim), the Q@K^T scale shared by the chunked core, the packed
// decode and the recurrent update. std::sqrt only becomes constexpr in C++26, so
// this cannot be spelled constexpr portably; the argument is a compile-time
// constant, so the compiler still folds the whole expression and no square root
// is evaluated at runtime.
float kAttentionScale = 1.0f / std::sqrt(static_cast<float>(kKdaHeadDim));

// Softplus parameters of the fused sigmoid gating, used by the gate variant
// without a lower bound: beta stays at the plain softplus and beyond the
// threshold the kernel falls back to the identity to avoid overflow
// (generate_kda.py SOFTPLUS_BETA / SOFTPLUS_THRESHOLD).
float kSoftplusBeta = 1.0f;
float kSoftplusThreshold = 20.0f;

// Triton >= 3.x appends two implicit CUdeviceptr params (global_scratch,
// profile_scratch) to every compiled kernel's .param list. None of the KDA
// artifacts reference them, so a single shared zero is safe.
CUdeviceptr kTritonScratchZero = 0;

#define KDA_TRITON_ARGS(...) __VA_ARGS__, &kTritonScratchZero, &kTritonScratchZero

size_t CeilDiv(size_t value, size_t divisor) { return (value + divisor - 1) / divisor; }

int NextPowerOfTwo(int value) {
  int result = 1;
  while (result < value) {
    result <<= 1;
  }
  return result;
}

bool IsVerifyMode(KdaForwardMode mode) {
  return mode == KdaForwardMode::kTargetVerifyChain || mode == KdaForwardMode::kTargetVerifyTree;
}

size_t AlignWorkspace(size_t bytes) {
  return (bytes + kWorkspaceAlignment - 1) & ~(kWorkspaceAlignment - 1);
}

// Carve ``elements`` of type T out of the workspace and advance ``offset`` past
// the aligned slice. A null base only accumulates the required size.
template <typename T>
T* TakeWorkspaceSlice(void* base, size_t& offset, size_t elements) {
  T* pointer = base == nullptr ? nullptr : reinterpret_cast<T*>(static_cast<uint8_t*>(base) + offset);
  offset += AlignWorkspace(elements * sizeof(T));
  return pointer;
}

// Artifact token of the recurrent-state dtype; see generate_kda.py's
// SSM_DTYPE_TOKENS.
std::string GetSsmDtypeName(DataType dtype) {
  return dtype == DataType::TYPE_BF16 ? "bf16" : "fp32";
}

std::string GetModeName(KdaForwardMode mode) {
  if (mode == KdaForwardMode::kTargetVerifyTree) {
    return "verify_tree";
  }
  return mode == KdaForwardMode::kTargetVerifyChain ? "verify_chain" : "decode";
}

}  // namespace

KdaWorkspaceLayout BuildKdaWorkspaceLayout(void* base, size_t num_tokens, int num_heads, size_t num_chunks) {
  KdaWorkspaceLayout layout;
  const size_t token_heads = num_tokens * num_heads;
  const size_t token_head_elements = token_heads * kKdaHeadDim;
  const size_t chunk_matrix_elements = token_heads * kKdaChunkSize;
  const size_t sub_chunk_matrix_elements = token_heads * kSubChunkSize;

  size_t offset = 0;
  layout.conv_qkv = TakeWorkspaceSlice<__nv_bfloat16>(base, offset, 3 * token_head_elements);
  layout.beta_fp32 = TakeWorkspaceSlice<float>(base, offset, token_heads);
  layout.rstd = TakeWorkspaceSlice<float>(base, offset, token_heads);
  layout.q_norm = TakeWorkspaceSlice<__nv_bfloat16>(base, offset, token_head_elements);
  layout.gate_cumsum = TakeWorkspaceSlice<float>(base, offset, token_head_elements);
  layout.aqk = TakeWorkspaceSlice<__nv_bfloat16>(base, offset, chunk_matrix_elements);
  layout.w = TakeWorkspaceSlice<__nv_bfloat16>(base, offset, token_head_elements);
  layout.u = TakeWorkspaceSlice<__nv_bfloat16>(base, offset, token_head_elements);
  layout.kg = TakeWorkspaceSlice<__nv_bfloat16>(base, offset, token_head_elements);

  // The solve stage and the state stage never run concurrently, so their
  // scratch is time-multiplexed over the same region.
  const size_t shared_offset = offset;
  size_t solve_offset = shared_offset;
  layout.k_norm = TakeWorkspaceSlice<__nv_bfloat16>(base, solve_offset, token_head_elements);
  layout.akkd = TakeWorkspaceSlice<float>(base, solve_offset, sub_chunk_matrix_elements);
  layout.akk = TakeWorkspaceSlice<__nv_bfloat16>(base, solve_offset, chunk_matrix_elements);

  size_t state_offset = shared_offset;
  layout.h = TakeWorkspaceSlice<__nv_bfloat16>(base, state_offset,
                                            num_chunks * num_heads * kKdaHeadDim * kKdaHeadDim);
  layout.v_new = TakeWorkspaceSlice<__nv_bfloat16>(base, state_offset, token_head_elements);

  layout.bytes = AlignWorkspace(std::max(solve_offset, state_offset));
  return layout;
}

size_t GetKdaWorkspaceSize(size_t num_tokens, int num_heads, size_t num_chunks) {
  return BuildKdaWorkspaceLayout(nullptr, num_tokens, num_heads, num_chunks).bytes;
}

void InvokeCausalConv1dFn(KdaCausalConv1dParams& params, cudaStream_t stream) {
  void* args[] = {KDA_TRITON_ARGS(
      &params.x, &params.weight, &params.conv_state, &params.state_indices, &params.has_initial_state,
      &params.cu_seqlens, &params.output, &params.seqlen, &params.num_cache_lines, &params.x_strides.seq,
      &params.x_strides.dim, &params.x_strides.token, &params.weight_strides.dim, &params.weight_strides.width,
      &params.state_strides.seq, &params.state_strides.dim, &params.state_strides.token,
      &params.output_strides.seq, &params.output_strides.dim, &params.output_strides.token)};
  Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
      "causal_conv1d_fwd_kernel", args,
      {{"D", std::to_string(params.num_channels)}, {"W", std::to_string(kKdaConvWidth)}}, params.num_sequences,
      CeilDiv(params.max_seq_len, kConvBlockM), CeilDiv(params.num_channels, kConvBlockN), stream, kKdaAllowPdl);
}

void InvokeCausalConv1dUpdate(KdaCausalConv1dUpdateParams& params, cudaStream_t stream) {
  const bool tree = params.mode == KdaForwardMode::kTargetVerifyTree;

  void* args[] = {KDA_TRITON_ARGS(
      &params.x, &params.weight, &params.conv_state, &params.cache_seqlens, &params.state_indices,
      &params.intermediate_conv_window, &params.intermediate_state_indices,
      &params.retrieve_next_token, &params.retrieve_next_sibling, &params.retrieve_parent_token, &params.output,
      &params.batch_size, &params.seqlen, &params.num_cache_lines, &params.x_strides.seq, &params.x_strides.dim,
      &params.x_strides.token, &params.weight_strides.dim, &params.weight_strides.width,
      &params.state_strides.seq, &params.state_strides.dim, &params.state_strides.token,
      &params.stride_state_indices, &params.intermediate_strides.seq, &params.intermediate_strides.step,
      &params.intermediate_strides.dim, &params.intermediate_strides.window,
      &params.stride_intermediate_state_indices, &params.retrieve_next_token_strides.seq,
      &params.retrieve_next_token_strides.token, &params.retrieve_next_sibling_strides.seq,
      &params.retrieve_next_sibling_strides.token, &params.retrieve_parent_token_strides.seq,
      &params.retrieve_parent_token_strides.token, &params.output_strides.seq, &params.output_strides.dim,
      &params.output_strides.token)};
  Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
      "causal_conv1d_update_kernel", args,
      {{"D", std::to_string(params.num_channels)},
       {"W", std::to_string(kKdaConvWidth)},
       {"MODE", GetModeName(params.mode)},
       // Only the tree walk needs a power-of-two token tile; the linear modes
       // step through tokens one by one.
       {"NP2", std::to_string(tree ? NextPowerOfTwo(params.seqlen) : 1)}},
      params.batch_size, CeilDiv(params.num_channels, kConvBlockN), 1, stream, kKdaAllowPdl);
}

void InvokeChunkKda(KdaChunkParams& params, cudaStream_t stream) {
  KdaWorkspaceLayout& scratch = params.scratch;
  int token_rows = params.num_tokens * params.num_heads;

  {
    void* args[] = {KDA_TRITON_ARGS(&params.q, &scratch.q_norm, &kL2Eps, &token_rows)};
    Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
        "l2norm_fwd_kernel", args,
        {{"D", std::to_string(kKdaHeadDim)}, {"BT", std::to_string(kL2BlockT)}},
        CeilDiv(token_rows, kL2BlockT), 1, 1, stream, kKdaAllowPdl);
  }
  {
    void* args[] = {KDA_TRITON_ARGS(&params.k, &scratch.k_norm, &kL2Eps, &token_rows)};
    Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
        "l2norm_fwd_kernel", args,
        {{"D", std::to_string(kKdaHeadDim)}, {"BT", std::to_string(kL2BlockT)}},
        CeilDiv(token_rows, kL2BlockT), 1, 1, stream, kKdaAllowPdl);
  }
  {
    // LB picks the gate activation. The softplus artifact compiles
    // ``lower_bound=None``, which Triton folds away, so only the safe-gate
    // artifact takes the bound as a runtime argument.
    const std::unordered_map<std::string, std::string> key = {
        {"H", std::to_string(params.num_heads)},
        {"BS", std::to_string(kGateBlockS)},
        {"LB", params.gate_lower_bound.has_value() ? "1" : "0"}};
    const size_t grid_x = CeilDiv(kKdaHeadDim, kGateBlockS);
    if (params.gate_lower_bound.has_value()) {
      float lower_bound = *params.gate_lower_bound;
      void* args[] = {KDA_TRITON_ARGS(&params.forget_gate, &params.a_log, &params.dt_bias,
                                            &scratch.gate_cumsum, &kLog2E, &params.cu_seqlens,
                                            &params.chunk_indices, &lower_bound, &params.num_tokens)};
      Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
          "kda_gate_chunk_cumsum_vector_kernel", args, key, grid_x, params.num_chunks, params.num_heads, stream,
          kKdaAllowPdl);
    } else {
      void* args[] = {KDA_TRITON_ARGS(&params.forget_gate, &params.a_log, &params.dt_bias,
                                            &scratch.gate_cumsum, &kLog2E, &params.cu_seqlens,
                                            &params.chunk_indices, &params.num_tokens)};
      Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
          "kda_gate_chunk_cumsum_vector_kernel", args, key, grid_x, params.num_chunks, params.num_heads, stream,
          kKdaAllowPdl);
    }
  }

  const bool small_grid = params.num_chunks * params.num_heads <= kFusedInterSolveBlockLimit;
  const size_t chunk_matrix_bytes =
      static_cast<size_t>(params.num_tokens) * params.num_heads * kKdaChunkSize * sizeof(__nv_bfloat16);
  if (small_grid) {
    // The fused kernel accumulates the intra-chunk scores in place, so the
    // out-of-range tail of the last chunk must start at zero.
    cudaMemsetAsync(scratch.aqk, 0, chunk_matrix_bytes, stream);
    const bool fuse_recompute = true;
    const bool fuse_diagonal = true;
    void* args[] = {KDA_TRITON_ARGS(
        &scratch.q_norm, &scratch.k_norm, &scratch.gate_cumsum, &params.beta, &scratch.aqk, &scratch.akkd,
        &scratch.akk, &kAttentionScale, &params.v, &scratch.w, &scratch.u, &scratch.kg, &params.cu_seqlens,
        &params.chunk_indices, &params.num_tokens)};
    Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
        "chunk_kda_fwd_kernel_inter_solve_fused", args,
        {{"H", std::to_string(params.num_heads)},
         {"FR", std::to_string(fuse_recompute)},
         {"FD", std::to_string(fuse_diagonal)},
         {"BK", std::to_string(kInterBlockK)},
         {"BV", std::to_string(kInterBlockV)}},
        params.num_chunks, params.num_heads, 1, stream, kKdaAllowPdl);
  } else {
    {
      void* args[] = {KDA_TRITON_ARGS(&scratch.q_norm, &scratch.k_norm, &scratch.gate_cumsum, &params.beta,
                                            &scratch.aqk, &scratch.akkd, &kAttentionScale, &params.cu_seqlens,
                                            &params.num_sequences, &params.num_tokens)};
      Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
          "chunk_kda_fwd_kernel_intra_token_parallel", args,
          {{"H", std::to_string(params.num_heads)}, {"BH", std::to_string(kIntraBlockH)}}, params.num_tokens,
          CeilDiv(params.num_heads, kIntraBlockH), 1, stream, kKdaAllowPdl);
    }
    // The unfused solve accumulates into akk, which therefore needs a zeroed
    // tail for the same reason as aqk above.
    cudaMemsetAsync(scratch.akk, 0, chunk_matrix_bytes, stream);
    {
      // The diagonal blocks and w/u already come from the two neighbouring
      // launches, so this configuration only solves the inter-chunk system.
      const bool fuse_recompute = false;
      const bool fuse_diagonal = false;
      void* args[] = {KDA_TRITON_ARGS(
          &scratch.q_norm, &scratch.k_norm, &scratch.gate_cumsum, &params.beta, &scratch.aqk, &scratch.akkd,
          &scratch.akk, &kAttentionScale, &params.v, &scratch.w, &scratch.u, &scratch.kg, &params.cu_seqlens,
          &params.chunk_indices, &params.num_tokens)};
      Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
          "chunk_kda_fwd_kernel_inter_solve_fused", args,
          {{"H", std::to_string(params.num_heads)},
           {"FR", std::to_string(fuse_recompute)},
           {"FD", std::to_string(fuse_diagonal)},
           {"BK", std::to_string(kInterBlockK)},
           {"BV", std::to_string(kInterBlockV)}},
          params.num_chunks, params.num_heads, 1, stream, kKdaAllowPdl);
    }
    {
      void* args[] = {KDA_TRITON_ARGS(&scratch.k_norm, &scratch.kg, &params.v, &params.beta, &scratch.w,
                                            &scratch.u, &scratch.akk, &scratch.gate_cumsum, &params.cu_seqlens,
                                            &params.chunk_indices, &params.num_tokens)};
      Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
          "recompute_w_u_fwd_kernel", args,
          {{"H", std::to_string(params.num_heads)},
           {"BK", std::to_string(kInterBlockK)},
           {"BV", std::to_string(kInterBlockV)}},
          params.num_chunks, params.num_heads, 1, stream, kKdaAllowPdl);
    }
  }

  {
    // Scalars must live in a local so the args array can hold a stable address.
    int state_stride = params.state_stride > 0 ? params.state_stride : params.num_heads * kKdaHeadDim * kKdaHeadDim;
    const int chunk_bucket = params.num_chunks <= kStateSmallChunkLimit
                                 ? 0
                                 : (params.num_chunks <= kStateMediumChunkLimit ? 1 : 2);
    void* args[] = {KDA_TRITON_ARGS(&scratch.kg, &scratch.u, &scratch.w, &scratch.v_new,
                                          &scratch.gate_cumsum, &scratch.h, &params.state, &params.state_indices,
                                          &state_stride, &params.cu_seqlens, &params.chunk_offsets,
                                          &params.num_tokens)};
    Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", args,
        {{"H", std::to_string(params.num_heads)},
         {"NTB", std::to_string(chunk_bucket)},
         {"BV", std::to_string(kStateBlockV)},
         {"SSM", GetSsmDtypeName(params.ssm_dtype)}},
        CeilDiv(kKdaHeadDim, kStateBlockV), params.num_sequences * params.num_heads, 1, stream, kKdaAllowPdl);
  }
  {
    void* args[] = {KDA_TRITON_ARGS(&scratch.q_norm, &scratch.v_new, &scratch.gate_cumsum, &scratch.h,
                                          &params.output, &scratch.aqk, &params.cu_seqlens, &params.chunk_indices,
                                          &kAttentionScale, &params.num_tokens)};
    Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
        "chunk_gla_fwd_kernel_o", args,
        {{"H", std::to_string(params.num_heads)},
         {"BK", std::to_string(kInterBlockK)},
         {"BV", std::to_string(kInterBlockV)}},
        CeilDiv(kKdaHeadDim, kInterBlockV), params.num_chunks, params.num_heads, stream, kKdaAllowPdl);
  }
}

void InvokeFusedRecurrentKdaPackedDecode(KdaPackedDecodeParams& params, cudaStream_t stream) {
  const int token_head_dim = params.num_heads * kKdaHeadDim;
  // Q, K and V are the three consecutive [H*D] halves of one packed conv row.
  int stride_mixed_qkv = 3 * token_head_dim;
  int stride_forget_gate = token_head_dim;
  int stride_beta = params.num_heads;
  const int dense_state_stride = params.num_heads * kKdaHeadDim * kKdaHeadDim;
  // Scalars must live in a local so the args array can hold a stable address.
  int initial_state_stride = params.initial_state_stride > 0 ? params.initial_state_stride : dense_state_stride;
  int final_state_stride = params.final_state_stride > 0 ? params.final_state_stride : dense_state_stride;
  // The bound stays in the ABI of both gate specializations; the softplus
  // artifact simply ignores it.
  float lower_bound = params.gate_lower_bound.value_or(0.0f);
  // h0 and ht are the same buffer: the decode step updates the state in place.
  void* args[] = {KDA_TRITON_ARGS(
      &params.mixed_qkv, &params.forget_gate, &params.beta, &params.a_log, &params.dt_bias, &params.output,
      &params.state, &params.state, &params.state_indices, &kAttentionScale, &lower_bound, &stride_mixed_qkv,
      &stride_forget_gate, &stride_beta, &initial_state_stride, &final_state_stride, &params.stride_indices)};
  Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
      "fused_recurrent_kda_packed_decode_kernel", args,
      {{"H", std::to_string(params.num_heads)},
       {"BK", std::to_string(kRecurrentBlockK)},
       {"BV", std::to_string(kRecurrentBlockV)},
       // LB picks the gate activation.
       {"LB", params.gate_lower_bound.has_value() ? "1" : "0"},
       {"SSM", GetSsmDtypeName(params.ssm_dtype)}},
      CeilDiv(kKdaHeadDim, kRecurrentBlockV), params.batch_size * params.num_heads, 1, stream, kKdaAllowPdl);
}

void InvokeFusedSigmoidGatingDeltaRuleUpdate(KdaRecurrentUpdateParams& params, cudaStream_t stream) {
  const bool verify = IsVerifyMode(params.mode);
  const bool tree = params.mode == KdaForwardMode::kTargetVerifyTree;
  const int token_head_dim = params.num_heads * kKdaHeadDim;
  int stride_forget_gate = token_head_dim;
  // Verify reads Q/K/V out of the interleaved [T, 3*H*D] conv output, decode out
  // of a tightly packed [T, H*D] tensor.
  int stride_qkv = params.qkv_token_stride > 0 ? params.qkv_token_stride : token_head_dim;
  int stride_beta = params.num_heads;
  // The bound stays in the ABI of both gate specializations; the softplus
  // artifact simply ignores it.
  float lower_bound = params.gate_lower_bound.value_or(0.0f);
  const int logical_verify_tokens = verify ? params.num_tokens / params.num_sequences : 1;
  std::unordered_map<std::string, std::string> key = {
      {"H", std::to_string(params.num_heads)},
      {"MODE", GetModeName(params.mode)},
      // Only the tree walk needs a power-of-two token tile.
      {"NP2", std::to_string(tree ? NextPowerOfTwo(logical_verify_tokens) : 1)},
      {"BK", std::to_string(kRecurrentBlockK)},
      {"BV", std::to_string(kRecurrentBlockV)},
      // LB picks the gate activation.
      {"LB", params.gate_lower_bound.has_value() ? "1" : "0"},
      {"SSM", GetSsmDtypeName(params.ssm_dtype)}};
  const size_t grid_x = CeilDiv(kKdaHeadDim, kRecurrentBlockK);
  const size_t grid_y = CeilDiv(kKdaHeadDim, kRecurrentBlockV);
  const size_t grid_z = params.num_sequences * params.num_heads;

  // The three artifacts differ in their argument list, not only in their
  // specialization key: decode drops the intermediate-state checkpoints and
  // verify-chain additionally drops the tree parent links.
  if (!verify) {
    void* args[] = {KDA_TRITON_ARGS(
        &params.a_log, &params.forget_gate, &params.dt_bias, &kSoftplusBeta, &kSoftplusThreshold, &lower_bound,
        &params.q, &params.k, &params.v, &params.beta, &params.output, &params.state, &params.state_indices,
        &params.cu_seqlens, &params.cache_steps, &params.tree_metadata_stride, &params.stride_parent_token,
        &kAttentionScale, &params.num_tokens, &stride_forget_gate, &stride_qkv, &stride_qkv, &stride_qkv,
        &stride_beta)};
    Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
        "fused_sigmoid_gating_delta_rule_update_kernel", args, key, grid_x, grid_y, grid_z, stream, kKdaAllowPdl);
    return;
  }

  if (!tree) {
    void* args[] = {KDA_TRITON_ARGS(
        &params.a_log, &params.forget_gate, &params.dt_bias, &kSoftplusBeta, &kSoftplusThreshold, &lower_bound,
        &params.q, &params.k, &params.v, &params.beta, &params.output, &params.state, &params.state_indices,
        &params.cu_seqlens, &params.intermediate_states, &params.intermediate_state_indices, &params.cache_steps,
        &params.tree_metadata_stride, &params.stride_parent_token, &kAttentionScale, &params.num_tokens,
        &stride_forget_gate, &stride_qkv, &stride_qkv, &stride_qkv, &stride_beta)};
    Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
        "fused_sigmoid_gating_delta_rule_update_kernel", args, key, grid_x, grid_y, grid_z, stream, kKdaAllowPdl);
    return;
  }

  void* args[] = {KDA_TRITON_ARGS(
      &params.a_log, &params.forget_gate, &params.dt_bias, &kSoftplusBeta, &kSoftplusThreshold, &lower_bound,
      &params.q, &params.k, &params.v, &params.beta, &params.output, &params.state, &params.state_indices,
      &params.cu_seqlens, &params.intermediate_states, &params.intermediate_state_indices, &params.cache_steps,
      &params.retrieve_parent_token, &params.tree_metadata_stride, &params.stride_parent_token, &kAttentionScale,
      &params.num_tokens, &stride_forget_gate, &stride_qkv, &stride_qkv, &stride_qkv, &stride_beta)};
  Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
      "fused_sigmoid_gating_delta_rule_update_kernel", args, key, grid_x, grid_y, grid_z, stream, kKdaAllowPdl);
}

void InvokeKdaGatedRmsNorm(KdaGatedRmsNormParams& params, cudaStream_t stream) {
  if (params.eps != kKdaNormEps) {
    KLLM_LOG_WARNING << "KDA gated RMSNorm eps is expected to be " << kKdaNormEps << ", got " << params.eps;
  }
  float eps = params.eps;
  void* args[] = {KDA_TRITON_ARGS(&params.x, &params.gate, &params.output, &params.weight, &params.bias,
                                        &params.residual, &params.residual_output, &params.mean, &params.rstd, &eps,
                                        &params.num_rows)};
  Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
      "layer_norm_gated_fwd_kernel", args,
      {{"D", std::to_string(kKdaHeadDim)}, {"BT", std::to_string(kNormBlockT)}, {"ACT", "sigmoid"}},
      CeilDiv(params.num_rows, kNormBlockT), 1, 1, stream, kKdaAllowPdl);
}

#undef KDA_TRITON_ARGS

}  // namespace ksana_llm
