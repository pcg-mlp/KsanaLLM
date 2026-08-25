/* Copyright 2026 Tencent Inc. All rights reserved.

==============================================================================*/

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "test.h"

#include "device/nvidia/cuda_utils.h"
#include "kernels/nvidia/triton_wrapper.h"
#include "utils/singleton.h"

namespace ksana_llm {
namespace {

// Geometry mirrors the 27B TP2 profile in gdn/tools/gen_gdn_cubins.py CONFIGS
// (H_key=8, HV=24, D=5120). Every kernel invocation below must match the
// AOT cubin ABI (param count/order/type) — a drift is what this test catches.
constexpr int kNumKeyHeads = 8;
constexpr int kNumValueHeads = 24;
constexpr int kHeadK = 128;
constexpr int kHeadV = 128;
constexpr int kConvDim = 5120;       // conv_dim = H_key*K*2 + HV*V
constexpr int kConvKernelWidth = 4;  // KW; state_len = KW - 1
constexpr int kChunkSize = 64;
constexpr int kNumTokens = 128;  // prefill length: 2 chunks
constexpr int kNumChunks = 2;    // cdiv(kNumTokens, kChunkSize)
constexpr int kNumSeqs = 1;
constexpr int kNumSlots = 2;  // slot 0 (NULL) + 1 live request
constexpr int kStateLen = kConvKernelWidth - 1;
// Conv state row span in bf16 elems (slot-major pool layout).
constexpr int kSlotStrideElems = kConvDim * kStateLen;
constexpr int kGdnNumCacheLines = 2048 + 1;  // live slots + slot 0
constexpr size_t kBf16Bytes = 2;
constexpr size_t kInt32Bytes = 4;
constexpr size_t kFp32Bytes = 4;

class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t bytes) : bytes_(bytes) {
    CUDA_CHECK(cudaMalloc(&data_, bytes_));
    CUDA_CHECK(cudaMemset(data_, 0, bytes_));
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) cudaFree(data_);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* data() const { return data_; }

  template <typename T>
  T* As() const {
    return static_cast<T*>(data_);
  }

  template <typename T>
  void Upload(const std::vector<T>& host) {
    ASSERT_LE(host.size() * sizeof(T), bytes_);
    CUDA_CHECK(cudaMemcpy(data_, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
  }

 private:
  void* data_{nullptr};
  size_t bytes_{0};
};

// Prefill buffers shared by the chunk-pipeline kernels (ForwardT layout).
struct PrefillBuffers {
  DeviceBuffer x;               // mixed_qkv [L, D] bf16 (in-place conv output)
  DeviceBuffer conv_w;          // conv1d weight [D, KW] bf16
  DeviceBuffer conv_pool;       // conv state pool [slots, D, state_len] bf16
  DeviceBuffer cache_indices;   // [num_chunks, 8] int32
  DeviceBuffer has_initial;     // [N] uint8
  DeviceBuffer cu_seqlens;      // [N+1] int32
  DeviceBuffer conv_batch;      // per conv-chunk batch [cdiv(L,8)+1] int32
  DeviceBuffer conv_chunk_off;  // per conv-chunk token offset [cdiv(L,8)+1] int32
  DeviceBuffer q_tmp;           // [L, H, K] bf16
  DeviceBuffer k_tmp;           // [L, H, K] bf16
  DeviceBuffer v;               // [L, HV, V] bf16
  DeviceBuffer g;               // [L, HV] fp32 (raw gate)
  DeviceBuffer beta;            // [L, HV] fp32
  DeviceBuffer a_log;           // [HV] fp32
  DeviceBuffer dt_bias;         // [HV, V] fp32
  DeviceBuffer cumsum;          // [align(L,64), HV] fp32 (chunk-local gate cumsum)
  DeviceBuffer a_mat;           // [NT, HV, 64, 64] fp32 (KKT A)
  DeviceBuffer a_inv;           // [NT, HV, 64, 64] bf16 (solve_tril result)
  DeviceBuffer h;               // [NT, HV, V, K] bf16 (chunk states; model dtype per vendored contract)
  DeviceBuffer h0;              // [N, HV, V, K] fp32 (initial state)
  DeviceBuffer ht;              // [N, HV, V, K] fp32 (final state)
  DeviceBuffer o_delta;         // [L, HV, V] bf16 (chunk_o output)
  DeviceBuffer z;               // [L*HV, V] fp32 (gate, row-major [M, N]; layer_norm_fwd_kernel reads full rows)
  DeviceBuffer rstd;            // [L, HV] fp32
  DeviceBuffer norm_w;          // [V] fp32

  PrefillBuffers()
      : x(static_cast<size_t>(kNumTokens) * kConvDim * kBf16Bytes),
        conv_w(static_cast<size_t>(kConvDim) * kConvKernelWidth * kBf16Bytes),
        conv_pool(static_cast<size_t>(kNumSlots) * kConvDim * kStateLen * kBf16Bytes),
        cache_indices(static_cast<size_t>(kNumChunks) * 8 * kInt32Bytes),
        has_initial(kNumSeqs),
        cu_seqlens(static_cast<size_t>(kNumSeqs + 1) * kInt32Bytes),
        conv_batch(static_cast<size_t>(kNumTokens / 8 + 1) * kInt32Bytes),
        conv_chunk_off(static_cast<size_t>(kNumTokens / 8 + 1) * kInt32Bytes),
        // Chunk kernels consume the GQA-expanded q/k ([L, HV, K]); sizing by
        // HV keeps every downstream read in-bounds (fused_post_conv writes
        // only H rows under its own stride, so this also holds for it).
        q_tmp(static_cast<size_t>(kNumTokens) * kNumValueHeads * kHeadK * kBf16Bytes),
        k_tmp(static_cast<size_t>(kNumTokens) * kNumValueHeads * kHeadK * kBf16Bytes),
        v(static_cast<size_t>(kNumTokens) * kNumValueHeads * kHeadV * kBf16Bytes),
        g(static_cast<size_t>(kNumTokens) * kNumValueHeads * kFp32Bytes),
        beta(static_cast<size_t>(kNumTokens) * kNumValueHeads * kFp32Bytes),
        a_log(static_cast<size_t>(kNumValueHeads) * kFp32Bytes),
        dt_bias(static_cast<size_t>(kNumValueHeads) * kHeadV * kFp32Bytes),
        cumsum(static_cast<size_t>(kNumTokens) * kNumValueHeads * kFp32Bytes),
        a_mat(static_cast<size_t>(kNumChunks) * kNumValueHeads * 64 * 64 * kFp32Bytes),
        a_inv(static_cast<size_t>(kNumChunks) * kNumValueHeads * 64 * 64 * kBf16Bytes),
        h(static_cast<size_t>(kNumChunks) * kNumValueHeads * kHeadV * kHeadK * kBf16Bytes),
        h0(static_cast<size_t>(kNumSeqs) * kNumValueHeads * kHeadV * kHeadK * kFp32Bytes),
        ht(static_cast<size_t>(kNumSeqs) * kNumValueHeads * kHeadV * kHeadK * kFp32Bytes),
        o_delta(static_cast<size_t>(kNumTokens) * kNumValueHeads * kHeadV * kBf16Bytes),
        z(static_cast<size_t>(kNumTokens) * kNumValueHeads * kHeadV * kFp32Bytes),
        rstd(static_cast<size_t>(kNumTokens) * kNumValueHeads * kFp32Bytes),
        norm_w(static_cast<size_t>(kHeadV) * kFp32Bytes) {}
};

// Decode buffers for conv1d_update + packed_decode (ForwardTDecode layout).
struct DecodeBuffers {
  DeviceBuffer x;             // mixed_qkv [1, D] bf16 (in-place)
  DeviceBuffer conv_w;        // conv1d weight [D, KW] bf16
  DeviceBuffer state;         // conv state pool [slots, D, state_len] bf16
  DeviceBuffer slot_indices;  // [1] int32 (live slot = 1)
  DeviceBuffer rec_state;     // recurrent state pool [slots, HV, V, K] fp32
  DeviceBuffer a;             // [1, HV] fp32
  DeviceBuffer b;             // [1, HV] fp32
  DeviceBuffer a_log;         // [HV] fp32
  DeviceBuffer dt_bias;       // [HV, V] fp32
  DeviceBuffer o;             // [1, HV, V] bf16
  DeviceBuffer z;             // [1, HV] fp32
  DeviceBuffer rstd;          // [1, HV] fp32
  DeviceBuffer norm_w;        // [V] fp32

  DecodeBuffers()
      : x(static_cast<size_t>(kConvDim) * kBf16Bytes),
        conv_w(static_cast<size_t>(kConvDim) * kConvKernelWidth * kBf16Bytes),
        state(static_cast<size_t>(kNumSlots) * kConvDim * kStateLen * kBf16Bytes),
        slot_indices(1 * kInt32Bytes),
        // rec pool [slots, HV, V, K] fp32; 2*kNumSlots rows (MTP1 headroom).
        rec_state(static_cast<size_t>(2 * kNumSlots) * kNumValueHeads * kHeadV * kHeadK * kFp32Bytes),
        a(static_cast<size_t>(kNumValueHeads) * kFp32Bytes),
        b(static_cast<size_t>(kNumValueHeads) * kFp32Bytes),
        a_log(static_cast<size_t>(kNumValueHeads) * kFp32Bytes),
        dt_bias(static_cast<size_t>(kNumValueHeads) * kHeadV * kFp32Bytes),
        o(static_cast<size_t>(kNumValueHeads) * kHeadV * kBf16Bytes),
        z(static_cast<size_t>(kNumValueHeads) * kFp32Bytes),
        rstd(static_cast<size_t>(kNumValueHeads) * kFp32Bytes),
        norm_w(static_cast<size_t>(kHeadV) * kFp32Bytes) {}
};

void ExpectLaunchOk(const std::function<void()>& launch, cudaStream_t stream) {
  launch();
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
}

}  // namespace

class GdnKernelLaunchTest : public testing::Test {
 protected:
  void SetUp() override {
    CUDA_CHECK(cudaStreamCreate(&stream_));
    // No SM gate: GDN cubins are generated per-SM on demand
    // (tools/gen_gdn_cubins.py ensure); the canary below is the only gate.
    // L40 is compute capability 8.9 (major=8), which a naive `major < 9` skip
    // would wrongly reject. Canary: any conv-update cubin for D=5120 proves
    // the generated artifact set exists.
    const std::string dir = Singleton<TritonWrapper>::GetInstance()->GetKernelsDir();
    bool found = false;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
      if (entry.path().extension() == ".cubin" &&
          entry.path().filename().string().find("gdn_causal_conv1d_update_kernel_D_5120_") == 0) {
        found = true;
        break;
      }
    }
    if (!found) {
      GTEST_SKIP() << "GDN AOT artifacts (D=5120) not found under " << dir;
    }
  }

  void TearDown() override {
    if (stream_ != nullptr) {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
    }
  }

  // Spec recurrent fwd: N=1, T=1+mtp tokens, SISQ=1+mtp. Pool = (1+mtp)*kNumSlots
  // rows (minimal correct: a stride/indices regression spills past the tail).
  void RunFusedRecurrentFwd(int mtp) {
    const std::string dir = Singleton<TritonWrapper>::GetInstance()->GetKernelsDir();
    const std::string sisq_tag = fmt::format("_SISQ_{}_", 1 + mtp);
    bool found = false;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
      const std::string name = entry.path().filename().string();
      if (entry.path().extension() == ".cubin" &&
          name.find("fused_recurrent_gated_delta_rule_fwd_kernel_H_8_HV_24_K_128_V_128_") == 0 &&
          name.find(sisq_tag) != std::string::npos) {
        found = true;
        break;
      }
    }
    if (!found) {
      GTEST_SKIP() << "fused_recurrent_fwd cubin (H=8/HV=24, SISQ=" << (1 + mtp) << ") not found under " << dir;
    }

    int64_t num_tokens = 1 + mtp;
    // (1+mtp) tokens (real + drafts): q/k [T,H,K], v [T,HV,V], g/beta [T,HV] fp32.
    DeviceBuffer q(static_cast<size_t>(num_tokens) * kNumKeyHeads * kHeadK * kBf16Bytes);
    DeviceBuffer k(static_cast<size_t>(num_tokens) * kNumKeyHeads * kHeadK * kBf16Bytes);
    DeviceBuffer v(static_cast<size_t>(num_tokens) * kNumValueHeads * kHeadV * kBf16Bytes);
    DeviceBuffer g(static_cast<size_t>(num_tokens) * kNumValueHeads * kFp32Bytes);
    DeviceBuffer beta(static_cast<size_t>(num_tokens) * kNumValueHeads * kFp32Bytes);
    DeviceBuffer o(static_cast<size_t>(num_tokens) * kNumValueHeads * kHeadV * kBf16Bytes);
    DeviceBuffer cu(2 * kInt32Bytes);
    DeviceBuffer ssm_indices(static_cast<size_t>(1 + mtp) * (1 + mtp) * kInt32Bytes);  // [1, 1+mtp]
    DeviceBuffer num_accepted(1 * kInt32Bytes);
    DeviceBuffer rec_pool(static_cast<size_t>(1 + mtp) * kNumSlots * kNumValueHeads * kHeadV * kHeadK *
                          kFp32Bytes);

    std::vector<int32_t> cu_host = {0, static_cast<int32_t>(num_tokens)};
    std::vector<int32_t> indices_host(1 + mtp);
    for (int step = 0; step <= mtp; ++step) {
      indices_host[step] = step * kNumSlots + 1;
    }
    cu.Upload<int32_t>(cu_host);
    ssm_indices.Upload<int32_t>(indices_host);
    num_accepted.Upload<int32_t>({1});

    void* q_ptr = q.data();
    void* k_ptr = k.data();
    void* v_ptr = v.data();
    void* g_ptr = g.data();
    void* beta_ptr = beta.data();
    void* o_ptr = o.data();
    void* h0_ptr = rec_pool.data();  // rec pool base (INPLACE)
    void* ht_ptr = h0_ptr;
    void* cu_ptr = cu.data();
    void* ssm_ptr = ssm_indices.data();
    void* num_acc_ptr = num_accepted.data();
    float scale_val = 1.0f / std::sqrt(static_cast<float>(kHeadK));
    int64_t num_seqs = 1;
    int32_t rec_row_stride = static_cast<int32_t>(kNumValueHeads) * kHeadV * kHeadK;  // fp32 elems
    void* fwd_args[16] = {&q_ptr, &k_ptr, &v_ptr, &g_ptr, &beta_ptr, &o_ptr, &h0_ptr, &ht_ptr,
                          &cu_ptr, &ssm_ptr, &num_acc_ptr, &scale_val,
                          &num_seqs, &num_tokens, &rec_row_stride, &rec_row_stride};

    std::unordered_map<std::string, std::string> fwd_map = {
        {"H", std::to_string(kNumKeyHeads)},
        {"HV", std::to_string(kNumValueHeads)},
        {"K", std::to_string(kHeadK)},
        {"V", std::to_string(kHeadV)},
        {"BK", "128"},
        {"BV", "32"},
        {"SISQ", std::to_string(1 + mtp)},
        {"SIT", "1"},
        {"UIS", "1"},
        {"INP", "1"},
        {"IBH", "0"},
        {"L2N", "0"},
        {"VAR", "1"},
        {"CB", "1"},
        {"SPEC", "1"},
        {"KDA", "0"},
        {"W", "1"},
        {"S", "3"},
        {"compute_type", "BF16"}};

    const size_t fwd_nv = (static_cast<size_t>(kHeadV) + 31) / 32;
    ExpectLaunchOk(
        [&] {
          Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
              "fused_recurrent_gated_delta_rule_fwd_kernel", fwd_args, fwd_map,
              /*grid_x=*/1, /*grid_y=*/fwd_nv, /*grid_z=*/kNumValueHeads, stream_, /*allow_pdl=*/false,
              /*block_x=*/32, /*block_y=*/1, /*block_z=*/1);
        },
        stream_);
  }

  cudaStream_t stream_{nullptr};
};

// ===== Prefill (ForwardT) kernels =====

// fused_post_conv: split mixed_qkv + L2-norm q/k + fold a/b (17 params).
TEST_F(GdnKernelLaunchTest, FusedPostConvLaunches) {
  PrefillBuffers b;
  int stride_x_tok = kConvDim;
  int stride_a_tok = kNumValueHeads;
  int stride_b_tok = kNumValueHeads;
  int stride_q_tok = kNumKeyHeads * kHeadK;
  int stride_k_tok = kNumKeyHeads * kHeadK;
  int stride_v_tok = kNumValueHeads * kHeadV;
  void* mx_ptr = b.x.data();
  void* a_ptr = b.g.data();
  void* b_ptr = b.beta.data();
  void* Alog_ptr = b.a_log.data();
  void* dt_ptr = b.dt_bias.data();
  void* q_ptr = b.q_tmp.data();
  void* k_ptr = b.k_tmp.data();
  void* v_ptr = b.v.data();
  void* g_ptr = b.g.data();
  void* beta_ptr = b.beta.data();
  int num_tokens = kNumTokens;

  void* fps_args[17] = {&mx_ptr,       &a_ptr,        &b_ptr,        &Alog_ptr,     &dt_ptr,       &q_ptr,
                        &k_ptr,        &v_ptr,        &g_ptr,        &beta_ptr,     &stride_x_tok, &stride_a_tok,
                        &stride_b_tok, &stride_q_tok, &stride_k_tok, &stride_v_tok, &num_tokens};

  std::unordered_map<std::string, std::string> fps_map = {{"H", std::to_string(kNumKeyHeads)},
                                                          {"HV", std::to_string(kNumValueHeads)},
                                                          {"K", std::to_string(kHeadK)},
                                                          {"V", std::to_string(kHeadV)},
                                                          {"BK", "128"},
                                                          {"BV", "128"},
                                                          {"BT", "16"},
                                                          {"L2", "1"},
                                                          {"GEXP", "0"},
                                                          {"W", "4"},
                                                          {"S", "2"},
                                                          {"compute_type", "BF16"}};

  const size_t fps_grid_x = (static_cast<size_t>(kNumTokens) + 15) / 16;
  ExpectLaunchOk(
      [&] {
        Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
            "fused_post_conv_kernel", fps_args, fps_map, fps_grid_x,
            static_cast<size_t>(kNumKeyHeads) + static_cast<size_t>(kNumValueHeads), 1, stream_, /*allow_pdl=*/false,
            /*block_x=*/-1, /*block_y=*/-1, /*block_z=*/-1);
      },
      stream_);
}

// fused_conv_post_c1: fused conv+post main kernel (25 runtime params; the
// generator asserts the same count at dump time, see fused_conv_post_c1.py).
TEST_F(GdnKernelLaunchTest, FusedConvPostC1Launches) {
  PrefillBuffers b;
  b.cu_seqlens.Upload<int32_t>({0, kNumTokens});
  void* x_ptr = b.x.data();
  void* w_ptr = b.conv_w.data();
  void* pool_ptr = b.conv_pool.data();
  void* cidx_ptr = b.cache_indices.data();
  void* hin_ptr = b.has_initial.data();
  void* qsl_ptr = b.cu_seqlens.data();
  void* a_ptr = b.g.data();
  void* b_ptr = b.beta.data();
  void* alog_ptr = b.a_log.data();
  void* dtb_ptr = b.dt_bias.data();
  void* q_ptr = b.q_tmp.data();
  void* k_ptr = b.k_tmp.data();
  void* v_ptr = b.v.data();
  void* g_ptr = b.g.data();
  void* beta_ptr = b.beta.data();
  int32_t stride_x_token = kConvDim;
  int32_t stride_w_dim = kConvKernelWidth;
  int32_t stride_pool_seq = kSlotStrideElems;
  int32_t stride_pool_dim = kStateLen;
  int32_t stride_cidx = 8;
  int32_t stride_a_tok = kNumValueHeads;
  int32_t stride_b_tok = kNumValueHeads;
  int32_t dim = kConvDim;
  int32_t seqlen = kNumTokens;
  int32_t num_cache_lines = kGdnNumCacheLines;
  void* c1_args[25] = {&x_ptr,       &w_ptr,       &pool_ptr,     &cidx_ptr,    &hin_ptr,
                       &qsl_ptr,     &a_ptr,       &b_ptr,        &alog_ptr,    &dtb_ptr,
                       &q_ptr,       &k_ptr,       &v_ptr,        &g_ptr,       &beta_ptr,
                       &stride_x_token, &stride_w_dim, &stride_pool_seq, &stride_pool_dim, &stride_cidx,
                       &stride_a_tok, &stride_b_tok, &dim, &seqlen, &num_cache_lines};

  std::unordered_map<std::string, std::string> c1_map = {{"H", std::to_string(kNumKeyHeads)},
                                                         {"HV", std::to_string(kNumValueHeads)},
                                                         {"K", std::to_string(kHeadK)},
                                                         {"V", std::to_string(kHeadV)},
                                                         {"BT", "16"},
                                                         {"L2", "1"},
                                                         {"GEXP", "0"},
                                                         {"NSEQ", "0"},
                                                         {"W", "4"},
                                                         {"S", "2"},
                                                         {"compute_type", "BF16"}};

  const size_t c1_grid_x = (static_cast<size_t>(kNumTokens) + 15) / 16;
  ExpectLaunchOk(
      [&] {
        Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
            "fused_conv_post_c1_kernel", c1_args, c1_map, c1_grid_x,
            static_cast<size_t>(2) * kNumKeyHeads + kNumValueHeads, 1, stream_, /*allow_pdl=*/false,
            /*block_x=*/-1, /*block_y=*/-1, /*block_z=*/-1);
      },
      stream_);
}

// fused_conv_post_c1 boundary: per-seq conv redo + pool write-back (15 runtime
// params; grid_x = N, grid_y = 2*H + HV).
TEST_F(GdnKernelLaunchTest, FusedConvPostC1BoundaryLaunches) {
  PrefillBuffers b;
  b.cu_seqlens.Upload<int32_t>({0, kNumTokens});
  b.cache_indices.Upload<int32_t>({1, 0, 0, 0, 0, 0, 0, 0});  // seq 0 -> slot 1
  void* x_ptr = b.x.data();
  void* w_ptr = b.conv_w.data();
  void* pool_ptr = b.conv_pool.data();
  void* cidx_ptr = b.cache_indices.data();
  void* hin_ptr = b.has_initial.data();
  void* qsl_ptr = b.cu_seqlens.data();
  void* q_ptr = b.q_tmp.data();
  void* k_ptr = b.k_tmp.data();
  void* v_ptr = b.v.data();
  int32_t stride_x_token = kConvDim;
  int32_t stride_w_dim = kConvKernelWidth;
  int32_t stride_pool_seq = kSlotStrideElems;
  int32_t stride_pool_dim = kStateLen;
  int32_t stride_cidx = 8;
  int32_t seqlen = kNumTokens;
  void* c1b_args[15] = {&x_ptr,       &w_ptr,       &pool_ptr, &cidx_ptr,    &hin_ptr,
                        &qsl_ptr,     &q_ptr,       &k_ptr,    &v_ptr,       &stride_x_token,
                        &stride_w_dim, &stride_pool_seq, &stride_pool_dim, &stride_cidx, &seqlen};

  std::unordered_map<std::string, std::string> c1b_map = {{"H", std::to_string(kNumKeyHeads)},
                                                          {"HV", std::to_string(kNumValueHeads)},
                                                          {"K", std::to_string(kHeadK)},
                                                          {"V", std::to_string(kHeadV)},
                                                          {"BT", "8"},
                                                          {"L2", "1"},
                                                          {"W", "4"},
                                                          {"S", "1"},
                                                          {"compute_type", "BF16"}};

  ExpectLaunchOk(
      [&] {
        Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
            "fused_conv_post_c1_boundary_kernel", c1b_args, c1b_map, /*grid_x=*/1,
            static_cast<size_t>(2) * kNumKeyHeads + kNumValueHeads, 1, stream_, /*allow_pdl=*/false,
            /*block_x=*/-1, /*block_y=*/-1, /*block_z=*/-1);
      },
      stream_);
}

// chunk_local_cumsum_scalar: gate cumsum (5 params).
TEST_F(GdnKernelLaunchTest, ChunkLocalCumsumScalarLaunches) {
  PrefillBuffers b;
  b.cu_seqlens.Upload<int32_t>({0, kNumTokens});
  b.conv_batch.Upload<int32_t>({0, kNumChunks});
  void* s_ptr = b.g.data();
  void* o_ptr = b.cumsum.data();
  void* cu_seqlens_ptr = b.cu_seqlens.data();
  void* conv_batch_ptr = b.conv_batch.data();
  int num_tokens = kNumTokens;
  void* clc_args[5] = {&s_ptr, &o_ptr, &cu_seqlens_ptr, &conv_batch_ptr, &num_tokens};

  std::unordered_map<std::string, std::string> clc_map = {{"B", "1"},
                                                          {"H", std::to_string(kNumValueHeads)},
                                                          {"BT", std::to_string(kChunkSize)},
                                                          {"REV", "0"},
                                                          {"VL", "1"},
                                                          {"HF", "0"},
                                                          {"W", "4"},
                                                          {"S", "1"},
                                                          {"compute_type", "BF16"}};

  ExpectLaunchOk(
      [&] {
        Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
            "chunk_local_cumsum_scalar_kernel", clc_args, clc_map, kNumChunks, kNumValueHeads, 1, stream_,
            /*allow_pdl=*/false,
            /*block_x=*/-1, /*block_y=*/-1, /*block_z=*/-1);
      },
      stream_);
}

// chunk_scaled_dot_kkt: gate-decayed attn matrix A (7 params, G=1).
TEST_F(GdnKernelLaunchTest, ChunkScaledDotKktLaunches) {
  PrefillBuffers b;
  b.cu_seqlens.Upload<int32_t>({0, kNumTokens});
  b.conv_batch.Upload<int32_t>({0, kNumChunks});
  void* k_ptr = b.k_tmp.data();
  void* beta_ptr = b.beta.data();
  void* g_ptr = b.cumsum.data();
  void* A_ptr = b.a_mat.data();
  void* cu_seqlens_ptr = b.cu_seqlens.data();
  void* conv_batch_ptr = b.conv_batch.data();
  int num_tokens = kNumTokens;
  void* csd_args[7] = {&k_ptr, &beta_ptr, &g_ptr, &A_ptr, &cu_seqlens_ptr, &conv_batch_ptr, &num_tokens};

  std::unordered_map<std::string, std::string> csd_map = {{"H", std::to_string(kNumValueHeads)},
                                                          {"Hg", std::to_string(kNumKeyHeads)},
                                                          {"K", std::to_string(kHeadK)},
                                                          {"BT", std::to_string(kChunkSize)},
                                                          {"BK", "64"},
                                                          {"VL", "1"},
                                                          {"G", "1"},
                                                          {"W", "4"},
                                                          {"S", "3"},
                                                          {"compute_type", "BF16"}};

  ExpectLaunchOk(
      [&] {
        Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
            "chunk_scaled_dot_kkt_fwd_kernel", csd_args, csd_map, kNumChunks, kNumValueHeads, 1, stream_,
            /*allow_pdl=*/false,
            /*block_x=*/-1, /*block_y=*/-1, /*block_z=*/-1);
      },
      stream_);
}

// solve_tril_64: invert per-chunk Cholesky factor (5 params).
TEST_F(GdnKernelLaunchTest, SolveTrilLaunches) {
  PrefillBuffers b;
  b.cu_seqlens.Upload<int32_t>({0, kNumTokens});
  b.conv_batch.Upload<int32_t>({0, kNumChunks});
  void* A_ptr = b.a_mat.data();
  void* Ai_ptr = b.a_inv.data();
  void* cu_seqlens_ptr = b.cu_seqlens.data();
  void* conv_batch_ptr = b.conv_batch.data();
  int num_tokens = kNumTokens;
  void* str_args[5] = {&A_ptr, &Ai_ptr, &cu_seqlens_ptr, &conv_batch_ptr, &num_tokens};

  std::unordered_map<std::string, std::string> str_map = {{"H", std::to_string(kNumValueHeads)},
                                                          {"BT", "64"},
                                                          {"TMA", "0"},
                                                          {"VL", "1"},
                                                          {"DP", "ieee"},
                                                          {"W", "4"},
                                                          {"S", "3"},
                                                          {"compute_type", "BF16"}};

  ExpectLaunchOk(
      [&] {
        Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
            "merge_16x16_to_64x64_inverse_kernel", str_args, str_map, kNumChunks, kNumValueHeads, 1, stream_,
            /*allow_pdl=*/false,
            /*block_x=*/-1, /*block_y=*/-1, /*block_z=*/-1);
      },
      stream_);
}

// wy_fast_recompute: recompute (w, u) (10 params).
TEST_F(GdnKernelLaunchTest, WyFastRecomputeLaunches) {
  PrefillBuffers b;
  b.cu_seqlens.Upload<int32_t>({0, kNumTokens});
  b.conv_batch.Upload<int32_t>({0, kNumChunks});
  void* k_ptr = b.k_tmp.data();
  void* v_ptr = b.v.data();
  void* beta_ptr = b.beta.data();
  void* w_ptr = b.o_delta.data();
  void* u_ptr = b.q_tmp.data();
  void* A_ptr = b.a_inv.data();
  void* g_ptr = b.cumsum.data();
  void* cu_seqlens_ptr = b.cu_seqlens.data();
  void* conv_batch_ptr = b.conv_batch.data();
  int num_tokens = kNumTokens;
  void* wyr_args[10] = {&k_ptr, &v_ptr, &beta_ptr,       &w_ptr,          &u_ptr,
                        &A_ptr, &g_ptr, &cu_seqlens_ptr, &conv_batch_ptr, &num_tokens};

  std::unordered_map<std::string, std::string> wyr_map = {{"H", std::to_string(kNumValueHeads)},
                                                          {"Hg", std::to_string(kNumKeyHeads)},
                                                          {"K", std::to_string(kHeadK)},
                                                          {"V", std::to_string(kHeadV)},
                                                          {"BT", std::to_string(kChunkSize)},
                                                          {"BK", "64"},
                                                          {"BV", "64"},
                                                          {"VL", "1"},
                                                          {"W", "4"},
                                                          {"S", "3"},
                                                          {"compute_type", "BF16"}};

  ExpectLaunchOk(
      [&] {
        Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
            "gdn_recompute_w_u_fwd_kernel", wyr_args, wyr_map, kNumChunks, kNumValueHeads, 1, stream_,
            /*allow_pdl=*/false,
            /*block_x=*/-1, /*block_y=*/-1, /*block_z=*/-1);
      },
      stream_);
}

// chunk_h: state evolution (12 params; H0OFF=1 direct pool rows).
TEST_F(GdnKernelLaunchTest, ChunkHLaunches) {
  PrefillBuffers b;
  b.cu_seqlens.Upload<int32_t>({0, kNumTokens});
  b.conv_chunk_off.Upload<int32_t>({0, kNumChunks});
  // H0OFF=1: h0/ht point at the rec-state pool base; per-seq h0_offsets
  // (int64, float elems) pick the pool row — positive = read initial + write
  // final. Row 0 is the NULL_BLOCK_ID reserve, so aim at row 1 of a 2-row pool.
  DeviceBuffer pool(static_cast<size_t>(2) * kNumValueHeads * kHeadV * kHeadK * kFp32Bytes);
  DeviceBuffer h0o(static_cast<size_t>(kNumSeqs) * sizeof(int64_t));
  h0o.Upload<int64_t>({static_cast<int64_t>(kNumValueHeads) * kHeadV * kHeadK});
  void* k_ptr = b.k_tmp.data();
  void* v_ptr = b.q_tmp.data();
  void* w_ptr = b.o_delta.data();
  void* vn_ptr = b.v.data();
  void* g_ptr = b.cumsum.data();
  void* h_ptr = b.h.data();
  void* h0_ptr = pool.data();
  void* ht_ptr = pool.data();
  void* h0o_ptr = h0o.data();
  void* cu_seqlens_ptr = b.cu_seqlens.data();
  void* conv_chunk_off_ptr = b.conv_chunk_off.data();
  int num_tokens = kNumTokens;
  void* ch_args[12] = {&k_ptr,     &v_ptr,  &w_ptr, &vn_ptr,         &g_ptr,
                       &h_ptr,     &h0_ptr, &ht_ptr, &h0o_ptr,        &cu_seqlens_ptr,
                       &conv_chunk_off_ptr,
                       &num_tokens};

  std::unordered_map<std::string, std::string> ch_map = {{"H", std::to_string(kNumValueHeads)},
                                                         {"Hg", std::to_string(kNumKeyHeads)},
                                                         {"K", std::to_string(kHeadK)},
                                                         {"V", std::to_string(kHeadV)},
                                                         {"BT", std::to_string(kChunkSize)},
                                                         {"BV", "64"},
                                                         {"G", "1"},
                                                         {"GK", "0"},
                                                         {"INIT", "1"},
                                                         {"STORE", "1"},
                                                         {"SAVE", "1"},
                                                         {"VL", "1"},
                                                         {"W", "4"},
                                                         {"S", "3"},
                                                         {"H0OFF", "1"},
                                                         {"compute_type", "BF16"}};

  ExpectLaunchOk(
      [&] {
        Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
            "gdn_chunk_gated_delta_rule_fwd_kernel_h_blockdim64", ch_args, ch_map,
            /*grid_x=*/2, /*grid_y=*/static_cast<size_t>(kNumSeqs) * kNumValueHeads, /*grid_z=*/1, stream_,
            /*allow_pdl=*/false,
            /*block_x=*/-1, /*block_y=*/-1, /*block_z=*/-1);
      },
      stream_);
}

// chunk_o: combine q/k/v/h/g into o_delta (10 params).
TEST_F(GdnKernelLaunchTest, ChunkOLaunches) {
  PrefillBuffers b;
  b.cu_seqlens.Upload<int32_t>({0, kNumTokens});
  b.conv_batch.Upload<int32_t>({0, kNumChunks});
  void* q_ptr = b.q_tmp.data();
  void* k_ptr = b.k_tmp.data();
  void* v_ptr = b.v.data();
  void* h_ptr = b.h.data();
  void* g_ptr = b.cumsum.data();
  void* o_ptr = b.o_delta.data();
  void* cu_seqlens_ptr = b.cu_seqlens.data();
  void* conv_batch_ptr = b.conv_batch.data();
  float scale = 1.0f / std::sqrt(static_cast<float>(kHeadK));
  int num_tokens = kNumTokens;
  void* co_args[10] = {&q_ptr, &k_ptr,          &v_ptr,          &h_ptr, &g_ptr,
                       &o_ptr, &cu_seqlens_ptr, &conv_batch_ptr, &scale, &num_tokens};

  std::unordered_map<std::string, std::string> co_map = {{"H", std::to_string(kNumValueHeads)},
                                                         {"Hg", std::to_string(kNumKeyHeads)},
                                                         {"K", std::to_string(kHeadK)},
                                                         {"V", std::to_string(kHeadV)},
                                                         {"BT", std::to_string(kChunkSize)},
                                                         {"BK", "64"},
                                                         {"BV", "64"},
                                                         {"G", "1"},
                                                         {"VL", "1"},
                                                         {"W", "4"},
                                                         {"S", "3"},
                                                         {"compute_type", "BF16"}};

  ExpectLaunchOk(
      [&] {
        Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
            "chunk_fwd_kernel_o", co_args, co_map, /*grid_x=*/2, /*grid_y=*/kNumChunks,
            /*grid_z=*/kNumValueHeads, stream_, /*allow_pdl=*/false,
            /*block_x=*/-1, /*block_y=*/-1, /*block_z=*/-1);
      },
      stream_);
}

// rmsnorm_gated: RMSNorm(o_delta) * SiLU(z) (12 params; used in prefill+decode).
TEST_F(GdnKernelLaunchTest, RmsNormGatedLaunches) {
  PrefillBuffers b;
  void* x_ptr = b.o_delta.data();
  void* y_ptr = b.o_delta.data();
  void* z_ptr = b.z.data();
  void* rstd_ptr = b.rstd.data();
  void* w_ptr = b.norm_w.data();
  int stride_v = kHeadV;
  int m = kNumTokens * kNumValueHeads;
  float eps = 1e-6f;
  // B (slot 3) / Mean (slot 5) are dead under HAS_BIAS=0 / IS_RMS_NORM=1 but the
  // generator keeps them in the signature (dummy placeholders), so the slots must
  // carry a valid address — never nullptr, which cuLaunchKernelEx rejects.
  int dead_zero = 0;
  void* rng_args[12] = {&x_ptr,    &y_ptr,    &w_ptr,     &dead_zero, &z_ptr, &dead_zero,
                        &rstd_ptr, &stride_v, &stride_v,  &stride_v,  &m,     &eps};

  std::unordered_map<std::string, std::string> rng_map = {{"N", std::to_string(kHeadV)},
                                                          {"BN", "128"},
                                                          {"R", "4"},
                                                          {"W", "4"},
                                                          {"S", "3"},
                                                          {"HB", "0"},
                                                          {"HZ", "1"},
                                                          {"NBG", "1"},
                                                          {"RMS", "1"},
                                                          {"ACT", "swish"},
                                                          {"compute_type", "BF16"}};

  const size_t rng_grid_x = (static_cast<size_t>(m) + 3) / 4;
  ExpectLaunchOk(
      [&] {
        Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
            "rmsnorm_gated_kernel", rng_args, rng_map, rng_grid_x, 1, 1, stream_, /*allow_pdl=*/false,
            /*block_x=*/-1, /*block_y=*/-1, /*block_z=*/-1);
      },
      stream_);
}

// ===== Decode (ForwardTDecode) kernels =====

// conv1d_update: 12 runtime params (constexpr/None/dead params stripped by the
// compiler — caller must pass exactly 12, not 14; a mismatch shifts every
// following slot and OOBs conv_state).
TEST_F(GdnKernelLaunchTest, CausalConv1dUpdateLaunches) {
  DecodeBuffers b;
  b.slot_indices.Upload<int32_t>({1});
  auto* x_ptr = b.x.As<__nv_bfloat16>();
  void* w_ptr = b.conv_w.data();
  void* state_ptr = b.state.data();
  void* indices_ptr = b.slot_indices.data();
  auto* o_ptr = b.x.As<__nv_bfloat16>();
  int32_t batch = 1;
  int32_t slot_stride_elems = kSlotStrideElems;
  int32_t state_dim_stride = kStateLen;
  int32_t state_tok_stride = 1;
  void* c1d_args[12] = {&x_ptr,
                        &w_ptr,
                        &state_ptr,
                        &indices_ptr,
                        &indices_ptr,
                        &indices_ptr,
                        &o_ptr,
                        &batch,
                        const_cast<int*>(&kGdnNumCacheLines),
                        &slot_stride_elems,
                        &state_dim_stride,
                        &state_tok_stride};

  std::unordered_map<std::string, std::string> c1d_map = {{"D", std::to_string(kConvDim)},
                                                          {"KW", std::to_string(kConvKernelWidth)},
                                                          {"BN", "256"},
                                                          {"NP2", "4"},
                                                          {"SL", "3"},
                                                          {"HB", "0"},
                                                          {"SILU", "1"},
                                                          {"VL", "0"},
                                                          {"APC", "0"},
                                                          {"SPEC", "0"},
                                                          {"NULL", "1"},
                                                          {"W", "4"},
                                                          {"S", "2"},
                                                          {"compute_type", "BF16"}};

  const size_t c1d_grid_y = (static_cast<size_t>(kConvDim) + 255) / 256;
  ExpectLaunchOk(
      [&] {
        Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
            "gdn_causal_conv1d_update_kernel", c1d_args, c1d_map,
            /*grid_x=*/1, /*grid_y=*/c1d_grid_y, /*grid_z=*/1, stream_, /*allow_pdl=*/false,
            /*block_x=*/128, /*block_y=*/1, /*block_z=*/1);
      },
      stream_);
}

// Spec conv1d_update (MTP, SPEC1/VL1/SL4): 16 runtime params (source-order:
// x, w, state, indices, num_accepted, query_start_loc, APC placeholders, o,
// batch, NCL, stride_x_token, state strides, stride_o_token). The cubin is
// generated only when the SL4 SPEC1 variant exists (MTP configs); skip otherwise.
TEST_F(GdnKernelLaunchTest, SpecCausalConv1dUpdateLaunches) {
  const std::string dir = Singleton<TritonWrapper>::GetInstance()->GetKernelsDir();
  bool found_sl4 = false;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.path().extension() == ".cubin" &&
        entry.path().filename().string().find(
            "gdn_causal_conv1d_update_kernel_D_5120_KW_4_BN_256_NP2_4_SL_4_") == 0) {
      found_sl4 = true;
      break;
    }
  }
  if (!found_sl4) {
    GTEST_SKIP() << "SL4 SPEC1 conv-update cubin (D=5120) not found under " << dir;
  }

  DecodeBuffers b;
  b.slot_indices.Upload<int32_t>({1});
  // SL4 conv state: [slots, D, 4] bf16 (wider than the SL3 DecodeBuffers.state).
  DeviceBuffer spec_state(static_cast<size_t>(kNumSlots) * kConvDim * 4 * kBf16Bytes);
  DeviceBuffer num_accepted(1 * kInt32Bytes);
  DeviceBuffer query_start_loc(2 * kInt32Bytes);
  num_accepted.Upload<int32_t>({1});
  query_start_loc.Upload<int32_t>({0, 2});

  auto* x_ptr = b.x.As<__nv_bfloat16>();
  void* w_ptr = b.conv_w.data();
  void* state_ptr = spec_state.data();
  void* indices_ptr = b.slot_indices.data();
  void* num_acc_ptr = num_accepted.data();
  void* qsl_ptr = query_start_loc.data();
  void* apc1_ptr = indices_ptr;  // dead (APC=0)
  void* apc2_ptr = indices_ptr;  // dead (APC=0)
  auto* o_ptr = b.x.As<__nv_bfloat16>();
  int32_t batch = 1;
  int64_t stride_x_token = kConvDim;
  int32_t slot_stride_elems = static_cast<int32_t>(kNumSlots * kConvDim * 4);  // bf16 elems
  int32_t state_dim_stride = 4;                                                // SL4
  int32_t state_tok_stride = 1;
  int64_t stride_o_token = kConvDim;
  void* c1d_args[16] = {&x_ptr,
                        &w_ptr,
                        &state_ptr,
                        &indices_ptr,
                        &num_acc_ptr,
                        &qsl_ptr,
                        &apc1_ptr,
                        &apc2_ptr,
                        &o_ptr,
                        &batch,
                        const_cast<int*>(&kGdnNumCacheLines),
                        &stride_x_token,
                        &slot_stride_elems,
                        &state_dim_stride,
                        &state_tok_stride,
                        &stride_o_token};

  std::unordered_map<std::string, std::string> c1d_map = {{"D", std::to_string(kConvDim)},
                                                          {"KW", std::to_string(kConvKernelWidth)},
                                                          {"BN", "256"},
                                                          {"NP2", "4"},
                                                          {"SL", "4"},
                                                          {"HB", "0"},
                                                          {"SILU", "1"},
                                                          {"VL", "1"},
                                                          {"APC", "0"},
                                                          {"SPEC", "1"},
                                                          {"NULL", "1"},
                                                          {"W", "4"},
                                                          {"S", "2"},
                                                          {"compute_type", "BF16"}};

  const size_t c1d_grid_y = (static_cast<size_t>(kConvDim) + 255) / 256;
  ExpectLaunchOk(
      [&] {
        Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
            "gdn_causal_conv1d_update_kernel", c1d_args, c1d_map,
            /*grid_x=*/1, /*grid_y=*/c1d_grid_y, /*grid_z=*/1, stream_, /*allow_pdl=*/false,
            /*block_x=*/128, /*block_y=*/1, /*block_z=*/1);
      },
      stream_);
}

// fused_recurrent_fwd (MTP spec recurrent): 16 runtime params (q,k,v,g,beta,o,
// h0,ht,cu,ssm,num_accepted,scale,N,T,stride_init,stride_final). stride_* runtime
// (do_not_specialize). Skip when the spec cubin isn't generated.
TEST_F(GdnKernelLaunchTest, FusedRecurrentFwdLaunches) { RunFusedRecurrentFwd(1); }

TEST_F(GdnKernelLaunchTest, FusedRecurrentFwdLaunchesMtp2) { RunFusedRecurrentFwd(2); }

// packed_decode: fused recurrent update (12 params; verified ABI).
TEST_F(GdnKernelLaunchTest, PackedDecodeLaunches) {
  DecodeBuffers b;
  b.slot_indices.Upload<int32_t>({1});
  void* mixed = b.x.data();
  void* a_p = b.a.data();
  void* b_p = b.b.data();
  void* alog = b.a_log.data();
  void* dtbias = b.dt_bias.data();
  void* o = b.o.data();
  void* h0 = b.rec_state.data();
  void* ht = b.rec_state.data();
  void* ssm = b.slot_indices.data();
  float scale_packed = 1.0f / std::sqrt(static_cast<float>(kHeadK));
  int32_t pd_slot_stride = static_cast<int32_t>(kNumValueHeads) * kHeadV * kHeadK;
  void* pd_args[12] = {&mixed, &a_p, &b_p, &alog,         &dtbias,         &o,
                       &h0,    &ht,  &ssm, &scale_packed, &pd_slot_stride, &pd_slot_stride};

  std::unordered_map<std::string, std::string> pd_map = {{"SMQ", std::to_string(kConvDim)},
                                                         {"SA", std::to_string(2 * kNumValueHeads)},
                                                         {"SB", std::to_string(2 * kNumValueHeads)},
                                                         {"SISQ", "1"},
                                                         {"H", std::to_string(kNumKeyHeads)},
                                                         {"HV", std::to_string(kNumValueHeads)},
                                                         {"K", std::to_string(kHeadK)},
                                                         {"V", std::to_string(kHeadV)},
                                                         {"BK", "128"},
                                                         {"BV", "32"},
                                                         {"SPT", "20.0"},
                                                         {"L2N", "1"},
                                                         {"W", "1"},
                                                         {"S", "3"},
                                                         {"compute_type", "BF16"}};

  const size_t pd_nv = (static_cast<size_t>(kHeadV) + 31) / 32;
  ExpectLaunchOk(
      [&] {
        Singleton<TritonWrapper>::GetInstance()->InvokeTritonKernel(
            "fused_recurrent_gated_delta_rule_packed_decode_kernel", pd_args, pd_map,
            /*grid_x=*/pd_nv, /*grid_y=*/kNumValueHeads, /*grid_z=*/1, stream_, /*allow_pdl=*/false,
            /*block_x=*/32, /*block_y=*/1, /*block_z=*/1);
      },
      stream_);
}

}  // namespace ksana_llm
