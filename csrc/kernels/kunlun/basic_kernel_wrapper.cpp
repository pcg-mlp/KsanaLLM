/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "kernels/kunlun/basic_kernel_wrapper.h"

#include <infer_ops.h>         // infer_ops::swiglu, infer_ops::add_rms_layer_norm
#include <xblas_legacy_api.h>  // xblas::fc_fusion (bf16 GEMM)
#include <xpu/xdnn.h>

// xpytorch::xpu::api host APIs (libxdnn_pytorch.so) — same path XSGLANG /
// torch_xmlir backend uses for torch.Tensor.scatter_ / scatter_nd. The mirror
// xdnn (= baidu::xpu::api) versions in libxpuapi.so hang on this build, so for
// scatter / scatter_nd we go through these wrappers instead. Same signatures.
#include "runtime/xpu_context.h"
#include "xpu/refactor/xpukernel_math.h"
// xdnn_pytorch Tensor-based bridge APIs (libxdnn_pytorch.so). Used for ops whose
// XSGLANG-aligned device kernel is only reachable through this layer — e.g.
// swiglu_forward lowers to `swish_glu_sdnn` (SDNN), whereas the raw-pointer
// refactor API xpukernel::swiglu lowers to a different kernel `swiglu_fwd`.
#include "xdnn_pytorch/xdnn_pytorch.h"
namespace xpukernel = xpytorch::xpu::api;

namespace {
// xpytorch context relies on a unify allocator for workspace alloc inside
// scatter; without one, the allocator helper aborts via spdlog. Plug in
// xpu_malloc / xpu_free once per process. (XSGLANG gets this for free because
// PyTorch installs its own allocator at framework init.)
struct XpuKernelAllocatorInstaller {
  XpuKernelAllocatorInstaller() {
    // The xpytorch ctx's unify_allocator is used for *device* workspace
    // allocations (it gets the result back via memcpy-on-device). Returning
    // host memory here SEGVs in xpytorch::xpu::api::do_host2device.
    // xpu_malloc with XPU_MEM_MAIN matches what XSGLANG's c10 allocator
    // returns under the hood (a device pointer).
    auto alloc_fn = +[](size_t bytes) -> void* {
      void* p = nullptr;
      if (xpu_malloc(&p, bytes, XPU_MEM_MAIN) != 0) return nullptr;
      return p;
    };
    auto free_fn = +[](void* p) {
      if (p) xpu_free(p);
    };
    xmlir_rt::set_unify_allocator(xmlir_rt::UnifyAllocator{alloc_fn, free_fn});
  }
};
static XpuKernelAllocatorInstaller g_xpukernel_allocator_installer;

// Map a KsanaLLM element type to the xmlir ScalarType carried by an
// xdnn_pytorch::Tensor. ksana_llm::float16/bfloat16 are 2-byte int typedefs
// sharing the memory layout of xmlir's ::float16/::bfloat16 (see XdnnType).
template <typename T>
inline xdnn_pytorch::ScalarType XmlirScalarType() {
  static_assert(sizeof(T) == 0, "XmlirScalarType: unsupported element type");
  return xdnn_pytorch::ScalarType::UnknownScalarType;
}
template <>
inline xdnn_pytorch::ScalarType XmlirScalarType<float>() {
  return xdnn_pytorch::ScalarType::kfloat32;
}
template <>
inline xdnn_pytorch::ScalarType XmlirScalarType<ksana_llm::float16>() {
  return xdnn_pytorch::ScalarType::kfloat16;
}
template <>
inline xdnn_pytorch::ScalarType XmlirScalarType<ksana_llm::bfloat16>() {
  return xdnn_pytorch::ScalarType::kbfloat16;
}

// Wrap a raw device pointer as a row-major contiguous xdnn_pytorch::Tensor view
// so it can be passed to the Tensor-based xpytorch bridge APIs (swiglu_forward,
// rms_layer_norm, ...) that XSGLANG reaches via torch.ops.custom_ops.*.
inline xdnn_pytorch::Tensor MakeXmlirTensor(void* storage, int64_t rows, int64_t cols, xdnn_pytorch::ScalarType dtype) {
  xdnn_pytorch::Shape sizes{rows, cols};
  xdnn_pytorch::Stride strides{cols, 1};  // row-major contiguous
  return xdnn_pytorch::Tensor(sizes, strides, dtype, storage, xdnn_pytorch::MemoryFormat::Default,
                              /*offset=*/0, /*is_cpu=*/false);
}
}  // namespace

#ifdef USE_CUSTOM_XPU_KERNELS
extern "C" int launch_vector_add_kernel(const float* a, const float* b, float* c, int n, void* stream);
#endif

namespace xblas = baidu::xpu::xblas;

namespace ksana_llm {

// ── Elementwise ops ───────────────────────────────────────────────────────────
//
// XSGLANG NEVER dispatches a standalone bf16/fp16 elementwise calc kernel —
// `torch.add`/`torch.mul` on the residual stream are fused into
// `infer_rms_layer_norm_ns` (Add+RMSNorm) and `swish_glu_sdnn` (SiLU+Mul) at
// the model-level graph layer.  KSANA's main residual / decoder path likewise
// runs `infer_ops::add_rms_layer_norm` / `infer_ops::swiglu` (see
// attention_kernel_wrapper.cpp), so the inference hot path produces no
// bf16 `calc_basic` either.
//
// The bf16/fp16 instantiations of XpuAdd below remain for model-level helpers
// (DeepStack residual injection in Qwen3-VL, generic AddLayer used by special
// models) and for unit tests.  When those non-fused paths execute they DO emit
// `xpukernel_xpu3::calc_basic<0, bfloat16>` — that kernel was visible in the
// 6/16 KSANA trace but has been eliminated from the 6/22 trace once sampling
// stopped routing through bf16 add.  Any residual occurrence corresponds to a
// model-specific code path that XSGLANG fuses upstream; aligning those would
// require fusing at the model graph layer, not at the kernel wrapper.

template <typename T>
int XpuAdd(xpukernel::Context* ctx, const T* x, const T* y, T* z, int64_t len) {
  return xpukernel::add<xdnn_t<T>>(ctx, xcast(x), xcast(y), xcast_m(z), len);
}

template int XpuAdd<float>(xpukernel::Context*, const float*, const float*, float*, int64_t);
template int XpuAdd<float16>(xpukernel::Context*, const float16*, const float16*, float16*, int64_t);
template int XpuAdd<bfloat16>(xpukernel::Context*, const bfloat16*, const bfloat16*, bfloat16*, int64_t);

template <typename TX, typename TY>
int XpuCast(xpukernel::Context* ctx, const TX* x, TY* y, int64_t len) {
  return xpukernel::cast<xdnn_t<TX>, xdnn_t<TY>>(ctx, xcast(x), xcast_m(y), len);
}

template int XpuCast<float, float16>(xpukernel::Context*, const float*, float16*, int64_t);
template int XpuCast<float16, float>(xpukernel::Context*, const float16*, float*, int64_t);
template int XpuCast<float, bfloat16>(xpukernel::Context*, const float*, bfloat16*, int64_t);
template int XpuCast<bfloat16, float>(xpukernel::Context*, const bfloat16*, float*, int64_t);
template int XpuCast<float16, bfloat16>(xpukernel::Context*, const float16*, bfloat16*, int64_t);
template int XpuCast<bfloat16, float16>(xpukernel::Context*, const bfloat16*, float16*, int64_t);
template int XpuCast<float, float>(xpukernel::Context*, const float*, float*, int64_t);
template int XpuCast<float16, float16>(xpukernel::Context*, const float16*, float16*, int64_t);
template int XpuCast<bfloat16, bfloat16>(xpukernel::Context*, const bfloat16*, bfloat16*, int64_t);
template int XpuCast<int32_t, int64_t>(xpukernel::Context*, const int32_t*, int64_t*, int64_t);
template int XpuCast<int64_t, int32_t>(xpukernel::Context*, const int64_t*, int32_t*, int64_t);

template <typename T>
int XpuSilu(xpukernel::Context* ctx, const T* x, T* y, int64_t len) {
  return xpukernel::silu<xdnn_t<T>>(ctx, xcast(x), xcast_m(y), len);
}

template int XpuSilu<float>(xpukernel::Context*, const float*, float*, int64_t);
template int XpuSilu<float16>(xpukernel::Context*, const float16*, float16*, int64_t);
template int XpuSilu<bfloat16>(xpukernel::Context*, const bfloat16*, bfloat16*, int64_t);

template <typename T>
int XpuSiluMul(xpukernel::Context* ctx, const T* input, const T* gate, T* output, int64_t m, int64_t n) {
  int64_t len = m * n;
  int ret = xpukernel::silu<xdnn_t<T>>(ctx, xcast(input), xcast_m(output), len);
  if (ret != 0) return ret;
  return xpukernel::mul<xdnn_t<T>>(ctx, xcast(output), xcast(gate), xcast_m(output), len);
}

template int XpuSiluMul<float>(xpukernel::Context*, const float*, const float*, float*, int64_t, int64_t);
template int XpuSiluMul<float16>(xpukernel::Context*, const float16*, const float16*, float16*, int64_t, int64_t);
template int XpuSiluMul<bfloat16>(xpukernel::Context*, const bfloat16*, const bfloat16*, bfloat16*, int64_t, int64_t);

template <typename T>
int XpuRowBasedSiluMul(xdnn::Context* ctx, const T* input, T* output, int64_t m, int64_t n) {
  // Align with XSGLANG's silu_and_mul C++ binding under XSGL_FAST_SWIGLU=1
  // (vA optimized launch in profile/run_server_vA.sh):
  //   sgl-kernel/csrc/elementwise/activation.cc:11
  //     xdnn::fast_swiglu(ctx, x, y, {m, 2n}, /*axis=*/1, /*turn=*/true)
  // This is the path that produces the SDNN device kernel
  //   xpu3::swish_glu_sdnn<bfloat16>(in*, out*, m, n, turn, max_x*, ?, max_y*, ?)
  // observed in the 6/16 XSGLANG trace.
  //
  // Without the XSGL_FAST_SWIGLU env, XSGLANG falls back to infer_ops::swiglu;
  // since vA / ksana profiling always opts in to the optimized swiglu, we
  // align on fast_swiglu here.
  std::vector<int64_t> xshape = {m, 2 * n};
  return xdnn::fast_swiglu<xdnn_t<T>>(ctx, xcast(input), xcast_m(output), xshape, /*axis=*/1, /*turn=*/true);
}

template int XpuRowBasedSiluMul<float>(xdnn::Context*, const float*, float*, int64_t, int64_t);
template int XpuRowBasedSiluMul<float16>(xdnn::Context*, const float16*, float16*, int64_t, int64_t);
template int XpuRowBasedSiluMul<bfloat16>(xdnn::Context*, const bfloat16*, bfloat16*, int64_t, int64_t);

template <typename T>
int XpuGatedGelu(xpukernel::Context* ctx, const T* input, const T* gated_weight, T* output, int64_t m, int64_t n) {
  int ret = xpukernel::gelu<xdnn_t<T>>(ctx, xcast(input), xcast_m(output), m * n,
                                       /*approximate=*/false);
  if (ret != 0) return ret;
  if (gated_weight) {
    ret = xpukernel::mul<xdnn_t<T>>(ctx, xcast(output), xcast(gated_weight), xcast_m(output), m * n);
  }
  return ret;
}

template int XpuGatedGelu<float>(xpukernel::Context*, const float*, const float*, float*, int64_t, int64_t);
template int XpuGatedGelu<float16>(xpukernel::Context*, const float16*, const float16*, float16*, int64_t, int64_t);
template int XpuGatedGelu<bfloat16>(xpukernel::Context*, const bfloat16*, const bfloat16*, bfloat16*, int64_t, int64_t);

template <typename T>
int XpuEmbeddingLookup(xpukernel::Context* ctx, const int64_t* input_ids, const T* weight, T* output, int64_t token_num,
                       int64_t hidden_units, int64_t vocab_size) {
  return xpukernel::embedding<xdnn_t<T>, int64_t>(ctx, xcast(weight), input_ids, xcast_m(output), vocab_size,
                                                  hidden_units, token_num, /*padding_idx=*/-1);
}

template int XpuEmbeddingLookup<float>(xpukernel::Context*, const int64_t*, const float*, float*, int64_t, int64_t,
                                       int64_t);
template int XpuEmbeddingLookup<float16>(xpukernel::Context*, const int64_t*, const float16*, float16*, int64_t,
                                         int64_t, int64_t);
template int XpuEmbeddingLookup<bfloat16>(xpukernel::Context*, const int64_t*, const bfloat16*, bfloat16*, int64_t,
                                          int64_t, int64_t);

template <typename T>
int XpuCopy2d(xpukernel::Context* ctx, const T* src, T* dst, int64_t m, int64_t ldx, int64_t ldy, int64_t n) {
  return xpukernel::copy2d<xdnn_t<T>>(ctx, xcast(src), xcast_m(dst), m, ldx, ldy, n);
}

template int XpuCopy2d<float16>(xpukernel::Context*, const float16*, float16*, int64_t, int64_t, int64_t, int64_t);
template int XpuCopy2d<bfloat16>(xpukernel::Context*, const bfloat16*, bfloat16*, int64_t, int64_t, int64_t, int64_t);
template int XpuCopy2d<float>(xpukernel::Context*, const float*, float*, int64_t, int64_t, int64_t, int64_t);

// ── Linear ops ────────────────────────────────────────────────────────────────

template <typename TX, typename TW, typename TY, typename TGEMM>
int XpuFc(xpukernel::Context* ctx, const TX* x, const TW* w, TY* y, int64_t m, int64_t n, int64_t k, bool x_trans,
          bool w_trans, const float* x_maxptr, const float* w_maxptr, float* y_maxptr) {
  return xpukernel::fc<xdnn_t<TX>, xdnn_t<TW>, xdnn_t<TY>, TGEMM>(ctx, xcast(x), xcast(w), xcast_m(y), m, n, k, x_trans,
                                                                  w_trans, x_maxptr, w_maxptr, y_maxptr);
}

template int XpuFc<float, float, float, int16_t>(xpukernel::Context*, const float*, const float*, float*, int64_t,
                                                 int64_t, int64_t, bool, bool, const float*, const float*, float*);
template int XpuFc<float16, float16, float16, int16_t>(xpukernel::Context*, const float16*, const float16*, float16*,
                                                       int64_t, int64_t, int64_t, bool, bool, const float*,
                                                       const float*, float*);

// bf16 fc: xdnn::fc<bfloat16> not exported; route through xblas::fc_fusion (libxpu_blas.so).
// Symbol verified: baidu::xpu::xblas::fc_fusion<bfloat16, bfloat16, bfloat16, float, float, float>

// ── FcFusion (xblas) ──────────────────────────────────────────────────────────

// Type traits for fc_fusion — determines internal compute types per data type.
// Mirrors xsglang klx_gemm.cc:FC_T specialization pattern.
template <typename T>
struct FcFusionTypeTraits;

template <>
struct FcFusionTypeTraits<float16> {
  using TGEMM = xdnn_fp16;  // float16 GEMM accumulation
  using TGEMM_O = float;
  using TINTER_RES = xdnn_fp16;
};

template <>
struct FcFusionTypeTraits<bfloat16> {
  using TGEMM = float;  // float32 GEMM accumulation for bf16 numerical stability
  using TGEMM_O = float;
  using TINTER_RES = xdnn_bf16;
};

template <>
struct FcFusionTypeTraits<float> {
  using TGEMM = float;
  using TGEMM_O = float;
  using TINTER_RES = float;
};

template <typename TX, typename TW, typename TY>
int XpuFcFusion(xdnn::Context* ctx, const TX* x, const TW* w, TY* y, int64_t m, int64_t n, int64_t k, bool x_trans,
                bool w_trans, const float* x_maxptr, const float* w_maxptr, float* y_maxptr) {
  using XT = xdnn_t<TX>;
  using WT = xdnn_t<TW>;
  using YT = xdnn_t<TY>;
  using Traits = FcFusionTypeTraits<TX>;
  using TGEMM = typename Traits::TGEMM;
  using TGEMM_O = typename Traits::TGEMM_O;
  using TINTER_RES = typename Traits::TINTER_RES;
  using TBIAS = float;
  using TSCALE = float;

  // ctx_guard: manages xblas workspace memory (L3/GM) for this operation.
  // Without this, xblas::fc_fusion may use stale/conflicting workspace memory.
  // This matches xsglang klx_gemm.cc line 153: `api::ctx_guard guard(ctx);`
  xdnn::ctx_guard guard(ctx);

  int64_t ldx = x_trans ? m : k;
  int64_t ldw = w_trans ? k : n;
  int64_t ldy = n;

  xblas::FcFusionTensor<const XT> a{xcast(x), x_maxptr, x_trans ? k : m, x_trans ? m : k, ldx, x_trans};
  xblas::FcFusionTensor<const WT> b{xcast(w), w_maxptr, w_trans ? n : k, w_trans ? k : n, ldw, w_trans};
  xblas::FcFusionTensor<const YT> c{nullptr, nullptr, m, n, ldy, false};
  xblas::FcFusionTensor<YT> d{xcast_m(y), y_maxptr, m, n, ldy, false};

  xblas::FcFusionDesc<TGEMM, TGEMM_O, TINTER_RES> desc{1.0f, 0.0f, nullptr};
  xblas::FcFusionEpilogue<TBIAS, TSCALE> epilogue{
      xdnn::Activation_t::LINEAR,  // act
      nullptr,                     // bias
      nullptr,                     // scale_a
      nullptr,                     // scale_b
      0,                           // scale_a_mode
      0,                           // scale_b_mode
      y_maxptr                     // res_maxptr
  };
  xblas::FcFusionMultiStreamDesc multi_stream_desc{};

  int ret = xblas::fc_fusion(ctx, a, b, c, d, desc, epilogue, multi_stream_desc);
  // 同步：确保 kernel 完成后再让 ctx_guard 析构释放 workspace，防止下一个 kernel 复用未完成的 L3/GM 地址
  xpu_wait(ctx->xpu_stream);
  return ret;
}

template int XpuFcFusion<float16, float16, float16>(xdnn::Context*, const float16*, const float16*, float16*, int64_t,
                                                    int64_t, int64_t, bool, bool, const float*, const float*, float*);
template int XpuFcFusion<bfloat16, bfloat16, bfloat16>(xdnn::Context*, const bfloat16*, const bfloat16*, bfloat16*,
                                                       int64_t, int64_t, int64_t, bool, bool, const float*,
                                                       const float*, float*);
template int XpuFcFusion<float, float, float>(xdnn::Context*, const float*, const float*, float*, int64_t, int64_t,
                                              int64_t, bool, bool, const float*, const float*, float*);

// ── Norm ops ──────────────────────────────────────────────────────────────────

template <typename T, typename TW>
int XpuLayerNorm(xpukernel::Context* ctx, const T* x, T* y, int64_t m, int64_t n, float eps, const TW* scale,
                 const TW* bias, float* mean, float* var) {
  return xpukernel::layer_norm<xdnn_t<T>, xdnn_t<TW>>(ctx, xcast(x), xcast_m(y), m, n, eps, xcast(scale), xcast(bias),
                                                      mean, var);
}

template int XpuLayerNorm<float, float>(xpukernel::Context*, const float*, float*, int64_t, int64_t, float,
                                        const float*, const float*, float*, float*);
template int XpuLayerNorm<float16, float16>(xpukernel::Context*, const float16*, float16*, int64_t, int64_t, float,
                                            const float16*, const float16*, float*, float*);
template int XpuLayerNorm<bfloat16, bfloat16>(xpukernel::Context*, const bfloat16*, bfloat16*, int64_t, int64_t, float,
                                              const bfloat16*, const bfloat16*, float*, float*);

template <typename T, typename TW, typename TY>
int XpuRmsLayerNorm(xdnn::Context* ctx, const T* x, TY* y, int64_t m, int64_t n, float eps, const TW* scale,
                    const TW* bias, float* var) {
  // XSGLANG trace shows rms_layer_norm device kernel `infer_rms_layer_norm_ns`
  // (864 calls), launched via infer_ops::rms_layer_norm — NOT xpukernel's
  // rms_layer_norm (which would lower to `rms_layer_norm_fwd`, a different
  // device kernel). Match XSGLANG by using infer_ops here.
  static_assert(std::is_same_v<T, TY>, "infer_ops::rms_layer_norm requires same input/output dtype");
  return infer_ops::rms_layer_norm<xdnn_t<T>, xdnn_t<TW>>(ctx, xcast(x), xcast_m(y), m, n, eps, xcast(scale),
                                                          xcast(bias), var);
}

template int XpuRmsLayerNorm<float, float, float>(xdnn::Context*, const float*, float*, int64_t, int64_t, float,
                                                  const float*, const float*, float*);
template int XpuRmsLayerNorm<float16, float16, float16>(xdnn::Context*, const float16*, float16*, int64_t, int64_t,
                                                        float, const float16*, const float16*, float*);
template int XpuRmsLayerNorm<bfloat16, bfloat16, bfloat16>(xdnn::Context*, const bfloat16*, bfloat16*, int64_t, int64_t,
                                                           float, const bfloat16*, const bfloat16*, float*);

// Behavior: residual += input; input = rms_norm(residual, weight).
// Aligned with xSGLang Qwen3-14B trace (7/16):
//   add_rms_layer_norm<bf16, bf16>(ctx, x=input, y=residual, z=input, m, n, eps,
//                                  scale, bias=NULL, var=NULL, z_add=residual,
//                                  maxptr=NULL, force_sdnn=false, is_gemma=false)
template <typename T, typename TW>
int XpuFusedAddRmsNorm(xdnn::Context* ctx, T* input, T* residual, const TW* weight, int64_t m, int64_t n, float eps) {
  return infer_ops::add_rms_layer_norm<xdnn_t<T>, xdnn_t<TW>>(ctx,
                                                              /*x=*/xcast(input),
                                                              /*y=*/xcast_m(residual),
                                                              /*z=*/xcast_m(input), m, n, eps,
                                                              /*scale=*/xcast(weight),
                                                              /*bias=*/nullptr,
                                                              /*var=*/nullptr,
                                                              /*z_add=*/xcast_m(residual),
                                                              /*maxptr=*/nullptr,
                                                              /*force_sdnn=*/false,
                                                              /*is_gemma=*/false);
}

template int XpuFusedAddRmsNorm<float, float>(xdnn::Context*, float*, float*, const float*, int64_t, int64_t, float);
template int XpuFusedAddRmsNorm<float16, float16>(xdnn::Context*, float16*, float16*, const float16*, int64_t, int64_t,
                                                  float);
template int XpuFusedAddRmsNorm<bfloat16, bfloat16>(xdnn::Context*, bfloat16*, bfloat16*, const bfloat16*, int64_t,
                                                    int64_t, float);

// ── Utility ops (XSGLANG-aligned) ─────────────────────────────────────────────

template <typename T, typename TACC>
int XpuArange(xpukernel::Context* ctx, T* y, TACC begin, TACC step, int64_t len) {
  return xpukernel::arange<xdnn_t<T>, TACC>(ctx, xcast_m(y), begin, step, len);
}

template int XpuArange<float16, float>(xpukernel::Context*, float16*, float, float, int64_t);
template int XpuArange<bfloat16, float>(xpukernel::Context*, bfloat16*, float, float, int64_t);
template int XpuArange<float, float>(xpukernel::Context*, float*, float, float, int64_t);
template int XpuArange<int32_t, int64_t>(xpukernel::Context*, int32_t*, int64_t, int64_t, int64_t);
template int XpuArange<int64_t, int64_t>(xpukernel::Context*, int64_t*, int64_t, int64_t, int64_t);

// float16 / bfloat16 / float: arange<T, float> is the library's API
// Integer paths the library exports: int32_t with int64 accumulator (the only
// available int32 variant), and int64/int64 — matches XSGLANG trace kernel
// `arange<long, long>`.

template <typename T>
int XpuConstant(xpukernel::Context* ctx, T* x, int64_t len, T val) {
  return xpukernel::constant<xdnn_t<T>>(ctx, xcast_m(x), len, reinterpret_cast<xdnn_t<T>&>(val));
}

template int XpuConstant<float>(xpukernel::Context*, float*, int64_t, float);
template int XpuConstant<float16>(xpukernel::Context*, float16*, int64_t, float16);
template int XpuConstant<bfloat16>(xpukernel::Context*, bfloat16*, int64_t, bfloat16);
template int XpuConstant<int32_t>(xpukernel::Context*, int32_t*, int64_t, int32_t);
template int XpuConstant<int64_t>(xpukernel::Context*, int64_t*, int64_t, int64_t);

template <typename T>
int XpuNeg(xpukernel::Context* ctx, const T* x, T* y, int64_t len) {
  return xpukernel::neg<xdnn_t<T>>(ctx, xcast(x), xcast_m(y), len);
}

template int XpuNeg<float>(xpukernel::Context*, const float*, float*, int64_t);
template int XpuNeg<float16>(xpukernel::Context*, const float16*, float16*, int64_t);
template int XpuNeg<bfloat16>(xpukernel::Context*, const bfloat16*, bfloat16*, int64_t);

template <typename T>
int XpuClamp(xpukernel::Context* ctx, const T* x, T* y, int64_t len, T min_val, T max_val) {
  return xpukernel::clamp<xdnn_t<T>>(ctx, xcast(x), xcast_m(y), len, reinterpret_cast<xdnn_t<T>&>(min_val),
                                     reinterpret_cast<xdnn_t<T>&>(max_val));
}

template int XpuClamp<float>(xpukernel::Context*, const float*, float*, int64_t, float, float);
template int XpuClamp<float16>(xpukernel::Context*, const float16*, float16*, int64_t, float16, float16);
template int XpuClamp<bfloat16>(xpukernel::Context*, const bfloat16*, bfloat16*, int64_t, bfloat16, bfloat16);

template <typename T>
int XpuAddScalar(xpukernel::Context* ctx, const T* x, T* y, float alpha, float other, int64_t len) {
  return xpukernel::add_scalar<xdnn_t<T>, float>(ctx, xcast(x), xcast_m(y), alpha, other, len);
}

template int XpuAddScalar<float>(xpukernel::Context*, const float*, float*, float, float, int64_t);
template int XpuAddScalar<float16>(xpukernel::Context*, const float16*, float16*, float, float, int64_t);
template int XpuAddScalar<bfloat16>(xpukernel::Context*, const bfloat16*, bfloat16*, float, float, int64_t);

template <typename T>
int XpuSubScalar(xpukernel::Context* ctx, const T* x, T* y, float alpha, float other, int64_t len) {
  return xpukernel::sub_scalar<xdnn_t<T>, float>(ctx, xcast(x), xcast_m(y), alpha, other, len);
}

template int XpuSubScalar<float>(xpukernel::Context*, const float*, float*, float, float, int64_t);
template int XpuSubScalar<float16>(xpukernel::Context*, const float16*, float16*, float, float, int64_t);
template int XpuSubScalar<bfloat16>(xpukernel::Context*, const bfloat16*, bfloat16*, float, float, int64_t);

// ── Index / select ops ─────────────────────────────────────────────────────────

template <typename T, typename TID>
int XpuGather(xpukernel::Context* ctx, const T* x, const TID* index, T* y, const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& idxshape, int64_t axis) {
  int64_t idx_len = 1;
  for (auto d : idxshape) idx_len *= d;
  return xpukernel::index_select<xdnn_t<T>, TID>(ctx, xcast(x), index, xcast_m(y), xshape, idx_len, axis);
}

template int XpuGather<float, int32_t>(xpukernel::Context*, const float*, const int32_t*, float*,
                                       const std::vector<int64_t>&, const std::vector<int64_t>&, int64_t);
template int XpuGather<float, int64_t>(xpukernel::Context*, const float*, const int64_t*, float*,
                                       const std::vector<int64_t>&, const std::vector<int64_t>&, int64_t);
template int XpuGather<float16, int32_t>(xpukernel::Context*, const float16*, const int32_t*, float16*,
                                         const std::vector<int64_t>&, const std::vector<int64_t>&, int64_t);
template int XpuGather<float16, int64_t>(xpukernel::Context*, const float16*, const int64_t*, float16*,
                                         const std::vector<int64_t>&, const std::vector<int64_t>&, int64_t);
template int XpuGather<bfloat16, int32_t>(xpukernel::Context*, const bfloat16*, const int32_t*, bfloat16*,
                                          const std::vector<int64_t>&, const std::vector<int64_t>&, int64_t);
template int XpuGather<bfloat16, int64_t>(xpukernel::Context*, const bfloat16*, const int64_t*, bfloat16*,
                                          const std::vector<int64_t>&, const std::vector<int64_t>&, int64_t);
template int XpuGather<int64_t, int64_t>(xpukernel::Context*, const int64_t*, const int64_t*, int64_t*,
                                         const std::vector<int64_t>&, const std::vector<int64_t>&, int64_t);

template <typename T, typename TID>
int XpuGatherNd(xpukernel::Context* ctx, const T* x, const TID* index, T* y, const std::vector<int64_t>& xshape,
                const std::vector<int64_t>& index_shape) {
  xpukernel::VectorParam<int64_t> xshape_param = {xshape.data(), static_cast<int64_t>(xshape.size()), nullptr};
  return xpukernel::gather_nd<xdnn_t<T>, TID>(ctx, xcast(x), index, xcast_m(y), xshape_param, index_shape);
}

template int XpuGatherNd<float, int32_t>(xpukernel::Context*, const float*, const int32_t*, float*,
                                         const std::vector<int64_t>&, const std::vector<int64_t>&);
template int XpuGatherNd<float, int64_t>(xpukernel::Context*, const float*, const int64_t*, float*,
                                         const std::vector<int64_t>&, const std::vector<int64_t>&);
template int XpuGatherNd<float16, int32_t>(xpukernel::Context*, const float16*, const int32_t*, float16*,
                                           const std::vector<int64_t>&, const std::vector<int64_t>&);
template int XpuGatherNd<float16, int64_t>(xpukernel::Context*, const float16*, const int64_t*, float16*,
                                           const std::vector<int64_t>&, const std::vector<int64_t>&);
template int XpuGatherNd<bfloat16, int32_t>(xpukernel::Context*, const bfloat16*, const int32_t*, bfloat16*,
                                            const std::vector<int64_t>&, const std::vector<int64_t>&);
template int XpuGatherNd<bfloat16, int64_t>(xpukernel::Context*, const bfloat16*, const int64_t*, bfloat16*,
                                            const std::vector<int64_t>&, const std::vector<int64_t>&);

template <typename T, typename TID>
int XpuIndexSelect(xpukernel::Context* ctx, const T* x, const TID* index, T* y, const std::vector<int64_t>& xshape,
                   int64_t index_len, int64_t axis) {
  return xpukernel::index_select<xdnn_t<T>, TID>(ctx, xcast(x), index, xcast_m(y), xshape, index_len, axis);
}

template int XpuIndexSelect<float, int32_t>(xpukernel::Context*, const float*, const int32_t*, float*,
                                            const std::vector<int64_t>&, int64_t, int64_t);
template int XpuIndexSelect<float, int64_t>(xpukernel::Context*, const float*, const int64_t*, float*,
                                            const std::vector<int64_t>&, int64_t, int64_t);
template int XpuIndexSelect<float16, int32_t>(xpukernel::Context*, const float16*, const int32_t*, float16*,
                                              const std::vector<int64_t>&, int64_t, int64_t);
template int XpuIndexSelect<float16, int64_t>(xpukernel::Context*, const float16*, const int64_t*, float16*,
                                              const std::vector<int64_t>&, int64_t, int64_t);
template int XpuIndexSelect<bfloat16, int32_t>(xpukernel::Context*, const bfloat16*, const int32_t*, bfloat16*,
                                               const std::vector<int64_t>&, int64_t, int64_t);
template int XpuIndexSelect<bfloat16, int64_t>(xpukernel::Context*, const bfloat16*, const int64_t*, bfloat16*,
                                               const std::vector<int64_t>&, int64_t, int64_t);
template int XpuIndexSelect<int32_t, int32_t>(xpukernel::Context*, const int32_t*, const int32_t*, int32_t*,
                                              const std::vector<int64_t>&, int64_t, int64_t);
template int XpuIndexSelect<int32_t, int64_t>(xpukernel::Context*, const int32_t*, const int64_t*, int32_t*,
                                              const std::vector<int64_t>&, int64_t, int64_t);
template int XpuIndexSelect<int64_t, int32_t>(xpukernel::Context*, const int64_t*, const int32_t*, int64_t*,
                                              const std::vector<int64_t>&, int64_t, int64_t);
template int XpuIndexSelect<int64_t, int64_t>(xpukernel::Context*, const int64_t*, const int64_t*, int64_t*,
                                              const std::vector<int64_t>&, int64_t, int64_t);

template <typename T, typename TID>
int XpuScatterElement(xpukernel::Context* ctx, const T* x, const T* updates, const TID* index, T* y,
                      const std::vector<int64_t>& xyshape, const std::vector<int64_t>& updshape,
                      const std::vector<int64_t>& idxshape, int64_t axis, int64_t reduction) {
  return xpukernel::scatter_element<xdnn_t<T>, TID>(ctx, xcast(x), xcast(updates), index, xcast_m(y), xyshape, updshape,
                                                    idxshape, axis, reduction);
}

template int XpuScatterElement<float, int32_t>(xpukernel::Context*, const float*, const float*, const int32_t*, float*,
                                               const std::vector<int64_t>&, const std::vector<int64_t>&,
                                               const std::vector<int64_t>&, int64_t, int64_t);
template int XpuScatterElement<float, int64_t>(xpukernel::Context*, const float*, const float*, const int64_t*, float*,
                                               const std::vector<int64_t>&, const std::vector<int64_t>&,
                                               const std::vector<int64_t>&, int64_t, int64_t);
template int XpuScatterElement<float16, int32_t>(xpukernel::Context*, const float16*, const float16*, const int32_t*,
                                                 float16*, const std::vector<int64_t>&, const std::vector<int64_t>&,
                                                 const std::vector<int64_t>&, int64_t, int64_t);
template int XpuScatterElement<float16, int64_t>(xpukernel::Context*, const float16*, const float16*, const int64_t*,
                                                 float16*, const std::vector<int64_t>&, const std::vector<int64_t>&,
                                                 const std::vector<int64_t>&, int64_t, int64_t);
template int XpuScatterElement<bfloat16, int32_t>(xpukernel::Context*, const bfloat16*, const bfloat16*, const int32_t*,
                                                  bfloat16*, const std::vector<int64_t>&, const std::vector<int64_t>&,
                                                  const std::vector<int64_t>&, int64_t, int64_t);
template int XpuScatterElement<bfloat16, int64_t>(xpukernel::Context*, const bfloat16*, const bfloat16*, const int64_t*,
                                                  bfloat16*, const std::vector<int64_t>&, const std::vector<int64_t>&,
                                                  const std::vector<int64_t>&, int64_t, int64_t);

// ── ScatterNd ──────────────────────────────────────────────────────────────────
template <typename T, typename TID>
int XpuScatterNd(xpukernel::Context* ctx, const T* x, const T* updates, T* y, const TID* index,
                 const std::vector<int64_t>& xshape, const std::vector<int64_t>& index_shape, bool is_overwrite) {
  int64_t idx_count = 1;
  for (auto d : index_shape) idx_count *= d;
  xpukernel::VectorParam<TID> index_param = {nullptr, idx_count, const_cast<TID*>(index)};
  xpukernel::VectorParam<int64_t> xshape_param = {xshape.data(), static_cast<int64_t>(xshape.size()), nullptr};
  return xpukernel::scatter_nd<xdnn_t<T>, TID>(ctx, xcast(x), xcast(updates), xcast_m(y), index_param, xshape_param,
                                               index_shape, is_overwrite);
}

template int XpuScatterNd<int32_t, int64_t>(xpukernel::Context*, const int32_t*, const int32_t*, int32_t*,
                                            const int64_t*, const std::vector<int64_t>&, const std::vector<int64_t>&,
                                            bool);
template int XpuScatterNd<bfloat16, int64_t>(xpukernel::Context*, const bfloat16*, const bfloat16*, bfloat16*,
                                             const int64_t*, const std::vector<int64_t>&, const std::vector<int64_t>&,
                                             bool);

// ── Memory / shape ops ─────────────────────────────────────────────────────────

template <typename T>
int XpuStridedSlice(xpukernel::Context* ctx, const T* x, T* y, const std::vector<int64_t>& xshape,
                    const std::vector<int64_t>& starts, const std::vector<int64_t>& ends,
                    const std::vector<int64_t>& strides, int64_t offset) {
  return xpukernel::strided_slice<xdnn_t<T>>(ctx, xcast(x), xcast_m(y), xshape, starts, ends, strides, offset);
}

template int XpuStridedSlice<float>(xpukernel::Context*, const float*, float*, const std::vector<int64_t>&,
                                    const std::vector<int64_t>&, const std::vector<int64_t>&,
                                    const std::vector<int64_t>&, int64_t);
template int XpuStridedSlice<float16>(xpukernel::Context*, const float16*, float16*, const std::vector<int64_t>&,
                                      const std::vector<int64_t>&, const std::vector<int64_t>&,
                                      const std::vector<int64_t>&, int64_t);
template int XpuStridedSlice<bfloat16>(xpukernel::Context*, const bfloat16*, bfloat16*, const std::vector<int64_t>&,
                                       const std::vector<int64_t>&, const std::vector<int64_t>&,
                                       const std::vector<int64_t>&, int64_t);
template int XpuStridedSlice<int32_t>(xpukernel::Context*, const int32_t*, int32_t*, const std::vector<int64_t>&,
                                      const std::vector<int64_t>&, const std::vector<int64_t>&,
                                      const std::vector<int64_t>&, int64_t);
template int XpuStridedSlice<int64_t>(xpukernel::Context*, const int64_t*, int64_t*, const std::vector<int64_t>&,
                                      const std::vector<int64_t>&, const std::vector<int64_t>&,
                                      const std::vector<int64_t>&, int64_t);

template <typename T>
int XpuConcat(xpukernel::Context* ctx, const std::vector<const T*>& x_list, T* y,
              const std::vector<std::vector<int64_t>>& xshape_list, int64_t axis) {
  std::vector<const xdnn_t<T>*> xdnn_list;
  xdnn_list.reserve(x_list.size());
  for (const T* p : x_list) {
    xdnn_list.push_back(xcast(p));
  }
  return xpukernel::concat<xdnn_t<T>>(ctx, xdnn_list, xcast_m(y), xshape_list, axis);
}

template int XpuConcat<float>(xpukernel::Context*, const std::vector<const float*>&, float*,
                              const std::vector<std::vector<int64_t>>&, int64_t);
template int XpuConcat<float16>(xpukernel::Context*, const std::vector<const float16*>&, float16*,
                                const std::vector<std::vector<int64_t>>&, int64_t);
template int XpuConcat<bfloat16>(xpukernel::Context*, const std::vector<const bfloat16*>&, bfloat16*,
                                 const std::vector<std::vector<int64_t>>&, int64_t);
template int XpuConcat<int32_t>(xpukernel::Context*, const std::vector<const int32_t*>&, int32_t*,
                                const std::vector<std::vector<int64_t>>&, int64_t);
template int XpuConcat<int64_t>(xpukernel::Context*, const std::vector<const int64_t*>&, int64_t*,
                                const std::vector<std::vector<int64_t>>&, int64_t);

// ── Split ─────────────────────────────────────────────────────────────────────
// Splits a tensor into multiple outputs along axis (inverse of concat).

template <typename T>
int XpuSplit(xpukernel::Context* ctx, const T* x, const std::vector<T*>& y_list,
             const std::vector<int64_t>& xshape, const std::vector<int64_t>& split_list, int64_t axis) {
  std::vector<xdnn_t<T>*> xdnn_list;
  xdnn_list.reserve(y_list.size());
  for (T* p : y_list) {
    xdnn_list.push_back(xcast_m(p));
  }
  return xpukernel::split<xdnn_t<T>>(ctx, xcast(x), xdnn_list, xshape, split_list, axis);
}

template int XpuSplit<float>(xpukernel::Context*, const float*, const std::vector<float*>&,
                             const std::vector<int64_t>&, const std::vector<int64_t>&, int64_t);
template int XpuSplit<float16>(xpukernel::Context*, const float16*, const std::vector<float16*>&,
                               const std::vector<int64_t>&, const std::vector<int64_t>&, int64_t);
template int XpuSplit<bfloat16>(xpukernel::Context*, const bfloat16*, const std::vector<bfloat16*>&,
                                const std::vector<int64_t>&, const std::vector<int64_t>&, int64_t);
template int XpuSplit<int32_t>(xpukernel::Context*, const int32_t*, const std::vector<int32_t*>&,
                               const std::vector<int64_t>&, const std::vector<int64_t>&, int64_t);
template int XpuSplit<int64_t>(xpukernel::Context*, const int64_t*, const std::vector<int64_t*>&,
                               const std::vector<int64_t>&, const std::vector<int64_t>&, int64_t);

template <typename T>
int XpuCumsum(xpukernel::Context* ctx, const T* x, T* y, const std::vector<int64_t>& xshape, bool reverse,
              bool exclusive, int64_t axis) {
  return xpukernel::cumsum<xdnn_t<T>>(ctx, xcast(x), xcast_m(y), xshape, reverse, exclusive, axis);
}

template int XpuCumsum<float>(xpukernel::Context*, const float*, float*, const std::vector<int64_t>&, bool, bool,
                              int64_t);
template int XpuCumsum<float16>(xpukernel::Context*, const float16*, float16*, const std::vector<int64_t>&, bool, bool,
                                int64_t);
template int XpuCumsum<bfloat16>(xpukernel::Context*, const bfloat16*, bfloat16*, const std::vector<int64_t>&, bool,
                                 bool, int64_t);
template int XpuCumsum<int32_t>(xpukernel::Context*, const int32_t*, int32_t*, const std::vector<int64_t>&, bool, bool,
                                int64_t);

// ── Sort / compare ops ─────────────────────────────────────────────────────────

template <typename T, typename TID>
int XpuSort(xpukernel::Context* ctx, const T* x, T* y, TID* index, int64_t m, int64_t n, bool compare) {
  return xpukernel::sort<xdnn_t<T>, TID>(ctx, xcast(x), xcast_m(y), index, m, n, compare);
}

template int XpuSort<float, int64_t>(xpukernel::Context*, const float*, float*, int64_t*, int64_t, int64_t, bool);
template int XpuSort<float, int32_t>(xpukernel::Context*, const float*, float*, int32_t*, int64_t, int64_t, bool);
template int XpuSort<float16, int64_t>(xpukernel::Context*, const float16*, float16*, int64_t*, int64_t, int64_t, bool);
template int XpuSort<bfloat16, int64_t>(xpukernel::Context*, const bfloat16*, bfloat16*, int64_t*, int64_t, int64_t,
                                        bool);

template <typename T>
int XpuEqScalar(xpukernel::Context* ctx, const T* x, bool* y, T other, int64_t len) {
  return xpukernel::eq_scalar<xdnn_t<T>>(ctx, xcast(x), y, reinterpret_cast<xdnn_t<T>&>(other), len);
}

template int XpuEqScalar<float>(xpukernel::Context*, const float*, bool*, float, int64_t);
template int XpuEqScalar<float16>(xpukernel::Context*, const float16*, bool*, float16, int64_t);
template int XpuEqScalar<bfloat16>(xpukernel::Context*, const bfloat16*, bool*, bfloat16, int64_t);
template int XpuEqScalar<int32_t>(xpukernel::Context*, const int32_t*, bool*, int32_t, int64_t);
template int XpuEqScalar<int64_t>(xpukernel::Context*, const int64_t*, bool*, int64_t, int64_t);

template <typename T>
int XpuNeScalar(xpukernel::Context* ctx, const T* x, bool* y, T other, int64_t len) {
  return xpukernel::ne_scalar<xdnn_t<T>>(ctx, xcast(x), y, reinterpret_cast<xdnn_t<T>&>(other), len);
}

template int XpuNeScalar<float>(xpukernel::Context*, const float*, bool*, float, int64_t);
template int XpuNeScalar<float16>(xpukernel::Context*, const float16*, bool*, float16, int64_t);
template int XpuNeScalar<bfloat16>(xpukernel::Context*, const bfloat16*, bool*, bfloat16, int64_t);
template int XpuNeScalar<int32_t>(xpukernel::Context*, const int32_t*, bool*, int32_t, int64_t);
template int XpuNeScalar<int64_t>(xpukernel::Context*, const int64_t*, bool*, int64_t, int64_t);

template <typename T>
int XpuGtScalar(xpukernel::Context* ctx, const T* x, bool* y, T other, int64_t len) {
  return xpukernel::gt_scalar<xdnn_t<T>>(ctx, xcast(x), y, reinterpret_cast<xdnn_t<T>&>(other), len);
}

template int XpuGtScalar<float>(xpukernel::Context*, const float*, bool*, float, int64_t);
template int XpuGtScalar<float16>(xpukernel::Context*, const float16*, bool*, float16, int64_t);
template int XpuGtScalar<bfloat16>(xpukernel::Context*, const bfloat16*, bool*, bfloat16, int64_t);
template int XpuGtScalar<int32_t>(xpukernel::Context*, const int32_t*, bool*, int32_t, int64_t);
template int XpuGtScalar<int64_t>(xpukernel::Context*, const int64_t*, bool*, int64_t, int64_t);

template <typename T>
int XpuGeScalar(xpukernel::Context* ctx, const T* x, bool* y, T other, int64_t len) {
  return xpukernel::ge_scalar<xdnn_t<T>>(ctx, xcast(x), y, reinterpret_cast<xdnn_t<T>&>(other), len);
}

template int XpuGeScalar<float>(xpukernel::Context*, const float*, bool*, float, int64_t);
template int XpuGeScalar<float16>(xpukernel::Context*, const float16*, bool*, float16, int64_t);
template int XpuGeScalar<bfloat16>(xpukernel::Context*, const bfloat16*, bool*, bfloat16, int64_t);
template int XpuGeScalar<int32_t>(xpukernel::Context*, const int32_t*, bool*, int32_t, int64_t);
template int XpuGeScalar<int64_t>(xpukernel::Context*, const int64_t*, bool*, int64_t, int64_t);

template <typename T>
int XpuLtScalar(xpukernel::Context* ctx, const T* x, bool* y, T other, int64_t len) {
  return xpukernel::lt_scalar<xdnn_t<T>>(ctx, xcast(x), y, reinterpret_cast<xdnn_t<T>&>(other), len);
}

template int XpuLtScalar<float>(xpukernel::Context*, const float*, bool*, float, int64_t);
template int XpuLtScalar<float16>(xpukernel::Context*, const float16*, bool*, float16, int64_t);
template int XpuLtScalar<bfloat16>(xpukernel::Context*, const bfloat16*, bool*, bfloat16, int64_t);
template int XpuLtScalar<int32_t>(xpukernel::Context*, const int32_t*, bool*, int32_t, int64_t);
template int XpuLtScalar<int64_t>(xpukernel::Context*, const int64_t*, bool*, int64_t, int64_t);

template <typename T>
int XpuLeScalar(xpukernel::Context* ctx, const T* x, bool* y, T other, int64_t len) {
  return xpukernel::le_scalar<xdnn_t<T>>(ctx, xcast(x), y, reinterpret_cast<xdnn_t<T>&>(other), len);
}

template int XpuLeScalar<float>(xpukernel::Context*, const float*, bool*, float, int64_t);
template int XpuLeScalar<float16>(xpukernel::Context*, const float16*, bool*, float16, int64_t);
template int XpuLeScalar<bfloat16>(xpukernel::Context*, const bfloat16*, bool*, bfloat16, int64_t);
template int XpuLeScalar<int32_t>(xpukernel::Context*, const int32_t*, bool*, int32_t, int64_t);
template int XpuLeScalar<int64_t>(xpukernel::Context*, const int64_t*, bool*, int64_t, int64_t);

// ── Stable sort ───────────────────────────────────────────────────────────────

template <typename T, typename TID>
int XpuStableSort(xpukernel::Context* ctx, const T* x, T* y, TID* index, int64_t m, int64_t n, bool compare) {
  return xpukernel::stable_sort<xdnn_t<T>, TID>(ctx, xcast(x), xcast_m(y), index, m, n, compare);
}

template int XpuStableSort<float, int64_t>(xpukernel::Context*, const float*, float*, int64_t*, int64_t, int64_t, bool);
template int XpuStableSort<float, int32_t>(xpukernel::Context*, const float*, float*, int32_t*, int64_t, int64_t, bool);
template int XpuStableSort<float16, int64_t>(xpukernel::Context*, const float16*, float16*, int64_t*, int64_t, int64_t,
                                             bool);
template int XpuStableSort<bfloat16, int64_t>(xpukernel::Context*, const bfloat16*, bfloat16*, int64_t*, int64_t,
                                              int64_t, bool);

// ── Unique ────────────────────────────────────────────────────────────────────

template <typename T, typename TID>
int XpuUniqueCount(xpukernel::Context* ctx, const T* x, int64_t* unique_len, int64_t x_len, T* sorted_x,
                   TID* sorted_idx, T* stage1_y, TID* stage1_idx, int64_t* count_min_max, bool is_ascend_sorted) {
  return xpukernel::unique_count<xdnn_t<T>, TID>(ctx, xcast(x), unique_len, x_len, xcast_m(sorted_x), sorted_idx,
                                                 xcast_m(stage1_y), stage1_idx, count_min_max, is_ascend_sorted);
}

template int XpuUniqueCount<float, int64_t>(xpukernel::Context*, const float*, int64_t*, int64_t, float*, int64_t*,
                                            float*, int64_t*, int64_t*, bool);
template int XpuUniqueCount<float16, int64_t>(xpukernel::Context*, const float16*, int64_t*, int64_t, float16*,
                                              int64_t*, float16*, int64_t*, int64_t*, bool);
template int XpuUniqueCount<int32_t, int64_t>(xpukernel::Context*, const int32_t*, int64_t*, int64_t, int32_t*,
                                              int64_t*, int32_t*, int64_t*, int64_t*, bool);
template int XpuUniqueCount<int64_t, int64_t>(xpukernel::Context*, const int64_t*, int64_t*, int64_t, int64_t*,
                                              int64_t*, int64_t*, int64_t*, int64_t*, bool);

// ── 1D copy ───────────────────────────────────────────────────────────────────

template <typename T>
int XpuCopy(xpukernel::Context* ctx, const T* x, T* y, int64_t len) {
  return xpukernel::copy<xdnn_t<T>>(ctx, xcast(x), xcast_m(y), len);
}

template int XpuCopy<float>(xpukernel::Context*, const float*, float*, int64_t);
template int XpuCopy<float16>(xpukernel::Context*, const float16*, float16*, int64_t);
template int XpuCopy<bfloat16>(xpukernel::Context*, const bfloat16*, bfloat16*, int64_t);
template int XpuCopy<int32_t>(xpukernel::Context*, const int32_t*, int32_t*, int64_t);
template int XpuCopy<int64_t>(xpukernel::Context*, const int64_t*, int64_t*, int64_t);

// ── Multiply / Softmax ────────────────────────────────────────────────────────

template <typename T>
int XpuMulTensor(xpukernel::Context* ctx, const T* x, const T* y, T* z, const std::vector<int64_t>& xshape,
                 bool accumulate) {
  (void)accumulate;
  return xpukernel::mul_tensor<xdnn_t<T>>(ctx, xcast(x), xcast(y), xcast_m(z), xshape, xshape);
}

template int XpuMulTensor<float>(xpukernel::Context*, const float*, const float*, float*, const std::vector<int64_t>&,
                                 bool);
template int XpuMulTensor<float16>(xpukernel::Context*, const float16*, const float16*, float16*,
                                   const std::vector<int64_t>&, bool);
template int XpuMulTensor<bfloat16>(xpukernel::Context*, const bfloat16*, const bfloat16*, bfloat16*,
                                    const std::vector<int64_t>&, bool);

template <typename T>
int XpuMulScalar(xpukernel::Context* ctx, const T* x, T* y, float other, int64_t len) {
  return xpukernel::mul_scalar<xdnn_t<T>>(ctx, xcast(x), xcast_m(y), other, len);
}

template int XpuMulScalar<float>(xpukernel::Context*, const float*, float*, float, int64_t);
template int XpuMulScalar<float16>(xpukernel::Context*, const float16*, float16*, float, int64_t);
template int XpuMulScalar<bfloat16>(xpukernel::Context*, const bfloat16*, bfloat16*, float, int64_t);

template <typename T>
int XpuSoftmax(xpukernel::Context* ctx, const T* x, T* y, const std::vector<int64_t>& xshape, int64_t axis) {
  return xpukernel::softmax<xdnn_t<T>>(ctx, xcast(x), xcast_m(y), xshape, axis);
}

template int XpuSoftmax<float>(xpukernel::Context*, const float*, float*, const std::vector<int64_t>&, int64_t);
template int XpuSoftmax<float16>(xpukernel::Context*, const float16*, float16*, const std::vector<int64_t>&, int64_t);
template int XpuSoftmax<bfloat16>(xpukernel::Context*, const bfloat16*, bfloat16*, const std::vector<int64_t>&,
                                  int64_t);

// ── Scatter ────────────────────────────────────────────────────────────────────

template <typename T, typename TID>
int XpuScatter(xpukernel::Context* ctx, const T* x, const T* updates, T* y, const TID* index, int64_t index_len,
               const std::vector<int64_t>& xshape, int64_t axis, bool is_overwrite) {
  xpukernel::VectorParam<TID> index_param = {nullptr, index_len, const_cast<TID*>(index)};
  return xpukernel::scatter<xdnn_t<T>, TID>(ctx, xcast(x), xcast(updates), xcast_m(y), index_param, xshape, axis,
                                            is_overwrite);
}

template int XpuScatter<int32_t, int64_t>(xpukernel::Context*, const int32_t*, const int32_t*, int32_t*, const int64_t*,
                                          int64_t, const std::vector<int64_t>&, int64_t, bool);
template int XpuScatter<int64_t, int64_t>(xpukernel::Context*, const int64_t*, const int64_t*, int64_t*, const int64_t*,
                                          int64_t, const std::vector<int64_t>&, int64_t, bool);

// ── XpuReplaceLastToken / XpuWriteRing (build-ahead, Phase 1) ─────────────────
// Handwritten XPU device kernels (custom_kernels.xpu + custom_kernels_launcher.cpp),
// not an xdnn/xpukernel op composition: the ReplaceLastToken semantics are a
// data-dependent placeholder guard (input_ids[dst_off] == -1?) followed by a
// req_id linear scan with early-exit -- there is no xdnn primitive for
// "conditional scan-and-scatter", and round-tripping through the host to do the
// scan would defeat the whole point of this device-resident token relay
// (build-ahead pipeline overlap avoids exactly that D2H/H2D round trip).
// batch_size / ring count are tiny in Phase 1 (concurrency=1 => batch_size==1),
// so both kernels run on a single XPU core (see custom_kernels.xpu) -- no
// cross-core synchronization to reason about.
int XpuReplaceLastToken(xpukernel::Context* ctx, int32_t* input_ids, const int64_t* cur_pairs, const int64_t* prev_ring,
                        int64_t batch_size, int64_t input_ids_len) {
  if (batch_size <= 0) {
    return 0;  // nothing to do; mirrors LaunchReplaceLastTokenKernel's early-out
  }
  return launch_replace_last_token_kernel(input_ids, cur_pairs, prev_ring, batch_size, input_ids_len,
                                          ctx->get_stream());
}

int XpuWriteRing(xpukernel::Context* ctx, int64_t* ring, const int64_t* prev_ring, const int64_t* cur_pairs,
                 const uint32_t* sampled_tokens, const int64_t* evict_ids, int64_t evict_count, int64_t batch_size,
                 int64_t ring_capacity) {
  if (batch_size <= 0) {
    return 0;  // nothing to do; mirrors LaunchWriteRingKernel's early-out
  }
  if (batch_size > ring_capacity) {
    // Defensive: never overrun the caller-allocated ring buffer. Should not happen on the
    // fast path (ring_capacity == 2*B_max, batch_size <= B_max by construction) -- guarded here
    // rather than inside the device kernel, since an in-kernel OOB write is silent device
    // memory corruption instead of a debuggable host-side error return.
    return -1;
  }
  return launch_write_ring_kernel(ring, prev_ring, cur_pairs, sampled_tokens, evict_ids, evict_count, batch_size,
                                  ring_capacity, ctx->get_stream());
}

// ── XpuMaskedFill ─────────────────────────────────────────────────────────────
template <typename T>
int XpuMaskedFill(xpukernel::Context* ctx, const T* x, const bool* mask, T* y, float value, int64_t len) {
  auto x_tensor = MakeXmlirTensor(const_cast<T*>(x), 1, len, XmlirScalarType<T>());
  auto mask_tensor = MakeXmlirTensor(const_cast<bool*>(mask), 1, len, xdnn_pytorch::ScalarType::kbool);
  auto y_tensor = MakeXmlirTensor(y, 1, len, XmlirScalarType<T>());

  auto val_scalar = xdnn_pytorch::Scalar::create<float>(value);
  std::vector<xdnn_pytorch::Tensor> workspace;
  return xdnn_pytorch::masked_fill_scalar(ctx, x_tensor, mask_tensor, val_scalar, y_tensor, workspace);
}

template int XpuMaskedFill<float>(xpukernel::Context*, const float*, const bool*, float*, float, int64_t);

// ── XpuMultinomial ────────────────────────────────────────────────────────────
template <typename T>
int XpuMultinomial(xpukernel::Context* ctx, const T* probs, int64_t* indices, int64_t m, int64_t n, int64_t num_samples,
                   int64_t seed, bool replacement) {
  auto probs_tensor = MakeXmlirTensor(const_cast<T*>(probs), m, n, XmlirScalarType<T>());
  auto out_tensor = MakeXmlirTensor(indices, m, num_samples, xdnn_pytorch::ScalarType::kint64);
  return xdnn_pytorch::multinomial(ctx, probs_tensor, num_samples, replacement, seed, out_tensor);
}

template int XpuMultinomial<float>(xpukernel::Context*, const float*, int64_t*, int64_t, int64_t, int64_t, int64_t,
                                   bool);

}  // namespace ksana_llm
