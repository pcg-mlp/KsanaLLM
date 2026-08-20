/* Copyright 2026 Tencent Inc.  All rights reserved.

 * custom_kernels_launcher.cpp — Host-side kernel launch wrappers
 *
 * 这个文件用 xpu-clang 的 "-x xpu --xpu-host-only -fxpu-launch-return" 模式编译，
 * 因为它包含 <<<>>> launch 语法。普通 g++ 无法处理这个语法。
 *
 * 编译 (由 CMake 自动调用):
 *   ${XTDK_DIR}/bin/clang -std=c++17 -O2 -x xpu --xpu-host-only -fxpu-launch-return
 *     --target=x86_64-baidu-linux-gnu -fPIC
 *     -I${XMLIR_DIR}/xre/include -I${PROJECT_SOURCE_DIR}/csrc
 *     -c custom_kernels_launcher.cpp -o custom_kernels_launcher.o
==============================================================================*/

#include <xpu/runtime.h>

#include <cstdint>

// 前向声明：设备端 kernel（定义在 custom_kernels.xpu 里）
__attribute__((global)) void vector_add_kernel(const float* a, const float* b, float* c, int n);
__attribute__((global)) void replace_last_token_kernel(int32_t* input_ids, const int64_t* cur_pairs,
                                                       const int64_t* prev_ring, int64_t batch_size,
                                                       int64_t input_ids_len);
__attribute__((global)) void write_ring_kernel(int64_t* ring, const int64_t* prev_ring, const int64_t* cur_pairs,
                                               const uint32_t* sampled_tokens, const int64_t* evict_ids,
                                               int64_t evict_count, int64_t batch_size, int64_t ring_capacity);

// ═══════════════════════════════════════════════════════════════
// Host launcher: 被 kernel_wrapper.cpp 里的 XpuAdd<float> 调用
// ═══════════════════════════════════════════════════════════════
extern "C" int launch_vector_add_kernel(const float* a, const float* b, float* c, int n, void* stream) {
  int cluster_num = 4;
  int core_num = 8;
  return vector_add_kernel<<<cluster_num, core_num, (XPUStream)stream>>>(a, b, c, n);
}

// ═══════════════════════════════════════════════════════════════
// Host launchers: 被 basic_kernel_wrapper.cpp 里的 XpuReplaceLastToken /
// XpuWriteRing 调用 (build-ahead Phase 1, 见 basic_kernel_wrapper.h)。
// batch_size 极小 (concurrency=1 => batch_size==1)，单 cluster/单 core 即够。
// ═══════════════════════════════════════════════════════════════
extern "C" int launch_replace_last_token_kernel(int32_t* input_ids, const int64_t* cur_pairs, const int64_t* prev_ring,
                                                int64_t batch_size, int64_t input_ids_len, void* stream) {
  return replace_last_token_kernel<<<1, 1, (XPUStream)stream>>>(input_ids, cur_pairs, prev_ring, batch_size,
                                                                input_ids_len);
}

extern "C" int launch_write_ring_kernel(int64_t* ring, const int64_t* prev_ring, const int64_t* cur_pairs,
                                        const uint32_t* sampled_tokens, const int64_t* evict_ids, int64_t evict_count,
                                        int64_t batch_size, int64_t ring_capacity, void* stream) {
  return write_ring_kernel<<<1, 1, (XPUStream)stream>>>(ring, prev_ring, cur_pairs, sampled_tokens, evict_ids,
                                                        evict_count, batch_size, ring_capacity);
}
