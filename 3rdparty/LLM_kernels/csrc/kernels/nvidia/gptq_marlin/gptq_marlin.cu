/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/vllm-project/vllm
 */

#include <sstream>
#include "csrc/kernels/nvidia/gptq_marlin/gptq_marlin.h"
#include "csrc/kernels/nvidia/gptq_marlin/gptq_marlin_kernel.cuh"
#include "csrc/kernels/nvidia/gptq_marlin/marlin.cuh"
#include "csrc/kernels/nvidia/gptq_marlin/marlin_dtypes.cuh"
#include "csrc/kernels/nvidia/gptq_marlin/marlin_params.h"
#include "csrc/kernels/nvidia/gptq_marlin/scalar_type.hpp"

namespace llm_kernels {
namespace nvidia {
namespace marlin {

#define __CALL_IF(W_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, HAS_ACT_ORDER, HAS_ZP, GROUP_BLOCKS,      \
                  NUM_THREADS)                                                                                         \
  else if (q_type == W_TYPE && thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS &&             \
           thread_k_blocks == THREAD_K_BLOCKS && has_act_order == HAS_ACT_ORDER && has_zp == HAS_ZP &&                 \
           group_blocks == GROUP_BLOCKS && num_threads == NUM_THREADS) {                                               \
    cudaFuncSetAttribute(Marlin<scalar_t, W_TYPE.id(), NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, \
                                pipe_stages, HAS_ACT_ORDER, HAS_ZP, GROUP_BLOCKS>,                                     \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);                                 \
    Marlin<scalar_t, W_TYPE.id(), NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, pipe_stages,         \
           HAS_ACT_ORDER, HAS_ZP, GROUP_BLOCKS>                                                                        \
        <<<blocks, NUM_THREADS, max_shared_mem, stream>>>(A_ptr, B_ptr, C_ptr, C_tmp_ptr, s_ptr, zp_ptr, g_idx_ptr,    \
                                                          num_groups, prob_m, prob_n, prob_k, locks, use_fp32_reduce); \
  }

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

typedef struct {
  int max_m_blocks;
  thread_config_t tb_cfg;
} exec_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128},
};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128},

};

int get_scales_cache_size(thread_config_t const& th_config, int prob_m, int prob_n, int prob_k, int num_bits,
                          int group_size, bool has_act_order, bool is_k_full) {
  bool cache_scales_chunk = has_act_order && !is_k_full;

  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);  // Worst case is 32 group size
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups = tb_groups * pipe_stages * 2;  // Chunk size is 2x pipeline over dim K
    load_groups = max(load_groups, 32);             // We load at least 32 scale groups
    return load_groups * tb_n * 2;

  } else {
    int tb_scales = tb_groups * tb_n * 2;

    return tb_scales * pipe_stages;
  }
}

bool is_valid_cache_size(thread_config_t const& th_config, int max_m_blocks, int prob_m, int prob_n, int prob_k,
                         int num_bits, int scales_cache_size, int max_shared_mem) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;

  int b_size = (tb_k * tb_n / pack_factor) * 4;

  // Get A size
  int m_blocks = div_ceil(prob_m, 16);
  int tb_max_m = 16;

  while (true) {
    if (m_blocks >= max_m_blocks) {
      tb_max_m *= max_m_blocks;
      break;
    }

    max_m_blocks--;
    if (max_m_blocks == 0) {
      if (true) {
        std::ostringstream oss;
        oss << "Unexpected m_blocks = " << m_blocks;
        throw std::runtime_error(oss.str());
      }
    }
  }

  int a_size = (tb_max_m * tb_k) * 2;

  float pipe_size = (a_size + b_size) * pipe_stages;

  if (max_shared_mem / 2 <= scales_cache_size) {
    throw std::runtime_error("Sanity check failed: max_shared_mem / 2 <= scales_cache_size");
  }

  return pipe_size < 0.95f * (max_shared_mem - scales_cache_size);
}

bool is_valid_config(thread_config_t const& th_config, int max_m_blocks, int prob_m, int prob_n, int prob_k,
                     int num_bits, int group_size, bool has_act_order, bool is_k_full, int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 || th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  //  Determine cache for scales
  int scales_cache_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits, group_size, has_act_order, is_k_full);

  // Check that pipeline fits into cache
  if (!is_valid_cache_size(th_config, max_m_blocks, prob_m, prob_n, prob_k, num_bits, scales_cache_size,
                           max_shared_mem)) {
    return false;
  }

  return true;
}

int determine_reduce_max_m(int prob_m, int max_par) {
  constexpr int tile_m_size = 16;

  if (prob_m <= tile_m_size) {
    return tile_m_size;

  } else if (prob_m <= tile_m_size * 2) {
    return tile_m_size * 2;

  } else if (prob_m <= tile_m_size * 3) {
    return tile_m_size * 3;

  } else if (prob_m <= tile_m_size * 4) {
    return tile_m_size * 4;

  } else {
    int cur_par = min(div_ceil(prob_m, tile_m_size * 4), max_par);
    return tile_m_size * 4 * cur_par;
  }
}

exec_config_t determine_thread_config(int prob_m, int prob_n, int prob_k, int num_bits, int group_size,
                                      bool has_act_order, bool is_k_full, int max_shared_mem) {
  int max_m_blocks = 4;
  while (max_m_blocks > 0) {
    if (prob_m <= 16) {
      for (auto th_config : small_batch_thread_configs) {
        if (is_valid_config(th_config, max_m_blocks, prob_m, prob_n, prob_k, num_bits, group_size, has_act_order,
                            is_k_full, max_shared_mem)) {
          return exec_config_t{max_m_blocks, th_config};
        }
      }
    } else {
      for (auto th_config : large_batch_thread_configs) {
        if (is_valid_config(th_config, max_m_blocks, prob_m, prob_n, prob_k, num_bits, group_size, has_act_order,
                            is_k_full, max_shared_mem)) {
          return exec_config_t{max_m_blocks, th_config};
        }
      }
    }

    max_m_blocks--;  // Process less M blocks per invocation to reduce cache
                     // usage
  }

  return exec_config_t{0, {-1, -1, -1}};
}

#define GPTQ_CALL_IF(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)             \
  __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
  __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
  __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
  __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
                                                                          \
  __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
  __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
  __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
  __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)  \
                                                                          \
  __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
  __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
  __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
  __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)  \
                                                                          \
  __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
  __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
  __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
  __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)  \
                                                                          \
  __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
  __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
  __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
  __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)

#define AWQ_CALL_IF(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)             \
  __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
  __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
  __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
  __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
                                                                         \
  __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
  __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
  __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
  __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
                                                                         \
  __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
  __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
  __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
  __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
                                                                         \
  __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
  __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
  __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
  __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)

template <typename scalar_t>
void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* s, void* zp, void* g_idx, void* perm,
               void* a_tmp, int prob_m, int prob_n, int prob_k, void* workspace, scalar_type::ScalarType const& q_type,
               bool has_act_order, bool is_k_full, bool has_zp, int num_groups, int group_size, int dev,
               cudaStream_t stream, int thread_k, int thread_n, int sms, int max_par, bool use_fp32_reduce) {
  if (prob_m <= 0 || prob_n <= 0 || prob_k <= 0) {
    std::ostringstream oss;
    oss << "Invalid MNK = [" << prob_m << ", " << prob_n << ", " << prob_k << "]";
    throw std::runtime_error(oss.str());
  }

  // TODO: remove alias when we start supporting other 8bit types
  int num_bits = q_type.size_bits();
  int tot_m = prob_m;
  int tot_m_blocks = div_ceil(tot_m, 16);
  int pad = 16 * tot_m_blocks - tot_m;

  if (sms == -1) {
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  }

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (max_shared_mem <= 0) {
    throw std::runtime_error("Sanity check failed: max_shared_mem must be greater than 0");
  }

  // Set thread config
  exec_config_t exec_cfg;
  if (thread_k != -1 && thread_n != -1) {
    // User-defined config
    exec_cfg = exec_config_t{4, thread_config_t{thread_k, thread_n, default_threads}};
  } else {
    // Auto config
    exec_cfg =
        determine_thread_config(prob_m, prob_n, prob_k, num_bits, group_size, has_act_order, is_k_full, max_shared_mem);
  }

  if (exec_cfg.max_m_blocks <= 0 || !is_valid_config(exec_cfg.tb_cfg, exec_cfg.max_m_blocks, prob_m, prob_n, prob_k,
                                                     num_bits, group_size, has_act_order, is_k_full, max_shared_mem)) {
    std::ostringstream oss;
    oss << "Invalid thread config: max_m_blocks = " << exec_cfg.max_m_blocks
        << ", thread_k = " << exec_cfg.tb_cfg.thread_k << ", thread_n = " << exec_cfg.tb_cfg.thread_n
        << ", num_threads = " << exec_cfg.tb_cfg.num_threads << " for MKN = [" << prob_m << ", " << prob_k << ", "
        << prob_n << "]"
        << " and num_bits = " << num_bits << ", group_size = " << group_size << ", has_act_order = " << has_act_order
        << ", is_k_full = " << is_k_full << ", max_shared_mem = " << max_shared_mem;
    throw std::runtime_error(oss.str());
  }

  int num_threads = exec_cfg.tb_cfg.num_threads;
  thread_k = exec_cfg.tb_cfg.thread_k;
  thread_n = exec_cfg.tb_cfg.thread_n;

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;

  int blocks = sms;

  if (prob_n % thread_n != 0) {
    std::ostringstream oss;
    oss << "prob_n = " << prob_n << " is not divisible by thread_n = " << thread_n;
    throw std::runtime_error(oss.str());
  }
  if (prob_k % thread_k != 0) {
    std::ostringstream oss;
    oss << "prob_k = " << prob_k << " is not divisible by thread_k = " << thread_k;
    throw std::runtime_error(oss.str());
  }

  int group_blocks = 0;
  if (has_act_order) {
    if (is_k_full) {
      group_blocks = group_size / 16;
    } else {
      group_blocks = 0;
    }

  } else {
    if (group_size == -1) {
      group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      if (prob_k % group_blocks != 0) {
        std::ostringstream oss;
        oss << "prob_k = " << prob_k << " is not divisible by group_blocks = " << group_blocks;
        throw std::runtime_error(oss.str());
      }
    }
  }

  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;
  const int4* s_ptr = (const int4*)s;
  const int4* zp_ptr = (const int4*)zp;
  const int* g_idx_ptr = (const int*)g_idx;
  const int* perm_ptr = (const int*)perm;
  int4* a_tmp_ptr = (int4*)a_tmp;

  int* locks = (int*)workspace;

  if (has_act_order) {
    // Permute A columns
    int block_rows = div_ceil(prob_m, blocks);
    permute_cols_kernel<<<blocks, default_threads, 0, stream>>>(A_ptr, perm_ptr, a_tmp_ptr, prob_m, prob_k, block_rows);
    A_ptr = a_tmp_ptr;
  }

  // If we have a full K, then we can run the non-act-order version of Marlin
  // (since the weight rows are reordered by increasing group ids, and by having
  // a full K, we have full original groups)
  if (is_k_full) {
    has_act_order = false;
  }

  // Main loop
  for (int i = 0; i < tot_m_blocks; i += exec_cfg.max_m_blocks) {
    int thread_m_blocks = tot_m_blocks - i;
    prob_m = tot_m - 16 * i;
    int par = 1;
    if (thread_m_blocks > exec_cfg.max_m_blocks) {
      // Note that parallel > 1 currently only works for inputs without any
      // padding
      par = (16 * thread_m_blocks - pad) / (16 * exec_cfg.max_m_blocks);
      if (par > max_par) par = max_par;
      prob_m = (16 * exec_cfg.max_m_blocks) * par;
      i += exec_cfg.max_m_blocks * (par - 1);
      thread_m_blocks = exec_cfg.max_m_blocks;
    }

    if (false) {
    }
    GPTQ_CALL_IF(scalar_type::kU4B8, 16, 4, 256)
    GPTQ_CALL_IF(scalar_type::kU4B8, 8, 8, 256)
    GPTQ_CALL_IF(scalar_type::kU4B8, 8, 4, 128)
    GPTQ_CALL_IF(scalar_type::kU4B8, 4, 8, 128)
    // GPTQ_CALL_IF(scalar_type::kU8B128, 16, 4, 256)
    // GPTQ_CALL_IF(scalar_type::kU8B128, 8, 8, 256)
    // GPTQ_CALL_IF(scalar_type::kU8B128, 8, 4, 128)
    // GPTQ_CALL_IF(scalar_type::kU8B128, 4, 8, 128)

    AWQ_CALL_IF(scalar_type::kU4, 16, 4, 256)
    AWQ_CALL_IF(scalar_type::kU4, 8, 8, 256)
    AWQ_CALL_IF(scalar_type::kU4, 8, 4, 128)
    AWQ_CALL_IF(scalar_type::kU4, 4, 8, 128)
    // AWQ_CALL_IF(scalar_type::kU8, 16, 4, 256)
    // AWQ_CALL_IF(scalar_type::kU8, 8, 8, 256)
    // AWQ_CALL_IF(scalar_type::kU8, 8, 4, 128)
    // AWQ_CALL_IF(scalar_type::kU8, 4, 8, 128)
    else {
      if (true) {
        std::ostringstream oss;
        oss << "Unsupported shapes: MNK = [" << prob_m << ", " << prob_n << ", " << prob_k << "], "
            << "has_act_order = " << has_act_order << ", "
            << "num_groups = " << num_groups << ", "
            << "group_size = " << group_size << ", "
            << "thread_m_blocks = " << thread_m_blocks << ", "
            << "thread_n_blocks = " << thread_n_blocks << ", "
            << "thread_k_blocks = " << thread_k_blocks << ", "
            << "num_bits = " << num_bits;
        throw std::runtime_error(oss.str());
      }
    }

    A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par;
    C_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
  }
}

}  // namespace marlin

void gptq_marlin_gemm(void* a, void* a_tmp, void* b_q_weight, void* b_scales, void* b_zeros, void* g_idx, void* perm,
                      void* workspace, void* c, void* c_tmp, int64_t size_m, int64_t size_n, int64_t size_k,
                      int64_t num_groups, bool is_k_full, bool has_zp, bool has_act_order, bool is_awq, int rank,
                      cudaStream_t stream) {
  marlin::scalar_type::ScalarType b_q_type = is_awq ? marlin::scalar_type::kU4 : marlin::scalar_type::kU4B8;

  int group_size = -1;
  if (has_act_order) {
    if (is_k_full) {
      group_size = size_k / num_groups;
    } else {
      group_size = 0;
    }
  } else {
    if (num_groups > 1) {
      group_size = size_k / num_groups;
    } else {
      group_size = -1;
    }
  }

  marlin::marlin_mm<half>(reinterpret_cast<half*>(a), reinterpret_cast<void*>(b_q_weight), reinterpret_cast<half*>(c),
                          reinterpret_cast<float*>(c_tmp), reinterpret_cast<half*>(b_scales),
                          reinterpret_cast<void*>(b_zeros), reinterpret_cast<void*>(g_idx),
                          reinterpret_cast<void*>(perm), reinterpret_cast<half*>(a_tmp), size_m, size_n, size_k,
                          reinterpret_cast<void*>(workspace), b_q_type, has_act_order, is_k_full, has_zp, num_groups,
                          group_size, rank, stream, -1, -1, -1, marlin::max_par, true);
}

}  // namespace nvidia
}  // namespace llm_kernels