/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * HummingJitCompiler: Three-step decoupled JIT for humming MoE GEMM kernels.
 *
 * Pipeline: Generate (Python/Jinja2) -> Cache (nvcc -> cubin) -> Use (CUDA Driver API).
 * Kernel metadata is read from cubin ELF sections (extern "C" __constant__ uint32_t).
 * Multi-process safe with flock.
 */

#pragma once

#include <cuda.h>

#include <climits>
#include <cstdint>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "csrc/kernels/nvidia/others/humming/launcher/kernel_meta.h"

namespace llm_kernels {
namespace nvidia {
namespace humming {

/// JIT compiler that generates, compiles, and loads humming kernels.
class HummingJitCompiler {
 public:
  /// Configuration for a single kernel variant to compile.
  struct KernelConfig {
    int64_t size_n;
    int64_t size_k;
    int64_t num_experts;
    int block_m;
    GemmType gemm_type;
    ComputeDtype compute_dtype;
    // Multi-config range: this kernel is optimal for size_m in [min_shape_m, max_shape_m]
    int64_t min_shape_m = 1;
    int64_t max_shape_m = INT64_MAX;

    /// Returns a unique string key for cache directory naming.
    std::string CacheKey() const;
  };

  /// @brief Constructs the JIT compiler with self-resolving paths.
  ///
  /// Path resolution order:
  ///   base_dir: $KSANA_HUMMING_BASE_DIR, or derived from __FILE__
  ///   cache_dir: $KSANA_HUMMING_KERNEL_PATH, or "~/.cache/KsanaLLM/humming_kernel_files"
  ///   include_dir: $HUMMING_INCLUDE_DIR, or auto-detect via python3, or base_dir + "/include"
  ///   script_dir: base_dir + "/jit"
  HummingJitCompiler();
  ~HummingJitCompiler();

  // Non-copyable, non-movable (owns CUmodule handles).
  HummingJitCompiler(const HummingJitCompiler&) = delete;
  HummingJitCompiler& operator=(const HummingJitCompiler&) = delete;

  /// Compile and load all kernel variants for a given problem shape.
  /// Internally queries H20 heuristic, generates/compiles/loads each variant.
  /// Thread-safe and multi-process safe. No-op if this shape was already compiled.
  void CompileKernels(int64_t size_n, int64_t size_k, int64_t num_experts, ComputeDtype compute_dtype,
                      GemmType gemm_type);

  /// Compile and load a set of kernel variants. Thread-safe and multi-process safe.
  /// Skips already-cached cubins unless HUMMING_REGENERATE=1 is set.
  void CompileKernels(const std::vector<KernelConfig>& configs);

  /// Select the best kernel for a given problem size. Returns nullptr if none found.
  /// Selection criteria: match (size_n, size_k, gemm_type, compute_dtype), minimize
  /// wasted tiles for size_m.
  const KernelMeta* SelectKernel(int64_t size_m, int64_t size_n, int64_t size_k, GemmType gemm_type,
                                 ComputeDtype compute_dtype) const;

  /// Get all loaded kernels (for debugging/testing).
  const std::vector<KernelMeta>& GetKernelTable() const { return kernel_table_; }

 private:
  /// Query humming H20 heuristic for optimal kernel configs for a given problem.
  /// Falls back to block_m={64,128} if the Python query fails.
  std::vector<KernelConfig> QueryH20Configs(int64_t size_n, int64_t size_k, int64_t num_experts,
                                            ComputeDtype compute_dtype, GemmType gemm_type);

  /// Step 1: Call humming_tool.py generate to produce .cu source.
  void GenerateSource(const KernelConfig& config, const std::string& output_path);

  /// Step 2: Invoke nvcc to compile .cu -> .cubin.
  void CompileCubin(const std::string& source_path, const std::string& cubin_path, const std::string& include_dir);

  /// Step 3: Load cubin via CUDA Driver API, read ELF metadata, register kernel.
  void LoadAndRegister(const KernelConfig& config, const std::string& cubin_path);

  /// Check if a cached cubin already exists and is valid.
  bool CacheExists(const KernelConfig& config) const;

  /// Get the cache directory path for a specific config.
  std::string GetCacheDir(const KernelConfig& config) const;

  /// Multi-process lock/unlock on a directory.
  int AcquireLock(const std::string& dir_path);
  void ReleaseLock(int fd);

  // ── ELF/Cubin reader utilities ──────────────────────────────────────────

  /// Read a uint32_t value from an extern "C" __constant__ symbol in a cubin ELF.
  static uint32_t ReadCubinUint32(const std::string& cubin_path, const std::string& symbol_name);

  /// Read a boolean (uint32 > 0) from an extern "C" __constant__ symbol.
  static bool ReadCubinBool(const std::string& cubin_path, const std::string& symbol_name);

  /// Find the mangled kernel function symbol in the ELF (matches STT_FUNC with base_name).
  static std::string FindKernelSymbol(const std::string& cubin_path, const std::string& base_name);

  // ── Member state ────────────────────────────────────────────────────────

  std::string cache_dir_;
  std::string include_dir_;
  std::string humming_tool_path_;
  std::string nvcc_path_;
  bool force_regenerate_;

  std::vector<KernelMeta> kernel_table_;
  std::vector<CUmodule> modules_;
  mutable std::mutex mutex_;
  mutable std::mutex log_mutex_;  // serializes std::cout output from parallel threads
  bool verbose_ = false;

  // Set of cache keys already loaded into kernel_table_.  Prevents duplicate
  // cuModuleLoad calls when CompileKernels is invoked more than once with
  // overlapping configs (e.g. per-layer MoE init).
  std::set<std::string> loaded_keys_;

  // Set of (size_n, size_k, num_experts, compute_dtype, gemm_type) tuples
  // for shapes that have already been compiled.  Prevents redundant H20 queries
  // and compilation when CompileKernels(shape) is called from multiple layers.
  std::set<std::tuple<int64_t, int64_t, int64_t, int, int>> compiled_shapes_;

  // Intra-process mutexes keyed by cache directory.  flock() is process-level
  // and does NOT block sibling threads; these mutexes provide thread-level
  // mutual exclusion so that at most one thread per cache-dir runs the
  // generate→compile→rename critical section.
  static std::mutex dir_mutexes_mutex_;
  static std::unordered_map<std::string, std::mutex*> dir_mutexes_;
  std::mutex* GetDirMutex(const std::string& dir);
};

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
