/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * HummingJitCompiler implementation.
 *
 * Three-step decoupled JIT: Generate (Python/Jinja2) -> Cache (nvcc -> cubin) -> Use.
 * Reads kernel metadata from cubin ELF sections.
 * Multi-process safe with flock. Supports HUMMING_REGENERATE env var.
 */

#include "csrc/kernels/nvidia/others/humming/jit/humming_jit_compiler.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "csrc/utils/nvidia/assert.h"

namespace llm_kernels {
namespace nvidia {
namespace humming {

// ── Static dir-mutex map for intra-process mutual exclusion ─────────────────
// flock() is process-level and does NOT block sibling threads within the same
// process.  These per-directory mutexes provide thread-level exclusion so that
// only one thread at a time enters the generate→compile→rename critical
// section for a given cache directory.

std::mutex HummingJitCompiler::dir_mutexes_mutex_;
std::unordered_map<std::string, std::mutex*> HummingJitCompiler::dir_mutexes_;

std::mutex* HummingJitCompiler::GetDirMutex(const std::string& dir) {
  std::lock_guard<std::mutex> lg(dir_mutexes_mutex_);
  auto& ptr = dir_mutexes_[dir];
  if (!ptr) ptr = new std::mutex();
  return ptr;
}

// ════════════════════════════════════════════════════════════════════════════════
// ELF64 structures for cubin reading (self-contained, no external elf.h dependency)
// ════════════════════════════════════════════════════════════════════════════════

namespace {

constexpr uint32_t kShtSymtab = 2;  // SHT_SYMTAB
constexpr uint32_t kShtNobits = 8;  // SHT_NOBITS
constexpr uint8_t kSttFunc = 2;     // STT_FUNC

#pragma pack(push, 1)
struct Elf64Ehdr {
  unsigned char e_ident[16];
  uint16_t e_type;
  uint16_t e_machine;
  uint32_t e_version;
  uint64_t e_entry;
  uint64_t e_phoff;
  uint64_t e_shoff;
  uint32_t e_flags;
  uint16_t e_ehsize;
  uint16_t e_phentsize;
  uint16_t e_phnum;
  uint16_t e_shentsize;
  uint16_t e_shnum;
  uint16_t e_shstrndx;
};

struct Elf64Shdr {
  uint32_t sh_name;
  uint32_t sh_type;
  uint64_t sh_flags;
  uint64_t sh_addr;
  uint64_t sh_offset;
  uint64_t sh_size;
  uint32_t sh_link;
  uint32_t sh_info;
  uint64_t sh_addralign;
  uint64_t sh_entsize;
};

struct Elf64Sym {
  uint32_t st_name;
  uint8_t st_info;
  uint8_t st_other;
  uint16_t st_shndx;
  uint64_t st_value;
  uint64_t st_size;
};
#pragma pack(pop)

/// Lightweight ELF reader for cubin files. Reads symbol table to find:
/// 1. __constant__ uint32_t metadata values (by name -> file offset -> read value)
/// 2. Kernel function symbols (STT_FUNC type matching a name pattern)
class CubinElfReader {
 public:
  struct SymbolInfo {
    std::string name;
    uint64_t file_offset;  // 0 for SHT_NOBITS sections
    uint8_t st_info;
    uint16_t st_shndx;
  };

  explicit CubinElfReader(const std::string& path) : path_(path) {
    std::ifstream fs(path, std::ios::binary);
    KLLM_KERNEL_CHECK_WITH_INFO(fs.good(), "Failed to open cubin file: %s", path.c_str());

    // Read ELF header
    Elf64Ehdr ehdr;
    fs.read(reinterpret_cast<char*>(&ehdr), sizeof(ehdr));
    KLLM_KERNEL_CHECK_WITH_INFO(fs.good(), "Failed to read ELF header from: %s", path.c_str());

    // Validate ELF magic
    KLLM_KERNEL_CHECK_WITH_INFO(
        ehdr.e_ident[0] == 0x7f && ehdr.e_ident[1] == 'E' && ehdr.e_ident[2] == 'L' && ehdr.e_ident[3] == 'F',
        "Not a valid ELF file: %s", path.c_str());

    // Read section headers
    std::vector<Elf64Shdr> shdrs(ehdr.e_shnum);
    fs.seekg(static_cast<std::streamoff>(ehdr.e_shoff));
    fs.read(reinterpret_cast<char*>(shdrs.data()), static_cast<std::streamsize>(sizeof(Elf64Shdr) * ehdr.e_shnum));
    KLLM_KERNEL_CHECK_WITH_INFO(fs.good(), "Failed to read section headers from: %s", path.c_str());

    // Find and read all symbol tables
    for (uint16_t i = 0; i < ehdr.e_shnum; ++i) {
      if (shdrs[i].sh_type == kShtSymtab) {
        ReadSymbolTable(fs, shdrs, i);
      }
    }
  }

  /// Read a uint32_t value from a named symbol.
  uint32_t GetUint32(const std::string& name) const {
    auto it = symbol_offsets_.find(name);
    KLLM_KERNEL_CHECK_WITH_INFO(it != symbol_offsets_.end(), "Symbol '%s' not found in cubin: %s", name.c_str(),
                                path_.c_str());

    // SHT_NOBITS section: value is 0
    if (it->second == 0ULL) return 0;

    std::ifstream fs(path_, std::ios::binary);
    KLLM_KERNEL_CHECK_WITH_INFO(fs.good(), "Failed to reopen cubin: %s", path_.c_str());

    uint32_t val = 0;
    fs.seekg(static_cast<std::streamoff>(it->second));
    fs.read(reinterpret_cast<char*>(&val), sizeof(uint32_t));
    return val;
  }

  /// Read a boolean from a named symbol (uint32 > 0 → true).
  bool GetBool(const std::string& name) const { return GetUint32(name) > 0; }

  /// Find the first function symbol (STT_FUNC) whose name contains base_name.
  std::string FindFuncSymbol(const std::string& base_name) const {
    for (const auto& sym : symbols_) {
      // Check if it's a function symbol (lower 4 bits of st_info == STT_FUNC)
      uint8_t sym_type = sym.st_info & 0x0f;
      if (sym_type != kSttFunc) continue;

      // Check if the name contains the base name
      if (sym.name.find(base_name) != std::string::npos) {
        return sym.name;
      }
    }
    return "";
  }

 private:
  void ReadSymbolTable(std::ifstream& fs, const std::vector<Elf64Shdr>& shdrs, uint16_t sym_idx) {
    const Elf64Shdr& symtab = shdrs[sym_idx];
    const Elf64Shdr& strtab = shdrs[symtab.sh_link];

    // Read string table
    std::vector<char> names(strtab.sh_size);
    fs.seekg(static_cast<std::streamoff>(strtab.sh_offset));
    fs.read(names.data(), static_cast<std::streamsize>(strtab.sh_size));

    // Read symbol entries
    size_t count = symtab.sh_size / symtab.sh_entsize;
    std::vector<Elf64Sym> syms(count);
    fs.seekg(static_cast<std::streamoff>(symtab.sh_offset));
    fs.read(reinterpret_cast<char*>(syms.data()), static_cast<std::streamsize>(symtab.sh_size));

    for (const auto& sym : syms) {
      if (sym.st_name == 0 || sym.st_name >= names.size()) continue;
      if (sym.st_shndx >= shdrs.size()) continue;

      std::string sym_name = &names[sym.st_name];
      const auto& shdr = shdrs[sym.st_shndx];

      uint64_t abs_offset = 0;
      if (shdr.sh_type == kShtNobits) {
        abs_offset = 0;
      } else {
        abs_offset = shdr.sh_offset + sym.st_value;
      }

      symbol_offsets_[sym_name] = abs_offset;
      symbols_.push_back({sym_name, abs_offset, sym.st_info, sym.st_shndx});
    }
  }

  std::string path_;
  std::map<std::string, uint64_t> symbol_offsets_;
  std::vector<SymbolInfo> symbols_;
};

/// Execute a shell command and return its exit code. Captures stdout+stderr.
int ExecuteCommand(const std::string& cmd, std::string* output = nullptr) {
  std::string full_cmd = cmd + " 2>&1";
  FILE* pipe = popen(full_cmd.c_str(), "r");
  if (!pipe) return -1;

  std::string result;
  std::array<char, 256> buffer;
  while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    result += buffer.data();
  }

  int ret = pclose(pipe);
  if (output) *output = result;
  return WEXITSTATUS(ret);
}

/// Create directory recursively (like mkdir -p).
void MkdirP(const std::string& path) {
  std::string cmd = "mkdir -p \"" + path + "\"";
  int ret = ExecuteCommand(cmd);
  KLLM_KERNEL_CHECK_WITH_INFO(ret == 0, "Failed to create directory: %s", path.c_str());
}

/// Check if a file exists.
bool FileExists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

/// Find nvcc in PATH or CUDA_HOME.
std::string FindNvcc() {
  // Try CUDA_HOME first
  const char* cuda_home = std::getenv("CUDA_HOME");
  if (cuda_home) {
    std::string nvcc_path = std::string(cuda_home) + "/bin/nvcc";
    if (FileExists(nvcc_path)) return nvcc_path;
  }

  // Try /usr/local/cuda
  if (FileExists("/usr/local/cuda/bin/nvcc")) {
    return "/usr/local/cuda/bin/nvcc";
  }

  // Fallback: hope it's in PATH
  return "nvcc";
}

/// Derive default humming_tool.py path from compile-time source file location.
/// __FILE__ -> .../humming/jit/humming_jit_compiler.cpp -> .../humming/jit/humming_tool.py
std::string GetDefaultHummingToolPath() {
  std::string file_path(__FILE__);
  size_t pos = file_path.rfind("/jit/");
  if (pos != std::string::npos) {
    return file_path.substr(0, pos) + "/jit/humming_tool.py";
  }
  return "";
}

/// Derive the humming operator base directory from compile-time source file location.
/// __FILE__ -> .../humming/jit/humming_jit_compiler.cpp -> .../humming
/// This is the parent of the local include/ directory containing the .cuh headers.
std::string GetDefaultBaseDir() {
  std::string file_path(__FILE__);
  size_t pos = file_path.rfind("/jit/");
  if (pos != std::string::npos) {
    return file_path.substr(0, pos);
  }
  return "";
}

/// Resolve the local humming include directory.
/// 1. $HUMMING_INCLUDE_DIR env var (explicit override)
/// 2. base_dir + "/include" where base_dir is derived from __FILE__
/// 3. Falls back to empty string (caller should handle)
///
/// Note: $KSANA_HUMMING_BASE_DIR is intentionally NOT used here. In the
/// constructor it has different semantics (points at the jit/ subdirectory
/// for humming_tool.py), so reusing it for include resolution would yield
/// a wrong path. Use $HUMMING_INCLUDE_DIR to override the include dir directly.
std::string ResolveLocalIncludeDir() {
  const char* env = std::getenv("HUMMING_INCLUDE_DIR");
  if (env && env[0] != '\0') return env;

  std::string base_dir = GetDefaultBaseDir();
  if (base_dir.empty()) return "";
  return base_dir + "/include";
}

}  // anonymous namespace

// ════════════════════════════════════════════════════════════════════════════════
// KernelConfig
// ════════════════════════════════════════════════════════════════════════════════

std::string HummingJitCompiler::KernelConfig::CacheKey() const {
  std::ostringstream oss;
  oss << "n" << size_n << "_k" << size_k << "_e" << num_experts << "_bm" << block_m << "_gt"
      << static_cast<int>(gemm_type) << "_dt" << static_cast<int>(compute_dtype);
  if (force_stream_k) oss << "_fsk1";
  if (force_block_n > 0) oss << "_fbn" << force_block_n;
  if (force_stages > 0) oss << "_fns" << force_stages;
  return oss.str();
}

// ════════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════════

HummingJitCompiler::HummingJitCompiler() : nvcc_path_(FindNvcc()), force_regenerate_(false) {
  // ── 1. Resolve humming_tool.py path: KSANA_HUMMING_BASE_DIR env, or derive from __FILE__ ──
  const char* base_env = std::getenv("KSANA_HUMMING_BASE_DIR");
  if (base_env && base_env[0] != '\0') {
    humming_tool_path_ = std::string(base_env) + "/humming_tool.py";
  } else {
    humming_tool_path_ = GetDefaultHummingToolPath();
    KLLM_KERNEL_CHECK_WITH_INFO(!humming_tool_path_.empty(),
                                "Cannot resolve humming_tool.py path from __FILE__ (%s). "
                                "Set $KSANA_HUMMING_BASE_DIR environment variable.",
                                __FILE__);
  }

  // ── 2. Resolve include dir: local copy of humming headers ──
  include_dir_ = ResolveLocalIncludeDir();
  KLLM_KERNEL_CHECK_WITH_INFO(!include_dir_.empty(),
                              "Cannot resolve local humming include directory. "
                              "Set $HUMMING_INCLUDE_DIR environment variable. "
                              "Expected .cuh headers under <base_dir>/include/humming/.");
  // Sentinel check: verify the local headers are actually present.
  KLLM_KERNEL_CHECK_WITH_INFO(FileExists(include_dir_ + "/humming/kernel/humming.cuh"),
                              "Local humming include directory '%s' does not contain "
                              "humming/kernel/humming.cuh. Expected the 0.1.5 .cuh headers under "
                              "<base_dir>/include/humming/. Set $HUMMING_INCLUDE_DIR to override.",
                              include_dir_.c_str());

  // ── 3. Cache dir: KSANA_HUMMING_KERNEL_PATH env, or ~/.cache/KsanaLLM/humming_kernel_files ──
  const char* cache_env = std::getenv("KSANA_HUMMING_KERNEL_PATH");
  if (cache_env && cache_env[0] != '\0') {
    cache_dir_ = cache_env;
  } else {
    const char* home = std::getenv("HOME");
    cache_dir_ = std::string(home ? home : "/tmp") + "/.cache/KsanaLLM/humming_kernel_files";
  }

  // ── 4. Check HUMMING_REGENERATE ──
  const char* regen_env = std::getenv("HUMMING_REGENERATE");
  if (regen_env && std::string(regen_env) == "1") {
    force_regenerate_ = true;
  }

  // ── 5. Check HUMMING_JIT_LOG ──
  const char* log_env = std::getenv("HUMMING_JIT_LOG");
  if (log_env && std::string(log_env) == "1") {
    verbose_ = true;
  }

  // Ensure cache directory exists
  MkdirP(cache_dir_);
}

HummingJitCompiler::~HummingJitCompiler() {
  // Unload all CUDA modules
  for (auto& mod : modules_) {
    if (mod) {
      cuModuleUnload(mod);
    }
  }
  modules_.clear();
  kernel_table_.clear();
}

// ════════════════════════════════════════════════════════════════════════════════
// Public API
// ════════════════════════════════════════════════════════════════════════════════

void HummingJitCompiler::CompileKernels(const std::vector<KernelConfig>& configs) {
  if (verbose_) {
    std::lock_guard<std::mutex> _ll(log_mutex_);
    std::cout << "[HUMMING JIT] CompileKernels: " << configs.size() << " config(s)"
              << (force_regenerate_ ? " (force regenerate)" : "") << std::endl;
  }

  // Phase 1: Separate cached vs. uncached configs
  std::vector<const KernelConfig*> cached, uncached;
  for (const auto& config : configs) {
    if (!force_regenerate_ && CacheExists(config)) {
      cached.push_back(&config);
    } else {
      uncached.push_back(&config);
    }
  }

  if (verbose_ && !cached.empty()) {
    std::cout << "[HUMMING JIT]   " << cached.size() << " cached, " << uncached.size() << " to build" << std::endl;
  }

  // Phase 2: Fast path — load already-cached cubins in main thread
  for (const auto* config : cached) {
    std::string key = config->CacheKey();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (loaded_keys_.count(key)) continue;  // already loaded from a prior CompileKernels call
    }
    if (verbose_) {
      std::lock_guard<std::mutex> _ll(log_mutex_);
      std::cout << "[HUMMING JIT]   [cache hit] " << key << std::endl;
    }
    std::string cubin_path = GetCacheDir(*config) + "/kernel.cubin";
    LoadAndRegister(*config, cubin_path);
    std::lock_guard<std::mutex> lock(mutex_);
    loaded_keys_.insert(key);
  }

  if (uncached.empty()) return;

  // Phase 3: Generate + compile in parallel across threads.
  // Each config gets its own cache dir and flock — no cross-config contention.
  // Threads do NOT call LoadAndRegister: CUDA driver API is per-thread and
  // spawned threads do not inherit the main thread's CUDA context.  Loading
  // is deferred to Phase 4 in the main thread.
  std::vector<std::thread> threads;
  std::atomic<bool> has_error{false};
  std::string error_msg;
  std::mutex error_mutex;  // protects error_msg
  std::atomic<int> active_count{static_cast<int>(uncached.size())};

  for (const auto* config : uncached) {
    threads.emplace_back([this, config, &has_error, &error_msg, &error_mutex, &active_count]() {
      try {
        std::string dir = GetCacheDir(*config);

        if (verbose_) {
          std::lock_guard<std::mutex> _ll(log_mutex_);
          std::cout << "[HUMMING JIT]   [build] " << config->CacheKey() << std::endl;
        }

        // Intra-process lock: flock() does NOT block sibling threads in the
        // same process, so we serialize per-cache-dir access with a mutex.
        std::mutex* dir_mutex = GetDirMutex(dir);
        std::lock_guard<std::mutex> intra_lock(*dir_mutex);

        MkdirP(dir);
        int lock_fd = AcquireLock(dir);

        // Double-check after acquiring lock (another process/thread may have compiled it)
        if (!force_regenerate_ && CacheExists(*config)) {
          if (verbose_) {
            std::lock_guard<std::mutex> _ll(log_mutex_);
            std::cout << "[HUMMING JIT]     (already built by sibling process)" << std::endl;
          }
          ReleaseLock(lock_fd);
          // Defer loading to Phase 4 (main thread has CUDA context)
          --active_count;
          return;
        }

        // Generate + Compile (the heavy work — benefits from parallelism)
        std::string source_path = dir + "/kernel.cu";
        std::string cubin_path = dir + "/kernel.cubin";
        try {
          GenerateSource(*config, source_path);
          CompileCubin(source_path, cubin_path, include_dir_);
        } catch (...) {
          ReleaseLock(lock_fd);
          throw;
        }
        ReleaseLock(lock_fd);
        // Defer loading to Phase 4 (main thread has CUDA context)

        if (verbose_) {
          --active_count;
          std::lock_guard<std::mutex> _ll(log_mutex_);
          std::cout << "[HUMMING JIT]     compiled (" << active_count.load() << " remaining)" << std::endl;
        }
      } catch (const std::exception& e) {
        if (!has_error.exchange(true)) {
          std::lock_guard<std::mutex> lg(error_mutex);
          error_msg = std::string(e.what()) + " (config: " + config->CacheKey() + ")";
        }
      } catch (...) {
        if (!has_error.exchange(true)) {
          std::lock_guard<std::mutex> lg(error_mutex);
          error_msg = "Unknown exception (config: " + config->CacheKey() + ")";
        }
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  KLLM_KERNEL_CHECK_WITH_INFO(!has_error.load(), "Parallel compilation failed: %s", error_msg.c_str());

  // Phase 4: Load all cubins in the main thread.
  // cuModuleLoad requires a CUDA context current on the calling thread.
  // Spawned threads (Phase 3) do not inherit the main thread's CUDA context,
  // so we load here where the context is guaranteed to be active.
  for (const auto* config : uncached) {
    std::string key = config->CacheKey();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (loaded_keys_.count(key)) continue;  // already loaded from a prior CompileKernels call
    }
    std::string cubin_path = GetCacheDir(*config) + "/kernel.cubin";
    LoadAndRegister(*config, cubin_path);
    std::lock_guard<std::mutex> lock(mutex_);
    loaded_keys_.insert(key);
  }
}

void HummingJitCompiler::CompileKernels(int64_t size_n, int64_t size_k, int64_t num_experts, ComputeDtype compute_dtype,
                                        GemmType gemm_type) {
  // Fast path: skip if this shape has already been compiled
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto key =
        std::make_tuple(size_n, size_k, num_experts, static_cast<int>(compute_dtype), static_cast<int>(gemm_type));
    if (compiled_shapes_.count(key)) return;
  }

  if (verbose_) {
    std::lock_guard<std::mutex> _ll(log_mutex_);
    std::cout << "[HUMMING JIT] CompileKernels(shape): N=" << size_n << " K=" << size_k << " E=" << num_experts
              << " dt=" << static_cast<int>(compute_dtype) << std::endl;
  }

  // Query humming heuristic
  auto configs = QueryKernelConfigs(size_n, size_k, num_experts, compute_dtype, gemm_type);

  // Compile all configs (vector overload — handles cache/load dedup internally)
  CompileKernels(configs);

  // Mark shape as compiled
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto key =
        std::make_tuple(size_n, size_k, num_experts, static_cast<int>(compute_dtype), static_cast<int>(gemm_type));
    compiled_shapes_.insert(key);
  }
}

const KernelMeta* HummingJitCompiler::SelectKernel(int64_t size_m, int64_t size_n, int64_t size_k, GemmType gemm_type,
                                                   ComputeDtype compute_dtype) const {
  std::lock_guard<std::mutex> lock(mutex_);

  const KernelMeta* best = nullptr;
  int64_t best_waste = INT64_MAX;

  for (const auto& meta : kernel_table_) {
    // Must match problem dimensions (N, K), gemm type, and compute dtype
    if (meta.size_n != size_n || meta.size_k != size_k) continue;
    if (meta.gemm_type != gemm_type) continue;
    if (meta.compute_dtype != compute_dtype) continue;

    // If this kernel has a shape_m range from heuristic, prefer range-based selection
    if (meta.min_shape_m > 1 || meta.max_shape_m < INT64_MAX) {
      // Range-based: check if size_m falls within this kernel's optimal range
      if (size_m >= meta.min_shape_m && size_m <= meta.max_shape_m) {
        // Exact range match — this is the best kernel (heuristic-determined)
        return &meta;
      }
      continue;
    }

    // Fallback to tile-waste heuristic for kernels without range metadata
    int64_t block_m = meta.block_m;
    int64_t num_tiles = (size_m + block_m - 1) / block_m;
    int64_t waste = num_tiles * block_m - size_m;

    if (waste < best_waste) {
      best_waste = waste;
      best = &meta;
    }
  }

  return best;
}

// ════════════════════════════════════════════════════════════════════════════════
// Private: Generate / Compile / Load
// ════════════════════════════════════════════════════════════════════════════════

void HummingJitCompiler::GenerateSource(const KernelConfig& config, const std::string& output_path) {
  std::string meta_path = output_path + ".meta.json";

  std::ostringstream cmd;
  cmd << "python3 " << humming_tool_path_ << " generate"
      << " --size_n " << config.size_n << " --size_k " << config.size_k << " --num_experts " << config.num_experts
      << " --block_m " << config.block_m << " --gemm_type " << static_cast<int>(config.gemm_type) << " --compute_dtype "
      << static_cast<int>(config.compute_dtype) << " --output " << output_path << " --meta_output " << meta_path;
  if (config.force_stream_k) cmd << " --force_stream_k 1";
  if (config.force_block_n > 0) cmd << " --force_block_n " << config.force_block_n;
  if (config.force_stages > 0) cmd << " --force_stages " << config.force_stages;

  if (verbose_) {
    std::lock_guard<std::mutex> _ll(log_mutex_);
    std::cout << "[HUMMING JIT]     python: humming_tool.py generate"
              << " N=" << config.size_n << " K=" << config.size_k << " E=" << config.num_experts
              << " block_m=" << config.block_m << " gt=" << static_cast<int>(config.gemm_type)
              << " dt=" << static_cast<int>(config.compute_dtype) << std::endl;
  }

  std::string output;
  int ret = ExecuteCommand(cmd.str(), &output);
  KLLM_KERNEL_CHECK_WITH_INFO(ret == 0, "humming_tool.py generate failed (exit=%d) for config %s.\nOutput: %s", ret,
                              config.CacheKey().c_str(), output.c_str());
  KLLM_KERNEL_CHECK_WITH_INFO(FileExists(output_path), "humming_tool.py generate did not produce output file: %s",
                              output_path.c_str());
}

void HummingJitCompiler::CompileCubin(const std::string& source_path, const std::string& cubin_path,
                                      const std::string& include_dir) {
  if (verbose_) {
    std::lock_guard<std::mutex> lg(log_mutex_);
    std::cout << "[HUMMING JIT]     nvcc: " << source_path << " -> " << cubin_path << std::endl;
  }

  // Write to temp file first, then atomically rename.  This prevents sibling
  // threads in the same process (which share the same flock) from seeing a
  // partially-written cubin when double-checking the cache.
  std::string tmp_path = cubin_path + ".tmp";

  // nvcc flags from humming's NVCCCompiler.get_flags():
  // -cubin -std=c++17 -O3 --use_fast_math
  // --ptxas-options=--register-usage-level=10
  // --expt-relaxed-constexpr --expt-extended-lambda
  // -gencode=arch=compute_90a,code=sm_90a
  // --compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi
  // --diag-suppress=39,161,174,177,940
  std::ostringstream cmd;
  cmd << nvcc_path_ << " " << source_path << " -o " << tmp_path << " -cubin"
      << " -std=c++17"
      << " -O3"
      << " --use_fast_math"
      << " --ptxas-options=--register-usage-level=10"
      << " --expt-relaxed-constexpr"
      << " --expt-extended-lambda"
      << " -gencode=arch=compute_90a,code=sm_90a"
      << " --compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi"
      << " --diag-suppress=39,161,174,177,940"
      << " -I" << include_dir;

  std::string output;
  int ret = ExecuteCommand(cmd.str(), &output);
  KLLM_KERNEL_CHECK_WITH_INFO(ret == 0, "nvcc compilation failed (exit=%d) for %s.\nCommand: %s\nOutput: %s", ret,
                              source_path.c_str(), cmd.str().c_str(), output.c_str());
  KLLM_KERNEL_CHECK_WITH_INFO(FileExists(tmp_path), "nvcc did not produce cubin file: %s", tmp_path.c_str());

  // Atomically rename temp → final.  Only after this point can other threads
  // or processes see the cubin via CacheExists().
  if (rename(tmp_path.c_str(), cubin_path.c_str()) != 0) {
    KLLM_KERNEL_CHECK_WITH_INFO(false, "Failed to rename %s -> %s (errno=%d)", tmp_path.c_str(), cubin_path.c_str(),
                                errno);
  }
}

void HummingJitCompiler::LoadAndRegister(const KernelConfig& config, const std::string& cubin_path) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Step 1: Load the cubin as a CUDA module
  CUmodule module = nullptr;
  CUresult cu_ret = cuModuleLoad(&module, cubin_path.c_str());

  if (verbose_) {
    std::lock_guard<std::mutex> _ll(log_mutex_);
    std::cout << "[HUMMING JIT]     cuModuleLoad: " << cubin_path << std::endl;
  }

  // CUDA_ERROR_INVALID_IMAGE (201) usually means the cubin was corrupted by a
  // prior buggy parallel compilation.  Delete it so the next run regenerates.
  if (cu_ret == 201) {
    std::cerr << "[HUMMING JIT] WARNING: Corrupted cubin detected (CUresult=201), "
              << "deleting: " << cubin_path << std::endl;
    std::cerr << "[HUMMING JIT] This was likely caused by a prior race condition. "
              << "Please re-run to regenerate." << std::endl;
    // Delete the cached files for this config so next run regenerates them
    std::string dir = cubin_path.substr(0, cubin_path.rfind('/'));
    std::string rm_cmd = "rm -rf \"" + dir + "\" 2>/dev/null";
    ExecuteCommand(rm_cmd);
  }

  KLLM_KERNEL_CHECK_WITH_INFO(cu_ret == CUDA_SUCCESS, "cuModuleLoad failed (CUresult=%d) for: %s",
                              static_cast<int>(cu_ret), cubin_path.c_str());
  modules_.push_back(module);

  // Step 2: Find the kernel function symbol from ELF
  std::string kernel_symbol = FindKernelSymbol(cubin_path, "humming");
  KLLM_KERNEL_CHECK_WITH_INFO(!kernel_symbol.empty(), "No kernel function symbol containing 'humming' found in: %s",
                              cubin_path.c_str());

  // Step 3: Get the kernel function handle
  CUfunction func = nullptr;
  cu_ret = cuModuleGetFunction(&func, module, kernel_symbol.c_str());
  KLLM_KERNEL_CHECK_WITH_INFO(cu_ret == CUDA_SUCCESS, "cuModuleGetFunction failed (CUresult=%d) for symbol '%s' in: %s",
                              static_cast<int>(cu_ret), kernel_symbol.c_str(), cubin_path.c_str());

  // Step 4: Read all metadata from the cubin ELF __constant__ symbols
  KernelMeta meta;
  meta.func = func;
  meta.kernel_name = kernel_symbol;
  meta.smem_size = static_cast<int>(ReadCubinUint32(cubin_path, "SMEM_SIZE"));
  meta.num_threads = static_cast<int>(ReadCubinUint32(cubin_path, "NUM_THREADS"));
  meta.size_n = static_cast<int64_t>(ReadCubinUint32(cubin_path, "PROBLEM_SHAPE_N"));
  meta.size_k = static_cast<int64_t>(ReadCubinUint32(cubin_path, "PROBLEM_SHAPE_K"));
  meta.block_m = static_cast<int>(ReadCubinUint32(cubin_path, "BLOCK_SHAPE_M"));
  meta.block_n = static_cast<int>(ReadCubinUint32(cubin_path, "BLOCK_SHAPE_N"));
  meta.block_k = static_cast<int>(ReadCubinUint32(cubin_path, "BLOCK_SHAPE_K"));
  meta.num_stages = static_cast<int>(ReadCubinUint32(cubin_path, "NUM_STAGES"));
  meta.num_ctas_per_sm = static_cast<int>(ReadCubinUint32(cubin_path, "NUM_CTAS_PER_SM"));
  meta.use_tma_a = ReadCubinBool(cubin_path, "USE_TMA_A");
  meta.use_tma_b = ReadCubinBool(cubin_path, "USE_TMA_B");
  meta.use_tma_c = ReadCubinBool(cubin_path, "USE_TMA_C");
  meta.use_tma_bs = ReadCubinBool(cubin_path, "USE_TMA_BS");
  meta.use_stream_k = ReadCubinBool(cubin_path, "USE_STREAM_K");
  meta.use_warp_spec = ReadCubinBool(cubin_path, "USE_WARP_SPEC");
  meta.use_fused_e8m0_scale = ReadCubinBool(cubin_path, "USE_FUSED_E8M0_SCALE");

  uint32_t gemm_type_id = ReadCubinUint32(cubin_path, "GEMM_TYPE_ID");
  meta.gemm_type = static_cast<GemmType>(gemm_type_id);
  meta.compute_dtype = config.compute_dtype;

  // Step 5: Set shared memory attribute
  cu_ret = cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, meta.smem_size);
  KLLM_KERNEL_CHECK_WITH_INFO(cu_ret == CUDA_SUCCESS, "cuFuncSetAttribute (smem=%d) failed (CUresult=%d) for: %s",
                              meta.smem_size, static_cast<int>(cu_ret), cubin_path.c_str());

  // Store the shape_m range from the config (heuristic range metadata)
  meta.min_shape_m = config.min_shape_m;
  meta.max_shape_m = config.max_shape_m;

  kernel_table_.push_back(meta);

  if (verbose_) {
    std::lock_guard<std::mutex> _ll(log_mutex_);
    std::cout << "[HUMMING JIT]     loaded: " << kernel_symbol << " smem=" << meta.smem_size
              << " threads=" << meta.num_threads << " N=" << meta.size_n << " K=" << meta.size_k << " block=("
              << meta.block_m << "," << meta.block_n << "," << meta.block_k << ")"
              << " ctas=" << meta.num_ctas_per_sm << " sk=" << meta.use_stream_k << " ws=" << meta.use_warp_spec
              << " tma=(" << meta.use_tma_a << meta.use_tma_b << meta.use_tma_c << meta.use_tma_bs << ")" << std::endl;
  }
}

// ════════════════════════════════════════════════════════════════════════════════
// Public: QueryKernelConfigs
// ════════════════════════════════════════════════════════════════════════════════

std::vector<HummingJitCompiler::KernelConfig> HummingJitCompiler::QueryKernelConfigs(int64_t size_n, int64_t size_k,
                                                                                     int64_t num_experts,
                                                                                     ComputeDtype compute_dtype,
                                                                                     GemmType gemm_type) {
  // Call Python script to query humming heuristic
  std::ostringstream cmd;
  cmd << "python3 " << humming_tool_path_ << " query-configs"
      << " --size_n " << size_n << " --size_k " << size_k << " --num_experts " << num_experts << " --compute_dtype "
      << static_cast<int>(compute_dtype);

  if (verbose_) {
    std::lock_guard<std::mutex> _ll(log_mutex_);
    std::cout << "[HUMMING JIT] QueryKernelConfigs: N=" << size_n << " K=" << size_k << " E=" << num_experts
              << " dt=" << static_cast<int>(compute_dtype) << std::endl;
  }

  std::string output;
  int ret = ExecuteCommand(cmd.str(), &output);

  std::vector<KernelConfig> configs;

  if (ret != 0 || output.empty()) {
    // Fallback: use default block_m={64, 128}
    configs.push_back({size_n, size_k, num_experts, 64, gemm_type, compute_dtype, 1, 64});
    configs.push_back({size_n, size_k, num_experts, 128, gemm_type, compute_dtype, 65, INT64_MAX});
    return configs;
  }

  // Parse JSON output: [{"shape_m_representative": N, "min_shape_m": M1, "max_shape_m": M2, ...}, ...]
  // Simple JSON parsing (no external dependency)
  size_t pos = 0;
  while ((pos = output.find("\"shape_m_representative\"", pos)) != std::string::npos) {
    // Extract shape_m_representative (used as --block_m in humming_tool.py generate)
    size_t colon = output.find(':', pos);
    size_t comma = output.find_first_of(",}", colon);
    std::string val_str = output.substr(colon + 1, comma - colon - 1);
    // Trim whitespace
    val_str.erase(0, val_str.find_first_not_of(" \t\n\r"));
    val_str.erase(val_str.find_last_not_of(" \t\n\r") + 1);
    int block_m = std::stoi(val_str);

    // Extract min_shape_m
    size_t min_pos = output.find("\"min_shape_m\"", pos);
    colon = output.find(':', min_pos);
    comma = output.find_first_of(",}", colon);
    val_str = output.substr(colon + 1, comma - colon - 1);
    val_str.erase(0, val_str.find_first_not_of(" \t\n\r"));
    val_str.erase(val_str.find_last_not_of(" \t\n\r") + 1);
    int64_t min_m = std::stoll(val_str);

    // Extract max_shape_m
    size_t max_pos = output.find("\"max_shape_m\"", pos);
    colon = output.find(':', max_pos);
    comma = output.find_first_of(",}", colon);
    val_str = output.substr(colon + 1, comma - colon - 1);
    val_str.erase(0, val_str.find_first_not_of(" \t\n\r"));
    val_str.erase(val_str.find_last_not_of(" \t\n\r") + 1);
    int64_t max_m = std::stoll(val_str);

    configs.push_back({size_n, size_k, num_experts, block_m, gemm_type, compute_dtype, min_m, max_m});

    pos = (max_pos != std::string::npos) ? max_pos + 1 : comma + 1;
  }

  // GEMM2 optimization: detect small-K large-N shapes and inject optimized configs.
  // The Python heuristic disables stream-K for K<=1024 and always uses block_n=128,
  // both suboptimal for GEMM2's (small K, large N) shape at small-medium M.
  // For large M (3073+), the heuristic configs (block_n=128, stream_k=false, stages=4)
  // are already near-optimal, so we keep them and only override small-medium M.
  if (size_k <= 512 && size_n >= 4096) {
    // Remove heuristic configs for small-medium M (they use stream_k=false, block_n=128)
    // Keep heuristic configs for large M (3073+) — they're optimal for large M
    configs.erase(
        std::remove_if(configs.begin(), configs.end(), [](const KernelConfig& c) { return c.max_shape_m <= 3072; }),
        configs.end());

    // Add optimized configs for small-medium M only
    // Small M (1-64 tokens, M_eff=6-384): block_m=64, stream-K, block_n=256, stages=3
    configs.push_back({size_n, size_k, num_experts, 64, gemm_type, compute_dtype, 1, 384, true, 256, 3});
    // Medium M (65-512 tokens, M_eff=390-3072): block_m=64, stream-K, block_n=256, stages=3
    configs.push_back({size_n, size_k, num_experts, 64, gemm_type, compute_dtype, 385, 3072, true, 256, 3});

    if (verbose_) {
      std::lock_guard<std::mutex> _ll(log_mutex_);
      std::cout << "[HUMMING JIT]   GEMM2 override: " << configs.size()
                << " config(s) (small-M stream_k=true, block_n=256; large-M heuristic)" << std::endl;
    }
  }

  // ── Variant config generation ──────────────────────────────────────────
  // Generate variant configs to expand the candidate space for profile-based
  // kernel selection. The profiler uses real weights (not dummy), so TMA
  // descriptor layout is correct for all block_n/block_k combinations.
  // For each heuristic config, generate up to 3 variants by flipping dimensions
  // the heuristic may have chosen suboptimally for H20. Priority order:
  //   1. stream_k variant (force stream_k ON; flip false→true only)
  //   2. block_n variant (try 256 instead of heuristic 128)
  //   3. stages variant (try stages=3 as alternative)
  // Generate up to 3 variants per source config (dedup may reduce count).
  // Hard cap: 25 total configs (enforced by dedup + truncation below).
  {
    std::vector<KernelConfig> base_configs = configs;  // snapshot before adding variants
    for (const auto& base : base_configs) {
      std::vector<KernelConfig> variants;
      // 1. stream_k variant (force stream_k ON)
      // Only flip false→true: force_stream_k==false means "use heuristic" (not
      // "force off") since GenerateSource only emits --force_stream_k 1 when
      // true. Flipping true→false would just reproduce the heuristic default
      // and likely duplicate an existing config.
      if (!base.force_stream_k) {
        KernelConfig v = base;
        v.force_stream_k = true;
        // Keep heuristic block_n/stages by leaving force_block_n/force_stages as inherited
        // (force_stream_k overrides use_stream_k in GenerateSource)
        variants.push_back(v);
      }
      // 2. block_n variant (try 256 instead of heuristic 128)
      {
        KernelConfig v = base;
        // Heuristic default is 128 for these shapes; 256 may improve large-N
        // throughput by reducing block count.
        v.force_block_n = 256;
        variants.push_back(v);
      }
      // 3. stages variant (try stages=3 as alternative)
      {
        KernelConfig v = base;
        // We don't know the heuristic stages here; pick stages=3 as a common alternative
        v.force_stages = 3;
        variants.push_back(v);
      }

      // Dedup variants against existing configs (by block_m, force_stream_k, force_block_n, force_stages)
      for (auto& v : variants) {
        if (configs.size() >= 25) break;  // hard cap
        bool dup = false;
        for (const auto& c : configs) {
          if (c.block_m == v.block_m && c.force_stream_k == v.force_stream_k && c.force_block_n == v.force_block_n &&
              c.force_stages == v.force_stages) {
            dup = true;
            break;
          }
        }
        if (!dup) {
          configs.push_back(v);
        }
      }
      if (configs.size() >= 25) break;
    }

    if (verbose_) {
      std::lock_guard<std::mutex> _ll(log_mutex_);
      std::cout << "[HUMMING JIT]   variant generation: " << configs.size() << " config(s) total (heuristic + variants)"
                << std::endl;
    }
  }

  if (configs.empty()) {
    // Fallback
    if (verbose_) {
      std::lock_guard<std::mutex> _ll(log_mutex_);
      std::cout << "[HUMMING JIT]   heuristic query returned empty, using fallback configs" << std::endl;
    }
    configs.push_back({size_n, size_k, num_experts, 64, gemm_type, compute_dtype, 1, 64});
    configs.push_back({size_n, size_k, num_experts, 128, gemm_type, compute_dtype, 65, INT64_MAX});
  } else if (verbose_) {
    std::lock_guard<std::mutex> _ll(log_mutex_);
    std::cout << "[HUMMING JIT]   heuristic returned " << configs.size() << " config(s)" << std::endl;
  }

  return configs;
}

// ════════════════════════════════════════════════════════════════════════════════
// Private: Cache management
// ════════════════════════════════════════════════════════════════════════════════

bool HummingJitCompiler::CacheExists(const KernelConfig& config) const {
  std::string cubin_path = GetCacheDir(config) + "/kernel.cubin";
  return FileExists(cubin_path);
}

std::string HummingJitCompiler::GetCacheDir(const KernelConfig& config) const {
  return cache_dir_ + "/" + config.CacheKey();
}

// ════════════════════════════════════════════════════════════════════════════════
// Private: Multi-process locking
// ════════════════════════════════════════════════════════════════════════════════

int HummingJitCompiler::AcquireLock(const std::string& dir_path) {
  std::string lock_path = dir_path + "/.lock";
  int fd = open(lock_path.c_str(), O_CREAT | O_RDWR, 0666);
  KLLM_KERNEL_CHECK_WITH_INFO(fd >= 0, "Failed to open lock file: %s", lock_path.c_str());

  int ret = flock(fd, LOCK_EX);
  KLLM_KERNEL_CHECK_WITH_INFO(ret == 0, "flock(LOCK_EX) failed for: %s", lock_path.c_str());

  return fd;
}

void HummingJitCompiler::ReleaseLock(int fd) {
  if (fd >= 0) {
    flock(fd, LOCK_UN);
    close(fd);
  }
}

// ════════════════════════════════════════════════════════════════════════════════
// Static: ELF/Cubin reader utilities
// ════════════════════════════════════════════════════════════════════════════════

uint32_t HummingJitCompiler::ReadCubinUint32(const std::string& cubin_path, const std::string& symbol_name) {
  CubinElfReader reader(cubin_path);
  return reader.GetUint32(symbol_name);
}

bool HummingJitCompiler::ReadCubinBool(const std::string& cubin_path, const std::string& symbol_name) {
  CubinElfReader reader(cubin_path);
  return reader.GetBool(symbol_name);
}

std::string HummingJitCompiler::FindKernelSymbol(const std::string& cubin_path, const std::string& base_name) {
  CubinElfReader reader(cubin_path);
  return reader.FindFuncSymbol(base_name);
}

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
