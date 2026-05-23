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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "csrc/utils/nvidia/assert.h"

namespace llm_kernels {
namespace nvidia {
namespace humming {

// ════════════════════════════════════════════════════════════════════════════════
// ELF64 structures for cubin reading (self-contained, no external elf.h dependency)
// ════════════════════════════════════════════════════════════════════════════════

namespace {

constexpr uint32_t kShtSymtab = 2;   // SHT_SYMTAB
constexpr uint32_t kShtNobits = 8;   // SHT_NOBITS
constexpr uint8_t kSttFunc = 2;      // STT_FUNC

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
        ehdr.e_ident[0] == 0x7f && ehdr.e_ident[1] == 'E' &&
            ehdr.e_ident[2] == 'L' && ehdr.e_ident[3] == 'F',
        "Not a valid ELF file: %s", path.c_str());

    // Read section headers
    std::vector<Elf64Shdr> shdrs(ehdr.e_shnum);
    fs.seekg(static_cast<std::streamoff>(ehdr.e_shoff));
    fs.read(reinterpret_cast<char*>(shdrs.data()),
            static_cast<std::streamsize>(sizeof(Elf64Shdr) * ehdr.e_shnum));
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
    KLLM_KERNEL_CHECK_WITH_INFO(it != symbol_offsets_.end(),
                                "Symbol '%s' not found in cubin: %s",
                                name.c_str(), path_.c_str());

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
  bool GetBool(const std::string& name) const {
    return GetUint32(name) > 0;
  }

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
  void ReadSymbolTable(std::ifstream& fs, const std::vector<Elf64Shdr>& shdrs,
                       uint16_t sym_idx) {
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
    fs.read(reinterpret_cast<char*>(syms.data()),
            static_cast<std::streamsize>(symtab.sh_size));

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

}  // anonymous namespace

// ════════════════════════════════════════════════════════════════════════════════
// KernelConfig
// ════════════════════════════════════════════════════════════════════════════════

std::string HummingJitCompiler::KernelConfig::CacheKey() const {
  std::ostringstream oss;
  oss << "n" << size_n
      << "_k" << size_k
      << "_e" << num_experts
      << "_bm" << block_m
      << "_gt" << static_cast<int>(gemm_type)
      << "_dt" << static_cast<int>(compute_dtype);
  return oss.str();
}

// ════════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════════

HummingJitCompiler::HummingJitCompiler()
    : nvcc_path_(FindNvcc()),
      force_regenerate_(false) {
  // ── 1. Resolve base_dir: HUMMING_BASE_DIR env, or derive from CWD=build/ ──
  const char* base_env = std::getenv("HUMMING_BASE_DIR");
  if (base_env && base_env[0] != '\0') {
    base_dir_ = base_env;
  } else {
    // Default: assume CWD is KsanaLLM/build/, derive relative path to humming operator
    base_dir_ = "../3rdparty/LLM_kernels/csrc/kernels/nvidia/others/humming";
  }

  // ── 2. Derived paths ──
  include_dir_ = base_dir_ + "/include";
  script_dir_ = base_dir_ + "/jit";

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
  for (const auto& config : configs) {
    std::string dir = GetCacheDir(config);
    std::string cubin_path = dir + "/kernel.cubin";

    // Fast path: already cached and not forcing regen
    if (!force_regenerate_ && CacheExists(config)) {
      LoadAndRegister(config, cubin_path);
      continue;
    }

    // Acquire cross-process lock for this kernel variant
    MkdirP(dir);
    int lock_fd = AcquireLock(dir);

    // Double-check after acquiring lock (another process may have compiled it)
    if (!force_regenerate_ && CacheExists(config)) {
      ReleaseLock(lock_fd);
      LoadAndRegister(config, cubin_path);
      continue;
    }

    // Step 1: Generate .cu source
    std::string source_path = dir + "/kernel.cu";
    GenerateSource(config, source_path);

    // Step 2: Compile .cu -> .cubin
    CompileCubin(source_path, cubin_path, include_dir_);

    ReleaseLock(lock_fd);

    // Step 3: Load and register the kernel
    LoadAndRegister(config, cubin_path);
  }
}

const KernelMeta* HummingJitCompiler::SelectKernel(int64_t size_m, int64_t size_n,
                                                   int64_t size_k, GemmType gemm_type,
                                                   ComputeDtype compute_dtype) const {
  std::lock_guard<std::mutex> lock(mutex_);

  const KernelMeta* best = nullptr;
  int64_t best_waste = INT64_MAX;

  for (const auto& meta : kernel_table_) {
    // Must match problem dimensions (N, K), gemm type, and compute dtype
    if (meta.size_n != size_n || meta.size_k != size_k) continue;
    if (meta.gemm_type != gemm_type) continue;
    if (meta.compute_dtype != compute_dtype) continue;

    // If this kernel has a shape_m range from H20 heuristic, prefer range-based selection
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

void HummingJitCompiler::GenerateSource(const KernelConfig& config,
                                        const std::string& output_path) {
  std::string meta_path = output_path + ".meta.json";

  std::ostringstream cmd;
  cmd << "python3 " << script_dir_ << "/generate_kernel.py"
      << " --size_n " << config.size_n
      << " --size_k " << config.size_k
      << " --num_experts " << config.num_experts
      << " --block_m " << config.block_m
      << " --gemm_type " << static_cast<int>(config.gemm_type)
      << " --compute_dtype " << static_cast<int>(config.compute_dtype)
      << " --output " << output_path
      << " --meta_output " << meta_path;

  std::string output;
  int ret = ExecuteCommand(cmd.str(), &output);
  KLLM_KERNEL_CHECK_WITH_INFO(ret == 0,
                              "generate_kernel.py failed (exit=%d) for config %s.\nOutput: %s",
                              ret, config.CacheKey().c_str(), output.c_str());
  KLLM_KERNEL_CHECK_WITH_INFO(FileExists(output_path),
                              "generate_kernel.py did not produce output file: %s",
                              output_path.c_str());
}

void HummingJitCompiler::CompileCubin(const std::string& source_path,
                                      const std::string& cubin_path,
                                      const std::string& include_dir) {
  // nvcc flags from humming's NVCCCompiler.get_flags():
  // -cubin -std=c++17 -O3 --use_fast_math
  // --ptxas-options=--register-usage-level=10
  // --expt-relaxed-constexpr --expt-extended-lambda
  // -gencode=arch=compute_90a,code=sm_90a
  // --compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi
  // --diag-suppress=39,161,174,177,940
  std::ostringstream cmd;
  cmd << nvcc_path_
      << " " << source_path
      << " -o " << cubin_path
      << " -cubin"
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
  KLLM_KERNEL_CHECK_WITH_INFO(ret == 0,
                              "nvcc compilation failed (exit=%d) for %s.\nCommand: %s\nOutput: %s",
                              ret, source_path.c_str(), cmd.str().c_str(), output.c_str());
  KLLM_KERNEL_CHECK_WITH_INFO(FileExists(cubin_path),
                              "nvcc did not produce cubin file: %s", cubin_path.c_str());
}

void HummingJitCompiler::LoadAndRegister(const KernelConfig& config,
                                         const std::string& cubin_path) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Step 1: Load the cubin as a CUDA module
  CUmodule module = nullptr;
  CUresult cu_ret = cuModuleLoad(&module, cubin_path.c_str());
  KLLM_KERNEL_CHECK_WITH_INFO(cu_ret == CUDA_SUCCESS,
                              "cuModuleLoad failed (CUresult=%d) for: %s",
                              static_cast<int>(cu_ret), cubin_path.c_str());
  modules_.push_back(module);

  // Step 2: Find the kernel function symbol from ELF
  std::string kernel_symbol = FindKernelSymbol(cubin_path, "humming");
  KLLM_KERNEL_CHECK_WITH_INFO(!kernel_symbol.empty(),
                              "No kernel function symbol containing 'humming' found in: %s",
                              cubin_path.c_str());

  // Step 3: Get the kernel function handle
  CUfunction func = nullptr;
  cu_ret = cuModuleGetFunction(&func, module, kernel_symbol.c_str());
  KLLM_KERNEL_CHECK_WITH_INFO(cu_ret == CUDA_SUCCESS,
                              "cuModuleGetFunction failed (CUresult=%d) for symbol '%s' in: %s",
                              static_cast<int>(cu_ret), kernel_symbol.c_str(),
                              cubin_path.c_str());

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
  cu_ret = cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                              meta.smem_size);
  KLLM_KERNEL_CHECK_WITH_INFO(cu_ret == CUDA_SUCCESS,
                              "cuFuncSetAttribute (smem=%d) failed (CUresult=%d) for: %s",
                              meta.smem_size, static_cast<int>(cu_ret), cubin_path.c_str());

  // Store the shape_m range from the config (H20 heuristic range metadata)
  meta.min_shape_m = config.min_shape_m;
  meta.max_shape_m = config.max_shape_m;

  kernel_table_.push_back(meta);
}

// ════════════════════════════════════════════════════════════════════════════════
// Public: QueryH20Configs
// ════════════════════════════════════════════════════════════════════════════════

std::vector<HummingJitCompiler::KernelConfig>
HummingJitCompiler::QueryH20Configs(int64_t size_n, int64_t size_k,
                                     int64_t num_experts, ComputeDtype compute_dtype,
                                     GemmType gemm_type) {
  // Call Python script to query humming H20 heuristic
  std::ostringstream cmd;
  cmd << "python3 " << script_dir_ << "/get_h20_configs.py"
      << " --size_n " << size_n
      << " --size_k " << size_k
      << " --num_experts " << num_experts
      << " --compute_dtype " << static_cast<int>(compute_dtype);

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
    // Extract shape_m_representative (used as --block_m in generate_kernel.py)
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

  if (configs.empty()) {
    // Fallback
    configs.push_back({size_n, size_k, num_experts, 64, gemm_type, compute_dtype, 1, 64});
    configs.push_back({size_n, size_k, num_experts, 128, gemm_type, compute_dtype, 65, INT64_MAX});
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

uint32_t HummingJitCompiler::ReadCubinUint32(const std::string& cubin_path,
                                             const std::string& symbol_name) {
  CubinElfReader reader(cubin_path);
  return reader.GetUint32(symbol_name);
}

bool HummingJitCompiler::ReadCubinBool(const std::string& cubin_path,
                                       const std::string& symbol_name) {
  CubinElfReader reader(cubin_path);
  return reader.GetBool(symbol_name);
}

std::string HummingJitCompiler::FindKernelSymbol(const std::string& cubin_path,
                                                 const std::string& base_name) {
  CubinElfReader reader(cubin_path);
  return reader.FindFuncSymbol(base_name);
}

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
