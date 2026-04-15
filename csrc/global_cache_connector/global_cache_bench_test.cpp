/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cuda_runtime.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "global_cache_connector/global_cache_connector_factory.h"

using namespace ksana_llm;

// ===== DeepSeek-R1 模型参数 (W4A8 量化, FP8 KV cache) =====
static constexpr int kNodeLayerNum = 62;     // 61 attention + 1 MTP
static constexpr int kKVDimsPerToken = 576;  // kv_lora_rank=512 + qk_rope_head_dim=64
static constexpr int kBlockSize = 64;        // tokens per block
static constexpr size_t kBlockBytes = kNodeLayerNum * kKVDimsPerToken * 1 * kBlockSize;  // 2,285,568

// ===== 信号机制 (通过 PutFirstTokens / QueryFirstTokens) =====
static constexpr int kSignalTokenBase = -999999;

struct SignalPayload {
  int32_t seq_id;
  int32_t num_tokens;
  int32_t iterations;
  int32_t run_id;
  int32_t max_tokens;  // 固定步长，reader 用此重建 token IDs
};

static std::vector<int> MakeSignalKey(int signal_seq) {
  // 用单个负数作为 token_ids key，每个 signal_seq 唯一
  return {kSignalTokenBase - signal_seq};
}

static std::vector<int> EncodePayload(const SignalPayload& p) {
  return {p.seq_id, p.num_tokens, p.iterations, p.run_id, p.max_tokens};
}

static SignalPayload DecodePayload(const std::vector<int>& tokens) {
  SignalPayload p{};
  if (tokens.size() >= 5) {
    p.seq_id = tokens[0];
    p.num_tokens = tokens[1];
    p.iterations = tokens[2];
    p.run_id = tokens[3];
    p.max_tokens = tokens[4];
  }
  return p;
}

// ===== CLI helpers =====
static std::vector<int> ParseIntList(const std::string& s) {
  std::vector<int> result;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    result.push_back(std::stoi(item));
  }
  return result;
}

struct Round {
  int num_tokens;
  int iterations;
};

static std::vector<Round> ZipRounds(const std::vector<int>& tokens, const std::vector<int>& iters) {
  std::vector<Round> rounds;
  if (iters.size() == 1) {
    // 广播：单个 iterations 值应用到所有 num_tokens
    for (int nt : tokens) rounds.push_back({nt, iters[0]});
  } else if (tokens.size() == 1) {
    // 广播：单个 num_tokens 值应用到所有 iterations
    for (int it : iters) rounds.push_back({tokens[0], it});
  } else if (tokens.size() == iters.size()) {
    for (size_t i = 0; i < tokens.size(); ++i) rounds.push_back({tokens[i], iters[i]});
  } else {
    fprintf(stderr, "Error: --num_tokens and --iterations must have same length or one must be single value\n");
    exit(1);
  }
  return rounds;
}

static void PrintUsage(const char* prog) {
  fprintf(stderr,
          "Usage: %s --config <config.yaml> --mode <writer|reader> [options]\n"
          "\n"
          "Writer: write data -> signal reader per iteration -> wait reader done -> exit\n"
          "Reader: wait for signals -> fetch data per iteration -> report -> send done\n"
          "\n"
          "Options:\n"
          "  --config <path>       YAML config file (required)\n"
          "  --mode <writer|reader> (required)\n"
          "  --num_tokens <list>   comma-separated token counts (default: 8192, multiple of %d)\n"
          "  --iterations <list>   comma-separated repeat counts (default: 3)\n"
          "  --tp_size <n>         tensor parallel size (default: 1)\n"
          "  --device_id <n>       CUDA device (default: 0)\n"
          "  --verbose             verbose output\n"
          "\n"
          "Examples:\n"
          "  --num_tokens 4096,8192,64000 --iterations 10     # 3 rounds, 10 iters each\n"
          "  --num_tokens 4096,64000 --iterations 3,10        # 2 rounds: (4096,3) (64000,10)\n"
          "\n"
          "Block: %zu bytes (%.2f MB)  [%d layers x 576 x FP8 x %d tokens]\n",
          prog, kBlockSize, kBlockBytes, kBlockBytes / (1024.0 * 1024.0), kNodeLayerNum, kBlockSize);
}

// ===== YAML helpers =====
static YAML::Node GetYamlNode(const YAML::Node& root, const std::string& dotpath) {
  YAML::Node node = YAML::Clone(root);
  std::string path = dotpath;
  size_t pos = 0;
  while ((pos = path.find('.')) != std::string::npos) {
    std::string key = path.substr(0, pos);
    path = path.substr(pos + 1);
    if (!node.IsMap() || !node[key]) return YAML::Node();
    node = node[key];
  }
  if (!node.IsMap() || !node[path]) return YAML::Node();
  return node[path];
}

template <typename T>
static T YamlGet(const YAML::Node& root, const std::string& dotpath, const T& default_val) {
  YAML::Node node = GetYamlNode(root, dotpath);
  if (node.IsScalar()) return node.as<T>();
  return default_val;
}

static GlobalCacheConnectorConfig ParseConnectorConfig(const YAML::Node& root, int num_tokens) {
  GlobalCacheConnectorConfig cfg;
  cfg.service_endpoint = YamlGet<std::string>(root, "setting.global_cache_connector.service_endpoint", "");
  cfg.backend_type = YamlGet<std::string>(root, "setting.global_cache_connector.backend_type", "mooncake-mock");
  cfg.timeout_ms = YamlGet<int>(root, "setting.global_cache_connector.timeout_ms", 5000);
  cfg.max_retry = YamlGet<int>(root, "setting.global_cache_connector.max_retry", 3);
  cfg.fetch_thread_num = YamlGet<int>(root, "setting.global_cache_connector.fetch_thread_num", 4);
  cfg.role = YamlGet<std::string>(root, "setting.global_cache_connector.role", "");
  cfg.protocol = YamlGet<std::string>(root, "setting.global_cache_connector.protocol", "tcp");
  cfg.rdma_devices = YamlGet<std::string>(root, "setting.global_cache_connector.rdma_devices", "");
  cfg.master_server_addr =
      YamlGet<std::string>(root, "setting.global_cache_connector.master_server_addr", "127.0.0.1:50051");
  cfg.global_segment_size =
      YamlGet<size_t>(root, "setting.global_cache_connector.global_segment_size", 16 * 1024 * 1024);
  cfg.local_buffer_size = YamlGet<size_t>(root, "setting.global_cache_connector.local_buffer_size", 16 * 1024 * 1024);
  size_t yaml_max = YamlGet<size_t>(root, "setting.batch_scheduler.max_step_tokens", static_cast<size_t>(num_tokens));
  cfg.max_step_tokens = std::max(yaml_max, static_cast<size_t>(num_tokens));
  return cfg;
}

// ===== Timing =====
static double NowMs() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();
}

struct LatencyStats {
  std::vector<double> samples_ms;
  void Add(double ms) { samples_ms.push_back(ms); }
  void Clear() { samples_ms.clear(); }
  double Avg() const {
    if (samples_ms.empty()) return 0;
    return std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) / samples_ms.size();
  }
  double Percentile(double p) const {
    if (samples_ms.empty()) return 0;
    auto sorted = samples_ms;
    std::sort(sorted.begin(), sorted.end());
    size_t idx = static_cast<size_t>(p / 100.0 * (sorted.size() - 1));
    return sorted[idx];
  }
  void Print(const char* label, size_t total_bytes) const {
    if (samples_ms.empty()) return;
    double avg = Avg(), p50 = Percentile(50), p99 = Percentile(99);
    double tp = avg > 0 ? (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (avg / 1000.0) : 0;
    fprintf(stderr, "  %-20s avg=%.3f ms  p50=%.3f ms  p99=%.3f ms  (n=%zu)  %.2f GB/s\n", label, avg, p50, p99,
            samples_ms.size(), tp);
  }
};

// ===== GPU memory =====

// 生成确定性填充数据（用于写后校验）
static void FillBlockPattern(uint8_t* buf, int seq_id, int block_idx) {
  uint32_t seed = static_cast<uint32_t>(seq_id * 100003 + block_idx * 997);
  for (size_t j = 0; j < kBlockBytes; ++j) {
    seed = seed * 1103515245 + 12345;
    buf[j] = static_cast<uint8_t>((seed >> 16) & 0xFF);
  }
}

static size_t VerifyBlockPattern(const uint8_t* buf, int seq_id, int block_idx) {
  std::vector<uint8_t> expected(kBlockBytes);
  FillBlockPattern(expected.data(), seq_id, block_idx);
  size_t mismatch = 0;
  for (size_t j = 0; j < kBlockBytes; ++j) {
    if (buf[j] != expected[j]) ++mismatch;
  }
  return mismatch;
}

struct GpuBlocks {
  std::vector<void*> ptrs;
  int device_id = 0;
  int current_blocks = 0;

  bool Alloc(int device, int num_blocks, int seq_id = -1) {
    Free();
    device_id = device;
    cudaSetDevice(device);
    ptrs.resize(num_blocks);
    std::vector<uint8_t> buf(kBlockBytes);
    for (int i = 0; i < num_blocks; ++i) {
      if (cudaMalloc(&ptrs[i], kBlockBytes) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for block %d on device %d\n", i, device);
        return false;
      }
      if (seq_id >= 0) {
        FillBlockPattern(buf.data(), seq_id, i);
      } else {
        std::mt19937 rng(42 + i);
        for (size_t j = 0; j < kBlockBytes; ++j) buf[j] = rng() & 0xFF;
      }
      cudaMemcpy(ptrs[i], buf.data(), kBlockBytes, cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    current_blocks = num_blocks;
    return true;
  }

  void FillPattern(int seq_id, int num_blocks) {
    cudaSetDevice(device_id);
    std::vector<uint8_t> buf(kBlockBytes);
    for (int i = 0; i < num_blocks; ++i) {
      FillBlockPattern(buf.data(), seq_id, i);
      cudaMemcpy(ptrs[i], buf.data(), kBlockBytes, cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
  }

  void Free() {
    if (ptrs.empty()) return;
    cudaSetDevice(device_id);
    for (auto p : ptrs) cudaFree(p);
    ptrs.clear();
    current_blocks = 0;
  }

  ~GpuBlocks() { Free(); }
};

// ===== Token sequence =====
static int32_t g_run_id = 0;
static int g_max_tokens = 0;  // 固定步长，避免不同 round 的 token ID 重叠

static std::vector<int> MakeTokenIds(int seq_id, int num_tokens) {
  std::vector<int> ids(num_tokens);
  int base = g_run_id + seq_id * g_max_tokens;  // 用 max_tokens 做步长
  for (int i = 0; i < num_tokens; ++i) ids[i] = base + i;
  return ids;
}

// ===== DoWriteOnce: 单次写入，返回延迟 ms =====
static double DoWriteOnce(GlobalCacheConnectorInterface* conn, int tp_size, int tp_rank, int num_tokens, int num_blocks,
                          int seq_id, GpuBlocks& gpu) {
  gpu.FillPattern(seq_id, num_blocks);
  auto token_ids = MakeTokenIds(seq_id, num_tokens);
  std::vector<int> block_ranges = {0, num_tokens - 1};

  double t0 = NowMs();
  Status s =
      conn->PutGlobalCacheBlocks(token_ids, tp_size, tp_rank, 0, kBlockSize, kBlockBytes, block_ranges, gpu.ptrs);
  double t1 = NowMs();

  if (!s.OK()) {
    fprintf(stderr, "  WRITE FAILED: %s\n", s.GetMessage().c_str());
    return -1;
  }
  return t1 - t0;
}

// ===== DoReadOnce: 单次读取 + 校验 =====
struct ReadResult {
  double latency_ms;
  int matched_blocks;
  int num_ranges;
  int verify_ok;
  int verify_fail;
};

static ReadResult DoReadOnce(GlobalCacheConnectorInterface* conn, int tp_size, int tp_rank, int num_tokens,
                             int num_blocks, int seq_id, GpuBlocks& gpu) {
  ReadResult result{};

  // 清零 GPU
  cudaSetDevice(gpu.device_id);
  for (int i = 0; i < num_blocks; ++i) cudaMemset(gpu.ptrs[i], 0, kBlockBytes);
  cudaDeviceSynchronize();

  auto token_ids = MakeTokenIds(seq_id, num_tokens);
  std::vector<int> matched_ranges;

  double t0 = NowMs();
  Status s = conn->QueryAndFetch(token_ids, 0, tp_size, tp_rank, 0, kBlockSize, kBlockBytes, gpu.ptrs, matched_ranges);
  double t1 = NowMs();
  result.latency_ms = t1 - t0;

  if (!s.OK()) {
    fprintf(stderr, "  READ FAILED: %s\n", s.GetMessage().c_str());
    result.latency_ms = -1;
    return result;
  }

  // matched_ranges -> block 数
  result.num_ranges = static_cast<int>(matched_ranges.size()) / 2;
  for (int ri = 0; ri < result.num_ranges; ++ri) {
    int start_tok = matched_ranges[ri * 2];
    int end_tok = matched_ranges[ri * 2 + 1];
    result.matched_blocks += end_tok / kBlockSize - start_tok / kBlockSize + 1;
  }

  // 数据校验
  std::vector<uint8_t> host_buf(kBlockBytes);
  for (int bi = 0; bi < num_blocks; ++bi) {
    cudaMemcpy(host_buf.data(), gpu.ptrs[bi], kBlockBytes, cudaMemcpyDeviceToHost);
    bool all_zero = true;
    for (size_t j = 0; j < 64 && all_zero; ++j) {
      if (host_buf[j] != 0) all_zero = false;
    }
    if (all_zero) continue;

    size_t mismatch = VerifyBlockPattern(host_buf.data(), seq_id, bi);
    if (mismatch == 0) {
      ++result.verify_ok;
    } else {
      ++result.verify_fail;
      if (result.verify_fail <= 3) {
        fprintf(stderr, "    VERIFY FAIL: block %d, %zu/%zu bytes mismatch\n", bi, mismatch, kBlockBytes);
      }
    }
  }
  return result;
}

// ===== Signal via PutFirstTokens / QueryFirstTokens =====
static void PutSignal(GlobalCacheConnectorInterface* conn, int signal_seq, const SignalPayload& payload) {
  auto key = MakeSignalKey(signal_seq);
  auto tokens = EncodePayload(payload);
  Status s = conn->PutFirstTokens(key, tokens);
  if (!s.OK()) {
    fprintf(stderr, "Signal #%d PUT FAILED: %s\n", signal_seq, s.GetMessage().c_str());
  }
}

static bool WaitForSignal(GlobalCacheConnectorInterface* conn, int signal_seq, SignalPayload& out,
                          int timeout_sec = 600) {
  auto key = MakeSignalKey(signal_seq);
  auto start = std::chrono::steady_clock::now();

  while (true) {
    std::vector<int> tokens;
    bool found = false;
    conn->QueryFirstTokens(key, tokens, found);

    if (found && tokens.size() >= 5) {
      out = DecodePayload(tokens);
      return true;
    }

    auto elapsed = std::chrono::steady_clock::now() - start;
    if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= timeout_sec) {
      fprintf(stderr, "Timeout waiting for signal #%d\n", signal_seq);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}

// ===== Warmup =====
static void DoWarmup(GlobalCacheConnectorInterface* conn, int tp_size, int tp_rank, GpuBlocks& gpu) {
  fprintf(stderr, "Warming up...\n");
  double t0 = NowMs();
  std::vector<int> tokens(kBlockSize);
  for (int i = 0; i < kBlockSize; ++i) tokens[i] = -(i + 1);
  std::vector<int> ranges = {0, kBlockSize - 1};
  std::vector<void*> ptrs = {gpu.ptrs[0]};
  conn->PutGlobalCacheBlocks(tokens, tp_size, tp_rank, 0, kBlockSize, kBlockBytes, ranges, ptrs);
  std::vector<int> matched;
  conn->QueryAndFetch(tokens, 0, tp_size, tp_rank, 0, kBlockSize, kBlockBytes, ptrs, matched);
  fprintf(stderr, "Warmup done (%.0f ms)\n", NowMs() - t0);
}

// ===== main =====
int main(int argc, char* argv[]) {
  std::string config_path, mode;
  std::string token_str = "8192", iter_str = "3";
  int tp_size = 1, device_id = 0;
  bool verbose = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    }
    if (arg == "--verbose") {
      verbose = true;
      continue;
    }
    if (i + 1 >= argc) {
      fprintf(stderr, "Missing value for: %s\n", arg.c_str());
      return 1;
    }
    std::string val = argv[++i];
    if (arg == "--config") {
      config_path = val;
    } else if (arg == "--mode") {
      mode = val;
    } else if (arg == "--num_tokens") {
      token_str = val;
    } else if (arg == "--iterations") {
      iter_str = val;
    } else if (arg == "--tp_size") {
      tp_size = std::stoi(val);
    } else if (arg == "--device_id") {
      device_id = std::stoi(val);
    } else {
      fprintf(stderr, "Unknown: %s\n", arg.c_str());
      return 1;
    }
  }

  if (config_path.empty() || (mode != "writer" && mode != "reader")) {
    PrintUsage(argv[0]);
    return 1;
  }

  // 解析列表参数
  auto token_list = ParseIntList(token_str);
  auto iter_list = ParseIntList(iter_str);
  auto rounds = ZipRounds(token_list, iter_list);

  // 验证参数
  for (auto& r : rounds) {
    if (r.num_tokens <= 0 || r.num_tokens % kBlockSize != 0) {
      fprintf(stderr, "Error: num_tokens %d must be positive multiple of %d\n", r.num_tokens, kBlockSize);
      return 1;
    }
  }

  // 用最大 num_tokens 初始化
  int max_tokens = 0;
  int total_iters = 0;
  for (auto& r : rounds) {
    max_tokens = std::max(max_tokens, r.num_tokens);
    total_iters += r.iterations;
  }
  g_max_tokens = max_tokens;

  // YAML
  YAML::Node root;
  try {
    root = YAML::LoadFile(config_path);
  } catch (const YAML::Exception& e) {
    fprintf(stderr, "Failed to load config: %s\n", e.what());
    return 1;
  }

  auto cfg = ParseConnectorConfig(root, max_tokens);
  cfg.device_id = device_id;
  cfg.device_count = tp_size;

  if (cfg.service_endpoint.empty() && cfg.backend_type == "mooncake-mock") {
    cfg.service_endpoint = "/tmp/global_cache_bench_metadata";
    system(("mkdir -p " + cfg.service_endpoint).c_str());
  }
  if (cfg.service_endpoint.empty()) {
    fprintf(stderr, "Error: service_endpoint is required\n");
    return 1;
  }

  int max_blocks = max_tokens / kBlockSize;
  int tp_rank = 0;

  fprintf(stderr, "=== Global Cache Bench (DeepSeek-R1 W4A8) ===\n");
  fprintf(stderr, "Mode:       %s\n", mode.c_str());
  fprintf(stderr, "Backend:    %s  Protocol: %s\n", cfg.backend_type.c_str(), cfg.protocol.c_str());
  fprintf(stderr, "Endpoint:   %s\n", cfg.service_endpoint.c_str());
  fprintf(stderr, "Master:     %s\n", cfg.master_server_addr.c_str());
  fprintf(stderr, "Rounds:     %zu (%d total iterations)\n", rounds.size(), total_iters);
  for (size_t i = 0; i < rounds.size(); ++i) {
    int nb = rounds[i].num_tokens / kBlockSize;
    size_t data = static_cast<size_t>(nb) * kBlockBytes;
    fprintf(stderr, "  [%zu] %d tokens (%d blocks, %.2f MB) x %d iters\n", i + 1, rounds[i].num_tokens, nb,
            data / (1024.0 * 1024.0), rounds[i].iterations);
  }
  fprintf(stderr, "Block:      %zu bytes (%.2f MB)\n", kBlockBytes, kBlockBytes / (1024.0 * 1024.0));
  fprintf(stderr, "TP: %d  Device: %d\n", tp_size, device_id);

  if (mode == "writer") {
    g_run_id = static_cast<int32_t>((static_cast<uint32_t>(getpid()) * 100003u) & 0x7FFFFFFF);
    fprintf(stderr, "Run ID:     %d (pid=%d)\n", g_run_id, getpid());
  }
  fprintf(stderr, "\n");

  // CUDA
  if (cudaSetDevice(device_id) != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice(%d) failed\n", device_id);
    return 1;
  }

  // Connector
  auto connector = GlobalCacheConnectorFactory::CreateConnector(cfg);
  if (!connector) {
    fprintf(stderr, "Failed to create connector\n");
    return 1;
  }

  // GPU（按最大 block 数分配）
  GpuBlocks gpu;
  if (!gpu.Alloc(device_id, max_blocks)) {
    fprintf(stderr, "Failed to allocate GPU\n");
    return 1;
  }
  fprintf(stderr, "GPU: %d blocks (%.2f MB)\n", max_blocks,
          static_cast<size_t>(max_blocks) * kBlockBytes / (1024.0 * 1024.0));

  // Warmup
  DoWarmup(connector.get(), tp_size, tp_rank, gpu);

  int signal_seq = 0;
  int global_seq = 0;  // 全局递增的 seq_id，避免跨 round key 冲突

  if (mode == "writer") {
    // ===== Writer: 逐 iteration 写入 + 信号入队，最后等 reader done =====
    for (size_t r = 0; r < rounds.size(); ++r) {
      int nt = rounds[r].num_tokens;
      int nb = nt / kBlockSize;
      int iters = rounds[r].iterations;
      size_t round_data = static_cast<size_t>(nb) * kBlockBytes;
      LatencyStats stats;

      fprintf(stderr, "\n=== Round %zu/%zu: WRITE %d tokens (%d blocks, %.2f MB) x %d iters ===\n", r + 1,
              rounds.size(), nt, nb, round_data / (1024.0 * 1024.0), iters);

      for (int iter = 0; iter < iters; ++iter) {
        double ms = DoWriteOnce(connector.get(), tp_size, tp_rank, nt, nb, global_seq, gpu);
        if (ms < 0) {
          ++global_seq;
          continue;
        }
        stats.Add(ms);
        double bw = (round_data / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
        fprintf(stderr, "  iter %d: %.3f ms  %.2f GB/s\n", iter, ms, bw);

        // 每次写完立即通知 reader
        SignalPayload sig{global_seq, nt, iters, g_run_id, g_max_tokens};
        PutSignal(connector.get(), signal_seq, sig);
        ++signal_seq;
        ++global_seq;
      }

      fprintf(stderr, "--- Round %zu Write Summary ---\n", r + 1);
      stats.Print("PUT", round_data);
    }

    // 等 reader 最终 done 信号（保持 RDMA 段存活）
    fprintf(stderr, "\nWriter waiting for reader done (signal #%d, timeout 600s)...\n", signal_seq);
    SignalPayload done_signal{};
    if (WaitForSignal(connector.get(), signal_seq, done_signal, 600)) {
      fprintf(stderr, "Reader done signal received.\n");
    } else {
      fprintf(stderr, "Timeout waiting for reader done signal.\n");
    }
    fprintf(stderr, "\nWriter done.\n");

  } else {
    // ===== Reader: 信号驱动，逐 iteration 读取 =====
    int prev_nt = 0;
    LatencyStats stats;
    int round_idx = 0;

    for (;;) {
      SignalPayload payload{};
      fprintf(stderr, "\nWaiting for signal #%d...\n", signal_seq);
      if (!WaitForSignal(connector.get(), signal_seq, payload)) {
        break;  // timeout -> writer 已结束
      }
      ++signal_seq;

      g_run_id = payload.run_id;
      g_max_tokens = payload.max_tokens;
      int nt = payload.num_tokens;
      int nb = nt / kBlockSize;
      int seq_id = payload.seq_id;
      size_t round_data = static_cast<size_t>(nb) * kBlockBytes;

      // Round 切换检测
      if (nt != prev_nt) {
        if (prev_nt > 0) {
          int prev_nb = prev_nt / kBlockSize;
          fprintf(stderr, "--- Round %d Read Summary ---\n", round_idx);
          stats.Print("FETCH", static_cast<size_t>(prev_nb) * kBlockBytes);
          stats.Clear();
        }
        ++round_idx;
        fprintf(stderr, "\n=== Round %d: READ %d tokens (%d blocks, %.2f MB) ===\n", round_idx, nt, nb,
                round_data / (1024.0 * 1024.0));
        prev_nt = nt;

        // Resize GPU if needed
        if (nb > gpu.current_blocks) {
          fprintf(stderr, "Resizing GPU: %d -> %d blocks\n", gpu.current_blocks, nb);
          gpu.Alloc(device_id, nb);
        }
      }

      auto result = DoReadOnce(connector.get(), tp_size, tp_rank, nt, nb, seq_id, gpu);
      if (result.latency_ms < 0) continue;

      stats.Add(result.latency_ms);
      size_t matched_data = static_cast<size_t>(result.matched_blocks) * kBlockBytes;
      double bw = (matched_data / (1024.0 * 1024.0 * 1024.0)) / (result.latency_ms / 1000.0);
      fprintf(stderr,
              "  iter %d: %.3f ms  matched=%d/%d blocks (%d ranges, %.2f MB)  %.2f GB/s  verify=%d ok, %d fail\n",
              seq_id, result.latency_ms, result.matched_blocks, nb, result.num_ranges, matched_data / (1024.0 * 1024.0),
              bw, result.verify_ok, result.verify_fail);
    }

    // 最后一轮 summary
    if (prev_nt > 0) {
      int prev_nb = prev_nt / kBlockSize;
      fprintf(stderr, "--- Round %d Read Summary ---\n", round_idx);
      stats.Print("FETCH", static_cast<size_t>(prev_nb) * kBlockBytes);
    }

    // 通知 writer 读取全部完成
    SignalPayload done_sig{0, 0, 0, g_run_id, g_max_tokens};
    PutSignal(connector.get(), signal_seq, done_sig);
    fprintf(stderr, "\nReader done signal #%d sent.\n", signal_seq);
  }

  gpu.Free();
  connector.reset();
  return 0;
}
