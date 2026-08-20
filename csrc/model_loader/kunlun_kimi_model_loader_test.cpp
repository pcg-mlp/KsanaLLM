/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#ifdef ENABLE_KUNLUN

#  include <gtest/gtest.h>

#  include <pybind11/embed.h>

#  include <unistd.h>  // getpid
#  include <filesystem>
#  include <fstream>
#  include <memory>
#  include <string>
#  include <vector>

#  include "fmt/core.h"
#  include "model_loader/config_parser/model_config_parser.h"
#  include "model_loader/model_loader_utils.h"
#  include "model_loader/weight_loader/model_weight_loader.h"
#  include "nlohmann/json.hpp"

#  include "models/deepseek_v3/deepseek_v3_config.h"

#  include "configure/environment.h"
#  include "configure/global_context.h"
#  include "device/common_device.h"
#  include "device/device_context.h"
#  include "device/device_types.h"
#  include "include/gtest/gtest.h"
#  include "test.h"
#  include "utils/logger.h"
#  include "utils/singleton.h"

#  include "helpers/environment_test_helper.h"

using namespace ksana_llm;

static const char* kKimiModelDir = "/data1/klx/Kimi-K2.6-klx-w4a8-gptq";

// Absolute directory containing the running test binary, e.g.
// .../builds/<COMMIT_ID>/build_kunlun/bin. Used as the tensor-dump root instead
// of the current working directory: the pipeline runs from a shared cwd
// (/workspace/KsanaLLM), so a cwd-anchored dump lets concurrent MRs overwrite
// each other. The binary path embeds the per-commit build dir, so anchoring here
// isolates every build and guarantees a writable location (we just linked into it).
//
// /proc/self/exe 是符号链接，用 read_symlink 而非 canonical：read_symlink 不做
// stat 确认存在、不逐段解析，语义更贴合且在 /proc 受限的加固容器中更不易失败。
// 若仍失败（/proc 未挂载或非 Linux），回退到 __FILE__ 的父目录作为兜底——
// 后者是编译期绝对路径，运行时一定可解析。
static std::filesystem::path GetExecutableDir() {
  try {
    return std::filesystem::read_symlink("/proc/self/exe").parent_path();
  } catch (const std::filesystem::filesystem_error& e) {
    KLLM_LOG_WARNING << "read_symlink(\"/proc/self/exe\") failed: " << e.what()
                     << ", falling back to __FILE__ parent dir" << std::endl;
    return std::filesystem::path(__FILE__).parent_path();
  }
}

// Read `first_k_dense_replace` from the model's config.json. This value is the
// boundary between dense MLP layers [0, first_k_dense_replace) and MoE layers
// [first_k_dense_replace, num_layer). Returns -1 if it cannot be determined.
static int ReadFirstKDenseReplace(const std::string& model_dir) {
  std::filesystem::path config_file = std::filesystem::path(model_dir) / "config.json";
  std::ifstream ifs(config_file);
  if (!ifs.is_open()) {
    KLLM_LOG_WARNING << "Failed to open config.json at " << config_file << std::endl;
    return -1;
  }
  nlohmann::json config_json;
  try {
    ifs >> config_json;
  } catch (const std::exception& e) {
    KLLM_LOG_WARNING << "Failed to parse config.json: " << e.what() << std::endl;
    return -1;
  }
  // Multi-modal models nest the language model config under "text_config".
  if (config_json.contains("text_config")) {
    config_json = config_json["text_config"];
  }
  if (!config_json.contains("first_k_dense_replace")) {
    return -1;
  }
  return config_json["first_k_dense_replace"].get<int>();
}

// Resolve the expected on-device DataType for a loaded weight. The npy dump +
// Python checker normalize everything to fp32, so a dtype regression is invisible
// there; this mirrors the loader's intended output dtype so it can be asserted in
// C++ where the Tensor still carries its DataType.
//   - expert_map (EP routing table, not a model weight)  : INT32
//   - *.weight_scale and gate.e_score_correction_bias   : FP32 (cast in loader)
//   - norm / layernorm weights and the MoE router gate  : BF16 (kept 16-bit)
//   - embed_tokens / lm_head                            : BF16
//   - all other quantized *.weight                      : INT8 (W4A8 packed)
static DataType GetExpectedWeightDtype(const std::string& name) {
  auto contains = [&](const char* s) { return name.find(s) != std::string::npos; };
  auto ends_with = [&](const std::string& suffix) {
    return name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
  };

  // EP routing map; auxiliary tensor, always INT32 (see DeepSeekV3WeightLoader::InitExpertMap).
  if (name == "expert_map") {
    return DataType::TYPE_INT32;
  }
  if (ends_with(".weight_scale") || contains("e_score_correction_bias")) {
    return DataType::TYPE_FP32;
  }
  // Covers input_layernorm / post_attention_layernorm / q_a_layernorm /
  // kv_a_layernorm / model.norm.
  if (contains("norm")) {
    return DataType::TYPE_BF16;
  }
  // MoE router gate weight (not the fused gate_up_proj).
  if (ends_with("mlp.gate.weight")) {
    return DataType::TYPE_BF16;
  }
  if (contains("embed_tokens") || contains("lm_head")) {
    return DataType::TYPE_BF16;
  }
  return DataType::TYPE_INT8;
}

class KunlunKimiLoaderTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    pybind11::initialize_interpreter();

    // 生成一个临时 yaml：关键并行参数（TP/EP 等）直接内联可见，model_dir 指向真实的
    // Kimi-K2.6 模型目录，由后续流程去读该目录下真实的 config.json（不合成/不修改 config.json）。
    // 这样不再依赖 examples/ksana_llm_kimi_k26_kunlun.yaml 这个外部文件。TP/EP 设为 2（2 卡分片），
    // 单卡加载 rank 0 的权重分片与校验依赖这些并行参数，配套的 check 脚本也会以 tp=2/ep=2 传入。
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    // 路径中嵌入 PID 做隔离，避免同机并发 CI 进程互相覆盖/删除临时 yaml。
    const std::string pid_suffix = std::to_string(getpid());
    temp_yaml_path_ = (temp_dir / ("ksana_kunlun_kimi_loader_test_" + pid_suffix + ".yaml")).string();
    {
      std::ofstream yaml(temp_yaml_path_);
      yaml << "setting:\n";
      yaml << "  global:\n";
      yaml << "    tensor_para_size: 2\n";
      yaml << "    pipeline_para_size: 1\n";
      yaml << "    attn_data_para_size: 1\n";
      yaml << "    expert_world_size: 1\n";
      yaml << "    expert_para_size: 2\n";
      yaml << "    enable_cuda_graph: true\n";
      yaml << "  block_manager:\n";
      yaml << "    block_token_num: 64\n";
      yaml << "    reserved_device_memory_ratio: 0.05\n";
      yaml << "    block_host_memory_factor: 0.0\n";
      yaml << "  attn_backend:\n";
      yaml << "    enable_blocked_multi_token_forwarding_kv: false\n";
      yaml << "model_spec:\n";
      yaml << "  base_model:\n";
      yaml << "    model_dir: \"" << kKimiModelDir << "\"\n";
    }

    Singleton<Environment>::GetInstance()->ParseConfig(temp_yaml_path_);
    RuntimeConfig runtime_config;
    Singleton<Environment>::GetInstance()->GetRuntimeConfig(runtime_config);
    parallel_basic_config_ = runtime_config.parallel_basic_config;

    // Single-card test: load weights for rank 0 with TP/EP from yaml config.
    auto env = Singleton<Environment>::GetInstance();
    context_ = std::make_shared<Context>();
    context_->global.reset(new GlobalContext(parallel_basic_config_.tensor_parallel_size,
                                             parallel_basic_config_.attn_data_parallel_size, 1, env));
    context_->global->InitDistributedConfig();
    context_->device = std::make_shared<DeviceContext>(1);
    KLLM_LOG_INFO << "Initialize Kunlun Kimi context success, TP=" << parallel_basic_config_.tensor_parallel_size
                  << " EP=" << parallel_basic_config_.expert_parallel_size << std::endl;

    // Snapshot the global PipelineConfig so TearDown can restore it. The weight-load
    // test narrows lower/upper_layer_idx on the process-wide Singleton<Environment>;
    // without this, that mutation would leak into any later test that reads the
    // singleton, making them order-dependent.
    Singleton<Environment>::GetInstance()->GetPipelineConfig(saved_pipeline_config_);
  }

  void TearDown() override {
    // Restore the PipelineConfig mutated by the test bodies before tearing down.
    Singleton<Environment>::GetInstance()->SetPipelineConfig(saved_pipeline_config_);
    pybind11::finalize_interpreter();

    // 清理临时 yaml（best-effort）。
    try {
      if (!temp_yaml_path_.empty() && std::filesystem::exists(temp_yaml_path_)) {
        std::filesystem::remove(temp_yaml_path_);
      }
    } catch (const std::exception&) {
      // best-effort cleanup
    }
  }

 protected:
  std::shared_ptr<Context> context_ = nullptr;
  ParallelismBasicConfig parallel_basic_config_;
  PipelineConfig saved_pipeline_config_;
  std::string temp_yaml_path_;
};

TEST_F(KunlunKimiLoaderTest, TestKimiK26ConfigParser) {
  const std::string& model_dir = kKimiModelDir;
  if (!std::filesystem::is_directory(model_dir)) {
    KLLM_LOG_WARNING << "Model dir " << model_dir << " not exist, skip." << std::endl;
    GTEST_SKIP() << "Model directory not found: " << model_dir;
  }

  ModelConfigParser model_config_parser;
  std::shared_ptr<BaseModelConfig> model_config;
  Status status = model_config_parser.ParseModelConfig(model_dir, parallel_basic_config_, model_config);
  EXPECT_TRUE(status.OK());
  EXPECT_TRUE(model_config);

  // Check real type.
  EXPECT_TRUE(typeid(DeepSeekV3Config) == typeid(*model_config));

  std::shared_ptr<DeepSeekV3Config> deepseek_v3_config = std::dynamic_pointer_cast<DeepSeekV3Config>(model_config);

  // Verify key config fields.
  EXPECT_EQ(deepseek_v3_config->model_dir, model_dir);
  EXPECT_GT(deepseek_v3_config->num_layer, 0);
  EXPECT_GT(deepseek_v3_config->hidden_units, 0);
  EXPECT_GT(deepseek_v3_config->vocab_size, 0);
  EXPECT_GT(deepseek_v3_config->num_key_value_heads, 0);
  EXPECT_GT(deepseek_v3_config->mla_config.q_lora_rank, 0);
  EXPECT_GT(deepseek_v3_config->mla_config.kv_lora_rank, 0);

  KLLM_LOG_INFO << "Kimi K2.6 config: num_layer=" << deepseek_v3_config->num_layer
                << " hidden_units=" << deepseek_v3_config->hidden_units
                << " vocab_size=" << deepseek_v3_config->vocab_size
                << " q_lora_rank=" << deepseek_v3_config->mla_config.q_lora_rank
                << " kv_lora_rank=" << deepseek_v3_config->mla_config.kv_lora_rank;
}

TEST_F(KunlunKimiLoaderTest, TestKimiK26WeightLoaderAndVerify) {
  const std::string& model_dir = kKimiModelDir;
  if (!std::filesystem::is_directory(model_dir)) {
    KLLM_LOG_WARNING << "Model dir " << model_dir << " not exist, skip." << std::endl;
    GTEST_SKIP() << "Model directory not found: " << model_dir;
  }

  ModelConfigParser model_config_parser;
  std::shared_ptr<BaseModelConfig> model_config;
  Status status = model_config_parser.ParseModelConfig(model_dir, parallel_basic_config_, model_config);
  EXPECT_TRUE(status.OK());
  EXPECT_TRUE(model_config);

  // Read the dense/MoE boundary from config.json and restrict weight loading to
  // just the last dense MLP layer (first_k_dense_replace - 1) and the first MoE
  // layer (first_k_dense_replace). Only these two layers are loaded and verified.
  int first_k_dense_replace = ReadFirstKDenseReplace(model_dir);
  ASSERT_GT(first_k_dense_replace, 0)
      << "first_k_dense_replace must be > 0 to have both a dense MLP layer and a MoE layer to test.";
  const int16_t lower_layer_idx = static_cast<int16_t>(first_k_dense_replace - 1);  // last dense MLP layer
  const int16_t upper_layer_idx = static_cast<int16_t>(first_k_dense_replace);      // first MoE layer

  auto env = Singleton<Environment>::GetInstance();
  PipelineConfig pipeline_config;
  env->GetPipelineConfig(pipeline_config);
  pipeline_config.lower_layer_idx = lower_layer_idx;
  pipeline_config.upper_layer_idx = upper_layer_idx;
  env->SetPipelineConfig(pipeline_config);
  KLLM_LOG_INFO << "Kimi K2.6 partial load: first_k_dense_replace=" << first_k_dense_replace
                << ", loading only layer range [" << lower_layer_idx << ", " << upper_layer_idx
                << "] (last MLP layer + first MoE layer)." << std::endl;

  ModelWeightLoader weight_loader(env, context_);
  std::shared_ptr<ModelWeight> dev_weight = std::make_shared<ModelWeight>();
  status = weight_loader.LoadWeight(model_config, dev_weight);
  EXPECT_TRUE(status.OK());
  EXPECT_TRUE(dev_weight);

  // Verify on-device dtype of every loaded weight. The npy dump collapses bf16 to
  // uint16 and the Python checker re-expands both sides to fp32, so it is dtype-blind
  // by construction; the Tensor here still carries its real DataType, so assert it.
  // (Shape/value are validated by check_kunlun_kimi_w4a8_weight_tensor.py.)
  for (const std::string& weight_name : dev_weight->GetWeightNames()) {
    const Tensor& weight_tensor = dev_weight->GetWeightTensor(weight_name);
    DataType expected_dtype = GetExpectedWeightDtype(weight_name);
    EXPECT_EQ(weight_tensor.dtype, expected_dtype)
        << "Weight " << weight_name << " has dtype " << GetTypeString(weight_tensor.dtype) << ", expected "
        << GetTypeString(expected_dtype);
  }

  // Dump every tensor to npy files, under the test binary's own directory so
  // concurrent pipeline builds (unique per COMMIT_ID) never collide.
  std::filesystem::path dump_path = GetExecutableDir();
  dump_path /= std::filesystem::path(Str2Vector(model_dir, "/").back()).filename();

  // Remove dump path and create it fresh.
  std::filesystem::remove_all(dump_path);
  std::filesystem::create_directories(dump_path);

  {
    std::filesystem::path rank_dump_path = dump_path;
    rank_dump_path /= std::to_string(0);  // single rank
    std::filesystem::create_directories(rank_dump_path);

    for (const std::string& weight_name : dev_weight->GetWeightNames()) {
      Tensor weight_tensor = dev_weight->GetWeightTensor(weight_name);

      std::filesystem::path tensor_data_file = rank_dump_path;
      tensor_data_file /= weight_name + ".npy";
      weight_tensor.SaveToNpyFile(tensor_data_file);
    }
  }

  // Locate the checker script relative to THIS source file, not the current
  // working directory. The pipeline runs the test binary from an unrelated cwd
  // (e.g. the container's default /workspace/KsanaLLM), so a cwd-relative path
  // resolves to a non-existent tree. __FILE__ is the compile-time absolute path
  // of this .cpp, and the checker lives in the same directory as it.
  std::filesystem::path checker_script =
      std::filesystem::path(__FILE__).parent_path() / "check_kunlun_kimi_w4a8_weight_tensor.py";
  // Pass dump_path explicitly (4th arg) so the checker does not fall back to its
  // own getcwd(), which may differ from where we dumped the tensors above.
  //
  // 路径用单引号包裹：std::system 经 /bin/sh -c 执行，未加引号的路径若含空格或
  // shell 元字符会被错误分词。单引号在 shell 中禁用所有特殊字符的解释，最安全。
  std::string command = fmt::format("python '{}' {} {} {} '{}'", checker_script.string(), model_dir,
                                    parallel_basic_config_.tensor_parallel_size,
                                    parallel_basic_config_.expert_parallel_size, dump_path.string());
  int ret_code = std::system(command.c_str());
  // std::system 返回 -1 表示 fork/exec 失败，此时 ret_code 不是 wait 状态值，
  // 对其取 WEXITSTATUS 是未定义行为。先断言非 -1，再解析退出码。
  ASSERT_NE(ret_code, -1) << "failed to launch checker script: " << command;
  EXPECT_EQ(WEXITSTATUS(ret_code), 0);

  // Check success file.
  std::filesystem::path succ_file_path = dump_path;
  succ_file_path /= "SUCCESS";
  EXPECT_TRUE(std::filesystem::is_regular_file(succ_file_path));
}

#else  // !ENABLE_KUNLUN

// Non-Kunlun build: provide a placeholder to keep the test target well-formed.
#  include <gtest/gtest.h>
TEST(KunlunKimiLoaderTest, Placeholder) { GTEST_SKIP() << "Kunlun Kimi loader test requires ENABLE_KUNLUN"; }

#endif  // ENABLE_KUNLUN
