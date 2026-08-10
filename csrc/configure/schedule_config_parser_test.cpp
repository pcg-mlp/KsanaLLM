/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "configure/schedule_config_parser.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "cache_manager/cache_layout/mamba_cache_layout.h"
#include "configure/yaml_reader.h"
#include "gtest/gtest.h"
#include "loguru.hpp"

namespace ksana_llm {

namespace {
class MaxLogVerbosityEnv : public ::testing::Environment {
 public:
  void SetUp() override { loguru::g_stderr_verbosity = loguru::Verbosity_MAX; }
};
static auto *unused_log_env = ::testing::AddGlobalTestEnvironment(new MaxLogVerbosityEnv());
}  // namespace

class ScheduleConfigParserTest : public ::testing::Test {
 protected:
  void SetUp() override { parser_ = std::make_shared<ScheduleConfigParser>(); }

  void TearDown() override {
    parser_->Reset();
    // Clean up temporary files
    CleanupTempFiles();
  }

  std::string CreateConnectorYamlConfig(const std::string &group_role = "none", const std::string &router_addr = "",
                                        const std::string &communication_type = "",
                                        const std::string &inference_addr = "",
                                        const std::string &coordinator_addr = "",
                                        const std::string &cluster_name = "") {
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    std::string filename = "schedule_config_test_" + std::to_string(test_file_counter_++) + ".yaml";
    std::filesystem::path temp_file_path = temp_dir / filename;
    temp_files_.push_back(temp_file_path.string());

    std::filesystem::path model_dir = temp_dir / ("test_model_dir_" + std::to_string(test_file_counter_));
    temp_dirs_.push_back(model_dir.string());
    if (std::filesystem::exists(model_dir)) {
      std::filesystem::remove_all(model_dir);
    }
    std::filesystem::create_directory(model_dir);

    std::ofstream file(temp_file_path);
    if (!file.is_open()) return "";

    file << "setting:\n";
    file << "  global:\n";
    file << "    tensor_para_size: 1\n";
    file << "  batch_scheduler:\n";
    file << "    max_token_len: 4096\n";
    file << "    max_step_tokens: 4096\n";
    file << "    max_batch_size: 32\n";
    file << "  block_manager:\n";
    file << "    block_token_num: 16\n";
    file << "    reserved_device_memory_ratio: 0.01\n";
    file << "  connector:\n";
    file << "    group_role: " << group_role << "\n";
    if (!router_addr.empty()) {
      file << "    router_addr: \"" << router_addr << "\"\n";
    }
    if (!communication_type.empty()) {
      file << "    communication_type: " << communication_type << "\n";
    }
    if (!inference_addr.empty()) {
      file << "    inference_addr: \"" << inference_addr << "\"\n";
    }
    if (!coordinator_addr.empty()) {
      file << "    coordinator_addr: \"" << coordinator_addr << "\"\n";
    }
    if (!cluster_name.empty()) {
      file << "    cluster_name: " << cluster_name << "\n";
    }
    file << "model_spec:\n";
    file << "  base_model:\n";
    file << "    model_dir: \"" << model_dir.string() << "\"\n";
    file.close();
    return temp_file_path.string();
  }

  // Helper function to create a temporary YAML configuration file with an
  // optional swa_block_num_ratio under setting.cache_manager.
  // Pass swa_block_num_ratio < 0 to omit the key entirely.
  std::string CreateTempYamlConfigWithSwa(float swa_block_num_ratio = -1.0f,
                                          bool enable_non_contiguous_swa_mode = false,
                                          bool has_enable_non_contiguous_swa_mode = false,
                                          float mamba_block_num_ratio = -1.0f) {
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    std::string filename = "schedule_config_test_" + std::to_string(test_file_counter_++) + ".yaml";
    std::filesystem::path temp_file_path = temp_dir / filename;
    temp_files_.push_back(temp_file_path.string());

    std::filesystem::path model_dir = temp_dir / ("test_model_dir_" + std::to_string(test_file_counter_));
    temp_dirs_.push_back(model_dir.string());
    if (std::filesystem::exists(model_dir)) {
      std::filesystem::remove_all(model_dir);
    }
    std::filesystem::create_directory(model_dir);

    std::ofstream file(temp_file_path);
    if (!file.is_open()) {
      return "";
    }

    file << "setting:\n";
    file << "  global:\n";
    file << "    tensor_para_size: 1\n";
    file << "  batch_scheduler:\n";
    file << "    max_token_len: 4096\n";
    file << "    max_step_tokens: 4096\n";
    file << "    max_batch_size: 32\n";
    file << "  block_manager:\n";
    file << "    block_token_num: 16\n";
    file << "    reserved_device_memory_ratio: 0.01\n";
    if (swa_block_num_ratio >= 0.0f || has_enable_non_contiguous_swa_mode || mamba_block_num_ratio >= 0.0f) {
      file << "  cache_manager:\n";
      if (swa_block_num_ratio >= 0.0f) {
        file << "    swa_block_num_ratio: " << swa_block_num_ratio << "\n";
      }
      if (has_enable_non_contiguous_swa_mode) {
        file << "    enable_non_contiguous_swa_mode: " << (enable_non_contiguous_swa_mode ? "true" : "false") << "\n";
      }
      if (mamba_block_num_ratio >= 0.0f) {
        file << "    mamba_block_num_ratio: " << mamba_block_num_ratio << "\n";
      }
    }
    file << "model_spec:\n";
    file << "  base_model:\n";
    file << "    model_dir: \"" << model_dir.string() << "\"\n";

    file.close();
    return temp_file_path.string();
  }

  // Helper function to create a temporary YAML configuration file
  std::string CreateTempYamlConfig(bool enable_prefix_caching = false, size_t split_fuse_token_num = 0,
                                   bool use_flashinfer_for_decode = false, size_t max_chunked_kv_tokens = 0,
                                   size_t max_step_tokens = 4096,
                                   bool enable_blocked_multi_token_forwarding_kv = false) {
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    std::string filename = "schedule_config_test_" + std::to_string(test_file_counter_++) + ".yaml";
    std::filesystem::path temp_file_path = temp_dir / filename;
    temp_files_.push_back(temp_file_path.string());

    // Create model directory
    std::filesystem::path model_dir = temp_dir / ("test_model_dir_" + std::to_string(test_file_counter_));
    temp_dirs_.push_back(model_dir.string());

    if (std::filesystem::exists(model_dir)) {
      std::filesystem::remove_all(model_dir);
    }
    std::filesystem::create_directory(model_dir);

    std::ofstream file(temp_file_path);
    if (!file.is_open()) {
      return "";
    }

    // Write YAML configuration
    file << "setting:\n";
    file << "  global:\n";
    file << "    tensor_para_size: 1\n";
    file << "    pipeline_para_size: 1\n";
    file << "  batch_scheduler:\n";
    file << "    max_waiting_queue_len: 100\n";
    file << "    max_token_len: 4096\n";
    file << "    max_step_tokens: " << max_step_tokens << "\n";
    file << "    max_batch_size: 32\n";
    file << "    enable_auto_prefix_cache: " << (enable_prefix_caching ? "true" : "false") << "\n";
    file << "    split_fuse_token_num: " << split_fuse_token_num << "\n";
    if (max_chunked_kv_tokens > 0) {
      file << "    max_chunked_kv_tokens: " << max_chunked_kv_tokens << "\n";
    }
    file << "  block_manager:\n";
    file << "    block_token_num: 16\n";
    file << "    reserved_device_memory_ratio: 0.01\n";
    file << "  attn_backend:\n";
    file << "    use_flashinfer_for_decode: " << (use_flashinfer_for_decode ? "true" : "false") << "\n";
    file << "    enable_blocked_multi_token_forwarding_kv: "
         << (enable_blocked_multi_token_forwarding_kv ? "true" : "false") << "\n";
    file << "model_spec:\n";
    file << "  base_model:\n";
    file << "    model_dir: \"" << model_dir.string() << "\"\n";

    file.close();
    return temp_file_path.string();
  }

  // 生成含 enable_executor_build_ahead / enable_async 的临时 YAML（fast-path 开关解析单测用）。
  std::string CreateTempYamlConfigWithExecutorBuildAhead(bool enable_executor_build_ahead, bool enable_async = false) {
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    std::string filename = "schedule_config_test_" + std::to_string(test_file_counter_++) + ".yaml";
    std::filesystem::path temp_file_path = temp_dir / filename;
    temp_files_.push_back(temp_file_path.string());

    std::filesystem::path model_dir = temp_dir / ("test_model_dir_" + std::to_string(test_file_counter_));
    temp_dirs_.push_back(model_dir.string());
    if (std::filesystem::exists(model_dir)) {
      std::filesystem::remove_all(model_dir);
    }
    std::filesystem::create_directory(model_dir);

    std::ofstream file(temp_file_path);
    if (!file.is_open()) {
      return "";
    }

    file << "setting:\n";
    file << "  global:\n";
    file << "    tensor_para_size: 1\n";
    file << "    pipeline_para_size: 1\n";
    file << "  batch_scheduler:\n";
    file << "    max_waiting_queue_len: 100\n";
    file << "    max_token_len: 4096\n";
    file << "    max_step_tokens: 4096\n";
    file << "    max_batch_size: 32\n";
    file << "    enable_executor_build_ahead: " << (enable_executor_build_ahead ? "true" : "false") << "\n";
    if (enable_async) {
      file << "    enable_async: true\n";
    }
    file << "  block_manager:\n";
    file << "    block_token_num: 16\n";
    file << "    reserved_device_memory_ratio: 0.01\n";
    file << "model_spec:\n";
    file << "  base_model:\n";
    file << "    model_dir: \"" << model_dir.string() << "\"\n";
    file.close();
    return temp_file_path.string();
  }

  // Helper function to load ModelConfig from config.json
  ModelConfig LoadModelConfig(const std::string &model_dir, bool is_visual = false) {
    ModelConfig model_config;

    // Directly set is_visual flag instead of parsing full config
    // This avoids dependencies on model loading infrastructure
    model_config.is_visual = is_visual;

    return model_config;
  }

  void CleanupTempFiles() {
    for (const auto &file : temp_files_) {
      try {
        if (std::filesystem::exists(file)) {
          std::filesystem::remove(file);
        }
      } catch (const std::exception &e) {
        std::cerr << "Failed to remove temp file: " << file << ", error: " << e.what() << std::endl;
      }
    }

    for (const auto &dir : temp_dirs_) {
      try {
        if (std::filesystem::exists(dir)) {
          std::filesystem::remove_all(dir);
        }
      } catch (const std::exception &e) {
        std::cerr << "Failed to remove temp dir: " << dir << ", error: " << e.what() << std::endl;
      }
    }

    temp_files_.clear();
    temp_dirs_.clear();
  }

  std::shared_ptr<ScheduleConfigParser> parser_;
  std::vector<std::string> temp_files_;
  std::vector<std::string> temp_dirs_;
  static int test_file_counter_;
};

int ScheduleConfigParserTest::test_file_counter_ = 0;

// Test: Visual Model automatically disables both PrefixCaching and SplitFuse
TEST_F(ScheduleConfigParserTest, VisualModelDisablesPrefixCachingAndSplitFuse) {
  // Create YAML config with both features enabled
  std::string yaml_path = CreateTempYamlConfig(true, 256);
  ASSERT_FALSE(yaml_path.empty()) << "Failed to create temporary YAML file";

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  std::string model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  ASSERT_FALSE(model_dir.empty());

  // Create visual model config - just set the flag directly
  ModelConfig model_config = LoadModelConfig(model_dir, true);
  EXPECT_TRUE(model_config.is_visual);

  // Parse schedule config
  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  // Verify both features are disabled for visual model
  CacheManagerConfig cache_config;
  parser_->GetCacheManagerConfig(cache_config);
  EXPECT_FALSE(cache_config.enable_prefix_caching) << "PrefixCaching should be disabled (false) for Visual Model";

  BatchSchedulerConfig batch_config;
  parser_->GetBatchSchedulerConfig(batch_config);
  EXPECT_EQ(batch_config.split_fuse_token_num, 0) << "SplitFuse should be disabled (0) for Visual Model";
}

// ================================================================
// DSv4 Chunk Prefill: split_fuse_token_num 必须是 block_token_num (256) 的整数倍
// ================================================================
// 说明:
//   - DSv4 启用 use_mla + use_dsv4_attn 时, block_token_num 在 ParseScheduleConfig
//     里被强制设为 256 (见 schedule_config_parser.cpp 中 use_dsv4_attn 分支)。
//   - 之后我们对 split_fuse_token_num 做整除性校验, 不满足直接抛异常。
//   - 校验仅在 use_dsv4_attn=true 且 split_fuse_token_num>0 时触发。

// 合法: split_fuse_token_num == block_token_num (256), 必须通过。
TEST_F(ScheduleConfigParserTest, DSv4ChunkPrefillAcceptsBlockAligned) {
  std::string yaml_path = CreateTempYamlConfig(/*enable_prefix_caching=*/true, /*split_fuse_token_num=*/256);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  std::string model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  ASSERT_FALSE(model_dir.empty());

  ModelConfig model_config = LoadModelConfig(model_dir, false);
  model_config.use_mla = true;
  model_config.use_dsv4_attn = true;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  BatchSchedulerConfig batch_config;
  parser_->GetBatchSchedulerConfig(batch_config);
  EXPECT_EQ(batch_config.split_fuse_token_num, 256u);
}

// 合法: split_fuse_token_num == 2 * block_token_num (512), 必须通过。
TEST_F(ScheduleConfigParserTest, DSv4ChunkPrefillAcceptsMultiBlock) {
  std::string yaml_path = CreateTempYamlConfig(/*enable_prefix_caching=*/true, /*split_fuse_token_num=*/512);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  std::string model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  ASSERT_FALSE(model_dir.empty());

  ModelConfig model_config = LoadModelConfig(model_dir, false);
  model_config.use_mla = true;
  model_config.use_dsv4_attn = true;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  BatchSchedulerConfig batch_config;
  parser_->GetBatchSchedulerConfig(batch_config);
  EXPECT_EQ(batch_config.split_fuse_token_num, 512u);
}

// 非法: 257 不是 256 的倍数, 必须抛异常。
TEST_F(ScheduleConfigParserTest, DSv4ChunkPrefillRejectsNonAligned257) {
  std::string yaml_path = CreateTempYamlConfig(/*enable_prefix_caching=*/true, /*split_fuse_token_num=*/257);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  std::string model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  ASSERT_FALSE(model_dir.empty());

  ModelConfig model_config = LoadModelConfig(model_dir, false);
  model_config.use_mla = true;
  model_config.use_dsv4_attn = true;

  EXPECT_ANY_THROW(parser_->ParseScheduleConfig(yaml_reader, model_config));
}

// 非法: 100 < 256 且不是其整数倍, 必须抛异常。
TEST_F(ScheduleConfigParserTest, DSv4ChunkPrefillRejectsNonAligned100) {
  std::string yaml_path = CreateTempYamlConfig(/*enable_prefix_caching=*/true, /*split_fuse_token_num=*/100);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  std::string model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  ASSERT_FALSE(model_dir.empty());

  ModelConfig model_config = LoadModelConfig(model_dir, false);
  model_config.use_mla = true;
  model_config.use_dsv4_attn = true;

  EXPECT_ANY_THROW(parser_->ParseScheduleConfig(yaml_reader, model_config));
}

// 边界: split_fuse_token_num == 0 表示禁用 split-fuse, 跳过校验, 不抛。
TEST_F(ScheduleConfigParserTest, DSv4ChunkPrefillSkipsCheckWhenDisabled) {
  std::string yaml_path = CreateTempYamlConfig(/*enable_prefix_caching=*/true, /*split_fuse_token_num=*/0);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  std::string model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  ASSERT_FALSE(model_dir.empty());

  ModelConfig model_config = LoadModelConfig(model_dir, false);
  model_config.use_mla = true;
  model_config.use_dsv4_attn = true;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  BatchSchedulerConfig batch_config;
  parser_->GetBatchSchedulerConfig(batch_config);
  EXPECT_EQ(batch_config.split_fuse_token_num, 0u);
}

// 隔离性: 非 DSv4 模型即便 split_fuse_token_num 非整除 block_token_num 也不应触发该 check。
// (此处 yaml 默认 block_token_num=16, 非 DSv4 不会被覆盖, 257 % 16 != 0 但属于非 DSv4 路径, 不应抛。)
TEST_F(ScheduleConfigParserTest, NonDSv4ChunkPrefillSkipsCheck) {
  std::string yaml_path = CreateTempYamlConfig(/*enable_prefix_caching=*/true, /*split_fuse_token_num=*/257);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  std::string model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  ASSERT_FALSE(model_dir.empty());

  ModelConfig model_config = LoadModelConfig(model_dir, false);
  // 不设置 use_mla / use_dsv4_attn, 保持默认 false。
  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  BatchSchedulerConfig batch_config;
  parser_->GetBatchSchedulerConfig(batch_config);
  EXPECT_EQ(batch_config.split_fuse_token_num, 257u);
}

// Test: UpdateMembers allows k/v scales equal to 1.0 when FlashInfer is enabled.
TEST_F(ScheduleConfigParserTest, UpdateMembersAllowsKvScalesEqualToOneWhenFlashInferEnabled) {
  std::string yaml_path = CreateTempYamlConfig(false, 0, true);
  ASSERT_FALSE(yaml_path.empty()) << "Failed to create temporary YAML file";

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  std::string model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  ASSERT_FALSE(model_dir.empty());

  ModelConfig model_config = LoadModelConfig(model_dir, false);
  model_config.weight_data_type = TYPE_FP16;
  model_config.k_scales = {1.0f, 1.0f};
  model_config.v_scales = {1.0f, 1.0f};

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  ASSERT_TRUE(status.OK()) << status.GetMessage();

  EXPECT_NO_THROW(parser_->UpdateMembers(model_dir, model_config));

  RuntimeConfig runtime_config;
  ASSERT_TRUE(parser_->GetRuntimeConfig(runtime_config).OK());
  EXPECT_TRUE(runtime_config.attn_backend_config.use_flashinfer_for_decode);
  EXPECT_EQ(runtime_config.attn_backend_config.kv_cache_dtype, model_config.weight_data_type);
}

// Test: UpdateMembers rejects k/v scales not equal to 1.0 when FlashInfer is enabled.
TEST_F(ScheduleConfigParserTest, UpdateMembersRejectsKvScalesNotEqualToOneWhenFlashInferEnabled) {
  std::string yaml_path = CreateTempYamlConfig(false, 0, true);
  ASSERT_FALSE(yaml_path.empty()) << "Failed to create temporary YAML file";

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  std::string model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  ASSERT_FALSE(model_dir.empty());

  ModelConfig model_config = LoadModelConfig(model_dir, false);
  model_config.weight_data_type = TYPE_FP16;
  model_config.k_scales = {1.0f, 0.5f};
  model_config.v_scales = {1.0f, 1.0f};

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  ASSERT_TRUE(status.OK()) << status.GetMessage();

  EXPECT_ANY_THROW(parser_->UpdateMembers(model_dir, model_config));
}

// Test: UpdateMembers allows k/v scales not equal to 1.0 when FlashInfer is disabled.
TEST_F(ScheduleConfigParserTest, UpdateMembersAllowsKvScalesNotEqualToOneWhenFlashInferDisabled) {
  std::string yaml_path = CreateTempYamlConfig(false, 0, false);
  ASSERT_FALSE(yaml_path.empty()) << "Failed to create temporary YAML file";

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  std::string model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  ASSERT_FALSE(model_dir.empty());

  ModelConfig model_config = LoadModelConfig(model_dir, false);
  model_config.weight_data_type = TYPE_FP16;
  model_config.k_scales = {1.0f, 0.5f};
  model_config.v_scales = {1.0f, 1.0f};

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  ASSERT_TRUE(status.OK()) << status.GetMessage();

  EXPECT_NO_THROW(parser_->UpdateMembers(model_dir, model_config));

  RuntimeConfig runtime_config;
  ASSERT_TRUE(parser_->GetRuntimeConfig(runtime_config).OK());
  EXPECT_FALSE(runtime_config.attn_backend_config.use_flashinfer_for_decode);
  EXPECT_EQ(runtime_config.attn_backend_config.kv_cache_dtype, model_config.weight_data_type);
}

// --- InitConnectorConfig tests ---

TEST_F(ScheduleConfigParserTest, ConnectorRoleNoneSkipsConnectorParsing) {
  std::string yaml_path = CreateConnectorYamlConfig("none", "localhost:8080");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  Status get_status = parser_->GetConnectorConfigs(connector_config);
  EXPECT_FALSE(get_status.OK()) << "Connector config should not be initialized when role is none";
}

TEST_F(ScheduleConfigParserTest, ConnectorRolePrefillParsesConfig) {
  std::string yaml_path =
      CreateConnectorYamlConfig("prefill", "http://router.example.com:9000", "tcp", "10.0.0.1:8080", "", "my_cluster");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  ASSERT_TRUE(parser_->GetConnectorConfigs(connector_config).OK());
  EXPECT_EQ(connector_config.group_role, GroupRole::PREFILL);
  EXPECT_EQ(connector_config.router_addr, "http://router.example.com:9000");
  EXPECT_EQ(connector_config.inference_addr, "10.0.0.1:8080");
  EXPECT_EQ(connector_config.cluster_name, "my_cluster");
}

TEST_F(ScheduleConfigParserTest, ConnectorRoleDecodeParsesConfig) {
  std::string yaml_path = CreateConnectorYamlConfig("decode", "10.0.0.1:8080");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  ASSERT_TRUE(parser_->GetConnectorConfigs(connector_config).OK());
  EXPECT_EQ(connector_config.group_role, GroupRole::DECODE);
}

TEST_F(ScheduleConfigParserTest, ConnectorRoleBothParsesConfig) {
  std::string yaml_path = CreateConnectorYamlConfig("both", "10.0.0.1:8080");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  ASSERT_TRUE(parser_->GetConnectorConfigs(connector_config).OK());
  EXPECT_EQ(connector_config.group_role, GroupRole::BOTH);
}

TEST_F(ScheduleConfigParserTest, ConnectorUnknownRoleFallsBackToNone) {
  std::string yaml_path = CreateConnectorYamlConfig("invalid_role", "10.0.0.1:8080");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  Status get_status = parser_->GetConnectorConfigs(connector_config);
  EXPECT_FALSE(get_status.OK()) << "Unknown role should fall back to NONE";
}

TEST_F(ScheduleConfigParserTest, ConnectorRoleCaseInsensitive) {
  std::string yaml_path = CreateConnectorYamlConfig("PREFILL", "10.0.0.1:8080");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  ASSERT_TRUE(parser_->GetConnectorConfigs(connector_config).OK());
  EXPECT_EQ(connector_config.group_role, GroupRole::PREFILL);
}

TEST_F(ScheduleConfigParserTest, RouterAddrHostPortGetHttpPrepended) {
  std::string yaml_path = CreateConnectorYamlConfig("prefill", "10.0.0.1:8080");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  ASSERT_TRUE(parser_->GetConnectorConfigs(connector_config).OK());
  EXPECT_EQ(connector_config.router_addr, "http://10.0.0.1:8080");
}

TEST_F(ScheduleConfigParserTest, RouterAddrWithSchemeKeptAsIs) {
  std::string yaml_path = CreateConnectorYamlConfig("prefill", "https://router.example.com:9000");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  ASSERT_TRUE(parser_->GetConnectorConfigs(connector_config).OK());
  EXPECT_EQ(connector_config.router_addr, "https://router.example.com:9000");
}

TEST_F(ScheduleConfigParserTest, RouterAddrEmptyIsAccepted) {
  std::string yaml_path = CreateConnectorYamlConfig("prefill", "");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  Status get_status = parser_->GetConnectorConfigs(connector_config);
  EXPECT_FALSE(get_status.OK()) << "Empty router_addr should not pass GetConnectorConfigs check";
}

TEST_F(ScheduleConfigParserTest, RouterAddrMissingColonWarns) {
  std::string yaml_path = CreateConnectorYamlConfig("prefill", "nocolonhere");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  ASSERT_TRUE(parser_->GetConnectorConfigs(connector_config).OK());
  EXPECT_EQ(connector_config.router_addr, "http://nocolonhere");
}

TEST_F(ScheduleConfigParserTest, RouterAddrColonAtStartWarns) {
  std::string yaml_path = CreateConnectorYamlConfig("prefill", ":8080");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  ASSERT_TRUE(parser_->GetConnectorConfigs(connector_config).OK());
  EXPECT_EQ(connector_config.router_addr, "http://:8080");
}

TEST_F(ScheduleConfigParserTest, RouterAddrColonAtEndWarns) {
  std::string yaml_path = CreateConnectorYamlConfig("prefill", "host:");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  ASSERT_TRUE(parser_->GetConnectorConfigs(connector_config).OK());
  EXPECT_EQ(connector_config.router_addr, "http://host:");
}

TEST_F(ScheduleConfigParserTest, RouterAddrPortOutOfRangeWarns) {
  std::string yaml_path = CreateConnectorYamlConfig("prefill", "host:99999");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  ASSERT_TRUE(parser_->GetConnectorConfigs(connector_config).OK());
  EXPECT_EQ(connector_config.router_addr, "http://host:99999");
}

TEST_F(ScheduleConfigParserTest, RouterAddrPortZeroWarns) {
  std::string yaml_path = CreateConnectorYamlConfig("prefill", "host:0");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  ASSERT_TRUE(parser_->GetConnectorConfigs(connector_config).OK());
  EXPECT_EQ(connector_config.router_addr, "http://host:0");
}

TEST_F(ScheduleConfigParserTest, RouterAddrNonNumericPortWarns) {
  std::string yaml_path = CreateConnectorYamlConfig("prefill", "host:abc");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  ASSERT_TRUE(parser_->GetConnectorConfigs(connector_config).OK());
  EXPECT_EQ(connector_config.router_addr, "http://host:abc");
}

TEST_F(ScheduleConfigParserTest, RouterAddrValidPortInRange) {
  std::string yaml_path = CreateConnectorYamlConfig("prefill", "myhost:443");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  ASSERT_TRUE(parser_->GetConnectorConfigs(connector_config).OK());
  EXPECT_EQ(connector_config.router_addr, "http://myhost:443");
}

TEST_F(ScheduleConfigParserTest, ConnectorCommunicationTypeNccl) {
  std::string yaml_path = CreateConnectorYamlConfig("prefill", "host:8080", "nccl");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  ASSERT_TRUE(parser_->GetConnectorConfigs(connector_config).OK());
  EXPECT_EQ(connector_config.communication_type, CommunicationType::NCCL);
}

TEST_F(ScheduleConfigParserTest, ConnectorCommunicationTypeZmq) {
  std::string yaml_path = CreateConnectorYamlConfig("prefill", "host:8080", "zmq");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  ASSERT_TRUE(parser_->GetConnectorConfigs(connector_config).OK());
  EXPECT_EQ(connector_config.communication_type, CommunicationType::ZMQ);
}

TEST_F(ScheduleConfigParserTest, ConnectorCommunicationTypeDefaultIsTcp) {
  std::string yaml_path = CreateConnectorYamlConfig("prefill", "host:8080", "unknown_type");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  ASSERT_TRUE(parser_->GetConnectorConfigs(connector_config).OK());
  EXPECT_EQ(connector_config.communication_type, CommunicationType::TCP);
}

TEST_F(ScheduleConfigParserTest, ConnectorAllFieldsParsedCorrectly) {
  std::string yaml_path =
      CreateConnectorYamlConfig("decode", "10.0.0.1:9090", "zmq", "10.0.0.2:7070", "10.0.0.3:6060", "test_cluster");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  ConnectorConfig connector_config;
  ASSERT_TRUE(parser_->GetConnectorConfigs(connector_config).OK());
  EXPECT_EQ(connector_config.group_role, GroupRole::DECODE);
  EXPECT_EQ(connector_config.router_addr, "http://10.0.0.1:9090");
  EXPECT_EQ(connector_config.inference_addr, "10.0.0.2:7070");
  EXPECT_EQ(connector_config.coordinator_addr, "10.0.0.3:6060");
  EXPECT_EQ(connector_config.cluster_name, "test_cluster");
  EXPECT_EQ(connector_config.communication_type, CommunicationType::ZMQ);
}

// Prefix cache 与 split fuse 配置解耦：split_fuse 不再依赖 prefix cache 开关。
TEST_F(ScheduleConfigParserTest, SplitFuseOnlyPreservesSplitFuseAndEnablesRuntimePrefixPath) {
  std::string yaml_path = CreateTempYamlConfig(/*enable_prefix_caching=*/false, /*split_fuse_token_num=*/256);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  std::string model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  ModelConfig model_config;
  model_config.num_key_value_heads = 1;
  model_config.max_training_seq_len = 4096;

  ASSERT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());
  ASSERT_TRUE(parser_->UpdateModelConfig(model_config).OK());

  CacheManagerConfig cache_config;
  parser_->GetCacheManagerConfig(cache_config);
  EXPECT_FALSE(cache_config.enable_prefix_caching);

  BatchSchedulerConfig batch_config;
  parser_->GetBatchSchedulerConfig(batch_config);
  EXPECT_EQ(batch_config.split_fuse_token_num, 256u);

  RuntimeConfig runtime_config;
  ASSERT_TRUE(parser_->GetRuntimeConfig(runtime_config).OK());
  EXPECT_TRUE(runtime_config.enable_prefix_caching);
}

TEST_F(ScheduleConfigParserTest, PrefixCacheOnlyEnablesRuntimePrefixPath) {
  std::string yaml_path = CreateTempYamlConfig(/*enable_prefix_caching=*/true, /*split_fuse_token_num=*/0);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  std::string model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  ModelConfig model_config;
  model_config.num_key_value_heads = 1;
  model_config.max_training_seq_len = 4096;

  ASSERT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());
  ASSERT_TRUE(parser_->UpdateModelConfig(model_config).OK());

  CacheManagerConfig cache_config;
  parser_->GetCacheManagerConfig(cache_config);
  EXPECT_TRUE(cache_config.enable_prefix_caching);

  BatchSchedulerConfig batch_config;
  parser_->GetBatchSchedulerConfig(batch_config);
  EXPECT_EQ(batch_config.split_fuse_token_num, 0u);

  RuntimeConfig runtime_config;
  ASSERT_TRUE(parser_->GetRuntimeConfig(runtime_config).OK());
  EXPECT_TRUE(runtime_config.enable_prefix_caching);
}

TEST_F(ScheduleConfigParserTest, PrefixCacheAndSplitFuseBothDisabled) {
  std::string yaml_path = CreateTempYamlConfig(/*enable_prefix_caching=*/false, /*split_fuse_token_num=*/0);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  std::string model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  ModelConfig model_config;
  model_config.num_key_value_heads = 1;
  model_config.max_training_seq_len = 4096;

  ASSERT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());
  ASSERT_TRUE(parser_->UpdateModelConfig(model_config).OK());

  CacheManagerConfig cache_config;
  parser_->GetCacheManagerConfig(cache_config);
  EXPECT_FALSE(cache_config.enable_prefix_caching);

  BatchSchedulerConfig batch_config;
  parser_->GetBatchSchedulerConfig(batch_config);
  EXPECT_EQ(batch_config.split_fuse_token_num, 0u);

  RuntimeConfig runtime_config;
  ASSERT_TRUE(parser_->GetRuntimeConfig(runtime_config).OK());
  EXPECT_FALSE(runtime_config.enable_prefix_caching);
}

TEST_F(ScheduleConfigParserTest, PrefixCacheAndSplitFuseBothEnabled) {
  std::string yaml_path = CreateTempYamlConfig(/*enable_prefix_caching=*/true, /*split_fuse_token_num=*/256);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  std::string model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  ModelConfig model_config;
  model_config.num_key_value_heads = 1;
  model_config.max_training_seq_len = 4096;

  ASSERT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());
  ASSERT_TRUE(parser_->UpdateModelConfig(model_config).OK());

  CacheManagerConfig cache_config;
  parser_->GetCacheManagerConfig(cache_config);
  EXPECT_TRUE(cache_config.enable_prefix_caching);

  BatchSchedulerConfig batch_config;
  parser_->GetBatchSchedulerConfig(batch_config);
  EXPECT_EQ(batch_config.split_fuse_token_num, 256u);

  RuntimeConfig runtime_config;
  ASSERT_TRUE(parser_->GetRuntimeConfig(runtime_config).OK());
  EXPECT_TRUE(runtime_config.enable_prefix_caching);
}

TEST_F(ScheduleConfigParserTest, ConnectorPDSeparationDisablesSplitFuse) {
  std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
  std::string filename = "schedule_config_test_" + std::to_string(test_file_counter_++) + ".yaml";
  std::filesystem::path temp_file_path = temp_dir / filename;
  temp_files_.push_back(temp_file_path.string());

  std::filesystem::path model_dir = temp_dir / ("test_model_dir_" + std::to_string(test_file_counter_));
  temp_dirs_.push_back(model_dir.string());
  if (std::filesystem::exists(model_dir)) std::filesystem::remove_all(model_dir);
  std::filesystem::create_directory(model_dir);

  std::ofstream file(temp_file_path);
  ASSERT_TRUE(file.is_open());

  file << "setting:\n";
  file << "  global:\n";
  file << "    tensor_para_size: 1\n";
  file << "  batch_scheduler:\n";
  file << "    max_token_len: 4096\n";
  file << "    max_step_tokens: 4096\n";
  file << "    max_batch_size: 32\n";
  file << "    enable_auto_prefix_cache: true\n";
  file << "    split_fuse_token_num: 256\n";
  file << "  block_manager:\n";
  file << "    block_token_num: 16\n";
  file << "    reserved_device_memory_ratio: 0.01\n";
  file << "  connector:\n";
  file << "    group_role: prefill\n";
  file << "    router_addr: \"10.0.0.1:8080\"\n";
  file << "model_spec:\n";
  file << "  base_model:\n";
  file << "    model_dir: \"" << model_dir.string() << "\"\n";
  file.close();

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(temp_file_path.string()).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  BatchSchedulerConfig batch_config;
  parser_->GetBatchSchedulerConfig(batch_config);
  EXPECT_EQ(batch_config.split_fuse_token_num, 0)
      << "split_fuse_token_num should be 0 when PD separation + prefix_cache + split_fuse are all enabled";
}

TEST_F(ScheduleConfigParserTest, ConnectorPDSeparationPreservesSplitFuseWhenPrefixCacheDisabled) {
  std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
  std::string filename = "schedule_config_test_" + std::to_string(test_file_counter_++) + ".yaml";
  std::filesystem::path temp_file_path = temp_dir / filename;
  temp_files_.push_back(temp_file_path.string());

  std::filesystem::path model_dir = temp_dir / ("test_model_dir_" + std::to_string(test_file_counter_));
  temp_dirs_.push_back(model_dir.string());
  if (std::filesystem::exists(model_dir)) std::filesystem::remove_all(model_dir);
  std::filesystem::create_directory(model_dir);

  std::ofstream file(temp_file_path);
  ASSERT_TRUE(file.is_open());

  file << "setting:\n";
  file << "  global:\n";
  file << "    tensor_para_size: 1\n";
  file << "  batch_scheduler:\n";
  file << "    max_token_len: 4096\n";
  file << "    max_step_tokens: 4096\n";
  file << "    max_batch_size: 32\n";
  file << "    enable_auto_prefix_cache: false\n";
  file << "    split_fuse_token_num: 256\n";
  file << "  block_manager:\n";
  file << "    block_token_num: 16\n";
  file << "    reserved_device_memory_ratio: 0.01\n";
  file << "  connector:\n";
  file << "    group_role: prefill\n";
  file << "    router_addr: \"10.0.0.1:8080\"\n";
  file << "model_spec:\n";
  file << "  base_model:\n";
  file << "    model_dir: \"" << model_dir.string() << "\"\n";
  file.close();

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(temp_file_path.string()).OK());
  ModelConfig model_config;
  model_config.num_key_value_heads = 1;
  model_config.max_training_seq_len = 4096;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();
  ASSERT_TRUE(parser_->UpdateModelConfig(model_config).OK());

  CacheManagerConfig cache_config;
  parser_->GetCacheManagerConfig(cache_config);
  EXPECT_FALSE(cache_config.enable_prefix_caching);

  BatchSchedulerConfig batch_config;
  parser_->GetBatchSchedulerConfig(batch_config);
  EXPECT_EQ(batch_config.split_fuse_token_num, 256u)
      << "split_fuse_token_num should be preserved when PD separation is enabled but prefix cache is disabled";

  RuntimeConfig runtime_config;
  ASSERT_TRUE(parser_->GetRuntimeConfig(runtime_config).OK());
  EXPECT_TRUE(runtime_config.enable_prefix_caching);
}

// ==================== GlobalCacheConnector 配置解析测试 ====================

TEST_F(ScheduleConfigParserTest, GlobalCacheConnectorNotConfigured) {
  // 默认 YAML 中没有 global_cache_connector 段
  std::string yaml_path = CreateConnectorYamlConfig("none");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK());

  GlobalCacheConnectorConfig config;
  Status get_status = parser_->GetGlobalCacheConnectorConfig(config);
  EXPECT_FALSE(get_status.OK()) << "Should fail when endpoint is not configured";
}

TEST_F(ScheduleConfigParserTest, GlobalCacheConnectorFullConfig) {
  std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
  std::string filename = "schedule_config_test_" + std::to_string(test_file_counter_++) + ".yaml";
  std::filesystem::path temp_file_path = temp_dir / filename;
  temp_files_.push_back(temp_file_path.string());

  std::filesystem::path model_dir = temp_dir / ("test_model_dir_" + std::to_string(test_file_counter_));
  temp_dirs_.push_back(model_dir.string());
  if (std::filesystem::exists(model_dir)) std::filesystem::remove_all(model_dir);
  std::filesystem::create_directory(model_dir);

  std::ofstream file(temp_file_path);
  ASSERT_TRUE(file.is_open());

  file << "setting:\n";
  file << "  global:\n";
  file << "    tensor_para_size: 1\n";
  file << "  batch_scheduler:\n";
  file << "    max_token_len: 4096\n";
  file << "    max_step_tokens: 4096\n";
  file << "    max_batch_size: 32\n";
  file << "  block_manager:\n";
  file << "    block_token_num: 16\n";
  file << "    reserved_device_memory_ratio: 0.01\n";
  file << "  global_cache_connector:\n";
  file << "    service_endpoint: \"10.0.0.1:12345\"\n";
  file << "    backend_type: mooncake\n";
  file << "    timeout_ms: 3000\n";
  file << "    max_retry: 5\n";
  file << "    fetch_thread_num: 8\n";
  file << "    role: prefill\n";
  file << "    protocol: rdma\n";
  file << "    rdma_devices: mlx5_0\n";
  file << "    master_server_addr: \"10.0.0.2:50051\"\n";
  file << "    global_segment_size: 33554432\n";
  file << "    local_buffer_size: 33554432\n";
  file << "model_spec:\n";
  file << "  base_model:\n";
  file << "    model_dir: \"" << model_dir.string() << "\"\n";
  file.close();

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(temp_file_path.string()).OK());
  ModelConfig model_config;

  Status status = parser_->ParseScheduleConfig(yaml_reader, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  GlobalCacheConnectorConfig config;
  Status get_status = parser_->GetGlobalCacheConnectorConfig(config);
  ASSERT_TRUE(get_status.OK()) << get_status.GetMessage();

  EXPECT_EQ(config.service_endpoint, "10.0.0.1:12345");
  EXPECT_EQ(config.backend_type, "mooncake");
  EXPECT_EQ(config.timeout_ms, 3000);
  EXPECT_EQ(config.max_retry, 5);
  EXPECT_EQ(config.fetch_thread_num, 8);
  EXPECT_EQ(config.role, "prefill");
  EXPECT_EQ(config.protocol, "rdma");
  EXPECT_EQ(config.rdma_devices, "mlx5_0");
  EXPECT_EQ(config.master_server_addr, "10.0.0.2:50051");
  EXPECT_EQ(config.global_segment_size, 33554432u);
  EXPECT_EQ(config.local_buffer_size, 33554432u);
}

TEST_F(ScheduleConfigParserTest, GlobalCacheConnectorDefaults) {
  std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
  std::string filename = "schedule_config_test_" + std::to_string(test_file_counter_++) + ".yaml";
  std::filesystem::path temp_file_path = temp_dir / filename;
  temp_files_.push_back(temp_file_path.string());

  std::filesystem::path model_dir = temp_dir / ("test_model_dir_" + std::to_string(test_file_counter_));
  temp_dirs_.push_back(model_dir.string());
  if (std::filesystem::exists(model_dir)) std::filesystem::remove_all(model_dir);
  std::filesystem::create_directory(model_dir);

  std::ofstream file(temp_file_path);
  ASSERT_TRUE(file.is_open());

  file << "setting:\n";
  file << "  global:\n";
  file << "    tensor_para_size: 1\n";
  file << "  batch_scheduler:\n";
  file << "    max_token_len: 4096\n";
  file << "    max_step_tokens: 4096\n";
  file << "    max_batch_size: 32\n";
  file << "  block_manager:\n";
  file << "    block_token_num: 16\n";
  file << "    reserved_device_memory_ratio: 0.01\n";
  file << "  global_cache_connector:\n";
  file << "    service_endpoint: \"localhost:9999\"\n";
  file << "model_spec:\n";
  file << "  base_model:\n";
  file << "    model_dir: \"" << model_dir.string() << "\"\n";
  file.close();

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(temp_file_path.string()).OK());
  ModelConfig model_config;

  ASSERT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());

  GlobalCacheConnectorConfig config;
  ASSERT_TRUE(parser_->GetGlobalCacheConnectorConfig(config).OK());

  EXPECT_EQ(config.service_endpoint, "localhost:9999");
  EXPECT_EQ(config.backend_type, "mooncake");
  EXPECT_EQ(config.timeout_ms, 5000);
  EXPECT_EQ(config.max_retry, 3);
  EXPECT_EQ(config.fetch_thread_num, 4);
  EXPECT_EQ(config.role, "");
  EXPECT_EQ(config.protocol, "tcp");
  EXPECT_EQ(config.master_server_addr, "127.0.0.1:50051");
  EXPECT_EQ(config.global_segment_size, 16u * 1024 * 1024);
  EXPECT_EQ(config.local_buffer_size, 16u * 1024 * 1024);
}

// ==================== swa_block_num_ratio 配置解析与自动调整测试 ====================

// Test: swa_block_num_ratio defaults to 0.1 when the key is absent from YAML,
// regardless of whether the model has SWA.
TEST_F(ScheduleConfigParserTest, SwaBlockNumRatioDefaultsToHalfWhenNotConfigured) {
  // Omit setting.cache_manager.swa_block_num_ratio from YAML (pass -1 sentinel);
  // the default value is 0.1 regardless of whether the model has SWA.
  std::string yaml_path = CreateTempYamlConfigWithSwa(-1.0f);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());

  // Non-SWA model: use_dsv4_attn=false, sliding_window=0
  // num_key_value_heads=1 bypasses the divisibility check in UpdateModelConfig.
  ModelConfig model_config;
  model_config.num_key_value_heads = 1;
  model_config.use_dsv4_attn = false;
  model_config.dsv4_attn_config.sliding_window = 0;

  ASSERT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());
  ASSERT_TRUE(parser_->UpdateModelConfig(model_config).OK());

  BatchSchedulerConfig batch_config;
  parser_->GetBatchSchedulerConfig(batch_config);
  EXPECT_FLOAT_EQ(batch_config.swa_block_num_ratio, 0.1f)
      << "swa_block_num_ratio should default to 0.1 when not configured in YAML";
}

// Test: swa_block_num_ratio is read from YAML when explicitly set.
TEST_F(ScheduleConfigParserTest, SwaBlockNumRatioReadFromYaml) {
  // Explicitly set swa_block_num_ratio to 0.3 in YAML
  std::string yaml_path = CreateTempYamlConfigWithSwa(0.3f);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());

  ModelConfig model_config;
  model_config.num_key_value_heads = 1;
  model_config.use_dsv4_attn = false;
  model_config.dsv4_attn_config.sliding_window = 0;

  ASSERT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());
  ASSERT_TRUE(parser_->UpdateModelConfig(model_config).OK());

  BatchSchedulerConfig batch_config;
  parser_->GetBatchSchedulerConfig(batch_config);
  EXPECT_FLOAT_EQ(batch_config.swa_block_num_ratio, 0.3f) << "swa_block_num_ratio should be the value read from YAML";
}

TEST_F(ScheduleConfigParserTest, NonContiguousSwaModeDefaultsToFalseWhenNotConfigured) {
  std::string yaml_path = CreateTempYamlConfigWithSwa(-1.0f);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());

  ModelConfig model_config;
  model_config.num_key_value_heads = 1;

  ASSERT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());

  CacheManagerConfig cache_config;
  ASSERT_TRUE(parser_->GetCacheManagerConfig(cache_config).OK());
  EXPECT_FALSE(cache_config.enable_non_contiguous_swa_mode);
}

TEST_F(ScheduleConfigParserTest, NonContiguousSwaModeReadFromYaml) {
  std::string yaml_path = CreateTempYamlConfigWithSwa(-1.0f, true, true);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());

  ModelConfig model_config;
  model_config.num_key_value_heads = 1;

  ASSERT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());

  CacheManagerConfig cache_config;
  ASSERT_TRUE(parser_->GetCacheManagerConfig(cache_config).OK());
  EXPECT_TRUE(cache_config.enable_non_contiguous_swa_mode);
}

// Test: when the model has SWA and swa_block_num_ratio is not set in YAML,
// it uses the default value of 0.1.
TEST_F(ScheduleConfigParserTest, SwaBlockNumRatioDefaultsToHalfForSwaModel) {
  // Omit swa_block_num_ratio from YAML
  std::string yaml_path = CreateTempYamlConfigWithSwa(-1.0f);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());

  // SWA model: use_dsv4_attn=true, sliding_window > 0
  // num_key_value_heads=1 bypasses the divisibility check in UpdateModelConfig.
  ModelConfig model_config;
  model_config.num_key_value_heads = 1;
  model_config.use_dsv4_attn = true;
  model_config.dsv4_attn_config.sliding_window = 4096;

  ASSERT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());
  ASSERT_TRUE(parser_->UpdateModelConfig(model_config).OK());

  BatchSchedulerConfig batch_config;
  parser_->GetBatchSchedulerConfig(batch_config);
  EXPECT_FLOAT_EQ(batch_config.swa_block_num_ratio, 0.1f)
      << "swa_block_num_ratio should default to 0.1 for SWA model when not configured";
}

// Test: when the model has SWA but swa_block_num_ratio is explicitly set to a
// non-zero value in YAML, UpdateModelConfig must NOT override it.
TEST_F(ScheduleConfigParserTest, SwaBlockNumRatioExplicitNonZeroPreservedForSwaModel) {
  // Explicitly set swa_block_num_ratio to 0.25 in YAML
  std::string yaml_path = CreateTempYamlConfigWithSwa(0.25f);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());

  // SWA model; num_key_value_heads=1 bypasses the divisibility check.
  ModelConfig model_config;
  model_config.num_key_value_heads = 1;
  model_config.use_dsv4_attn = true;
  model_config.dsv4_attn_config.sliding_window = 4096;

  ASSERT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());
  ASSERT_TRUE(parser_->UpdateModelConfig(model_config).OK());

  BatchSchedulerConfig batch_config;
  parser_->GetBatchSchedulerConfig(batch_config);
  EXPECT_FLOAT_EQ(batch_config.swa_block_num_ratio, 0.25f)
      << "Explicit non-zero swa_block_num_ratio should not be overridden by auto-adjustment";
}

// Test: when swa_block_num_ratio is explicitly set to 0 in YAML, it is corrected
// to the default 0.1 because 0 is not a valid ratio (<=0 triggers correction).
TEST_F(ScheduleConfigParserTest, SwaBlockNumRatioExplicitZeroCorrectedToDefault) {
  // Explicitly set swa_block_num_ratio to 0 in YAML
  std::string yaml_path = CreateTempYamlConfigWithSwa(0.0f);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());

  // SWA model; num_key_value_heads=1 bypasses the divisibility check.
  ModelConfig model_config;
  model_config.num_key_value_heads = 1;
  model_config.use_dsv4_attn = true;
  model_config.dsv4_attn_config.sliding_window = 4096;

  ASSERT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());
  ASSERT_TRUE(parser_->UpdateModelConfig(model_config).OK());

  BatchSchedulerConfig batch_config;
  parser_->GetBatchSchedulerConfig(batch_config);
  EXPECT_FLOAT_EQ(batch_config.swa_block_num_ratio, 0.1f)
      << "swa_block_num_ratio=0 in YAML should be corrected to the default 0.1";
}

TEST_F(ScheduleConfigParserTest, MambaBlockNumDerivedFromDeviceBlocks) {
  // Same derivation as SWA: mamba_block_num = device_blocks_num * ratio,
  // with the mamba pool memory cost folded into each KV block.
  BlockManagerConfig block_config;
  block_config.block_device_memory_ratio = 1.0f;
  block_config.reserved_device_memory_ratio = 0.01f;
  block_config.block_host_memory_factor = 0.0f;
  block_config.device_allocator_config.block_size = 10;
  block_config.device_allocator_config.block_token_num = 1;
  block_config.host_allocator_config.block_size = 10;
  block_config.mamba_device_allocator_config.block_size = 100;
  parser_->SetBlockManagerConfig(block_config);

  BatchSchedulerConfig batch_config;
  batch_config.launch_block_threshold = 0;
  batch_config.mamba_block_num_ratio = 0.5f;
  parser_->SetBatchSchedulerConfig(batch_config);

  ASSERT_TRUE(parser_->CalculateBlockNumber(/*device_free=*/1000, /*device_total=*/1000).OK());
  ASSERT_TRUE(parser_->GetBlockManagerConfig(block_config).OK());
  // effective_block_size = 10 + 0.5 * 100 = 60 -> device_blocks = 1000 / 60 = 16.
  EXPECT_EQ(block_config.device_allocator_config.blocks_num, 16u);
  EXPECT_EQ(block_config.mamba_device_allocator_config.blocks_num, 8u);
}

TEST_F(ScheduleConfigParserTest, MambaLayoutPropagatesBlockSize) {
  class TestModelCacheLayout : public BaseModelCacheLayout {
   public:
    TestModelCacheLayout() {
      RegisterBlockCacheLayout("kv-cache", std::make_shared<MambaBlockCacheLayout>(64));
      RegisterBlockCacheLayout("mamba-state", std::make_shared<MambaBlockCacheLayout>(200));
    }
  };

  ModelConfig model_config;
  model_config.num_layer = 2;
  ASSERT_TRUE(parser_->InitializeBlockManagerConfig(model_config, std::make_shared<TestModelCacheLayout>()).OK());

  BlockManagerConfig block_config;
  ASSERT_TRUE(parser_->GetBlockManagerConfig(block_config).OK());
  EXPECT_EQ(block_config.mamba_device_allocator_config.block_size, 200u);
}

TEST_F(ScheduleConfigParserTest, MambaBlockNumRatioReadFromYaml) {
  std::string yaml_path = CreateTempYamlConfigWithSwa(/*swa_block_num_ratio=*/-1.0f,
                                                      /*enable_non_contiguous_swa_mode=*/false,
                                                      /*has_enable_non_contiguous_swa_mode=*/false,
                                                      /*mamba_block_num_ratio=*/0.25f);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;
  model_config.num_key_value_heads = 1;
  ASSERT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());

  BatchSchedulerConfig batch_config;
  ASSERT_TRUE(parser_->GetBatchSchedulerConfig(batch_config).OK());
  EXPECT_FLOAT_EQ(batch_config.mamba_block_num_ratio, 0.25f);
}

TEST_F(ScheduleConfigParserTest, NonMambaModelAllowsDisabledMambaBlockNumRatio) {
  std::string yaml_path = CreateTempYamlConfigWithSwa(/*swa_block_num_ratio=*/-1.0f,
                                                      /*enable_non_contiguous_swa_mode=*/false,
                                                      /*has_enable_non_contiguous_swa_mode=*/false,
                                                      /*mamba_block_num_ratio=*/0.0f);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;
  model_config.num_key_value_heads = 1;

  Status status;
  EXPECT_NO_THROW(status = parser_->ParseScheduleConfig(yaml_reader, model_config));
  ASSERT_TRUE(status.OK()) << status.GetMessage();

  BatchSchedulerConfig batch_config;
  ASSERT_TRUE(parser_->GetBatchSchedulerConfig(batch_config).OK());
  EXPECT_FLOAT_EQ(batch_config.mamba_block_num_ratio, 0.0f);
}

// max_chunked_kv_tokens 设置逻辑: 默认→max_step_token_num, 显式设置→保留, 小于 max_token_len→clamp.
TEST_F(ScheduleConfigParserTest, MaxChunkedKvTokensConfig) {
  auto run = [this](size_t yaml_value, size_t expected) {
    std::string yaml_path = CreateTempYamlConfig(false, 0, false, yaml_value);
    EXPECT_FALSE(yaml_path.empty());
    YamlReader yaml_reader;
    EXPECT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
    std::string model_dir =
        yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
    ModelConfig model_config;
    model_config.num_key_value_heads = 1;      // bypass TP divisibility check
    model_config.max_training_seq_len = 4096;  // prevent max_token_len from being reduced to 0
    EXPECT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());
    EXPECT_TRUE(parser_->UpdateModelConfig(model_config).OK());
    BatchSchedulerConfig batch_config;
    parser_->GetBatchSchedulerConfig(batch_config);
    RuntimeConfig runtime_config;
    EXPECT_TRUE(parser_->GetRuntimeConfig(runtime_config).OK());
    EXPECT_EQ(batch_config.max_chunked_kv_tokens, expected);
    EXPECT_EQ(runtime_config.max_chunked_kv_tokens, expected);
  };

  run(0, 4096u);     // 未设置→回退为 max_step_token_num
  run(8192, 8192u);  // >= max_token_len→保留原值
  run(256, 4096u);   // < max_token_len→clamp
}

// max_step_token_num is reset to max_token_len when split-fuse is disabled or
// blocked multi-token forwarding KV is not supported, and max_step < max_token.
TEST_F(ScheduleConfigParserTest, MaxStepTokenNumResetToMaxTokenLen) {
  auto run = [this](size_t split_fuse, size_t max_step_tokens, bool enable_blocked_mtkv, size_t expected) {
    std::string yaml_path = CreateTempYamlConfig(true, split_fuse, false, 0, max_step_tokens, enable_blocked_mtkv);
    EXPECT_FALSE(yaml_path.empty());
    YamlReader yaml_reader;
    EXPECT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
    std::string model_dir =
        yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
    ModelConfig model_config;
    model_config.num_key_value_heads = 1;      // bypass TP divisibility check
    model_config.max_training_seq_len = 4096;  // prevent max_token_len from being reduced to 0
    EXPECT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());
    EXPECT_TRUE(parser_->UpdateModelConfig(model_config).OK());
    BatchSchedulerConfig batch_config;
    parser_->GetBatchSchedulerConfig(batch_config);
    EXPECT_EQ(batch_config.max_step_token_num, expected);
  };

  run(0, 2048, false, 4096u);    // split_fuse=0 → reset to max_token_len
  run(256, 2048, false, 4096u);  // blocked_mtkv=false → reset
  run(256, 2048, true, 2048u);   // both enabled → stays
  run(0, 4096, false, 4096u);    // max_step >= max_token → stays
}

// compressor_config.state_dtype 设置逻辑:
//   - use_dsv4_attn=true 时, UpdateMembers 自动设为 TYPE_BF16 (env KLLM_USE_FP32_COMPRESSOR
//     未设置的默认分支)。
//   - use_dsv4_attn=false 时分支不进入, 保持结构体默认值 TYPE_BF16。
TEST_F(ScheduleConfigParserTest, CompressorStateDtypeConfig) {
  auto run = [this](bool use_dsv4_attn, DataType expected) {
    std::string yaml_path = CreateTempYamlConfig(false, 0, false);
    EXPECT_FALSE(yaml_path.empty());
    YamlReader yaml_reader;
    EXPECT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
    std::string model_dir =
        yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
    ModelConfig model_config = LoadModelConfig(model_dir, false);
    model_config.weight_data_type = TYPE_FP16;
    model_config.use_mla = use_dsv4_attn;
    model_config.use_dsv4_attn = use_dsv4_attn;
    EXPECT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());
    EXPECT_NO_THROW(parser_->UpdateMembers(model_dir, model_config));
    RuntimeConfig runtime_config;
    EXPECT_TRUE(parser_->GetRuntimeConfig(runtime_config).OK());
    EXPECT_EQ(runtime_config.compressor_config.state_dtype, expected);
  };

  run(true, TYPE_BF16);   // DSv4 → 自动设为 BF16
  run(false, TYPE_BF16);  // 非 DSv4 → 不改写, 保持结构体默认 BF16
}

// enable_executor_build_ahead 缺省为 false，并同步到 RuntimeConfig。
TEST_F(ScheduleConfigParserTest, ExecutorBuildAheadDefaultsFalseWhenOmitted) {
  std::string yaml_path = CreateTempYamlConfig(false, 0, false);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  std::string model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  ModelConfig model_config = LoadModelConfig(model_dir, false);
  model_config.num_key_value_heads = 1;
  model_config.max_training_seq_len = 4096;

  ASSERT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());

  BatchSchedulerConfig batch_config;
  parser_->GetBatchSchedulerConfig(batch_config);
  EXPECT_FALSE(batch_config.enable_executor_build_ahead);

  RuntimeConfig runtime_config;
  ASSERT_TRUE(parser_->GetRuntimeConfig(runtime_config).OK());
  EXPECT_FALSE(runtime_config.enable_executor_build_ahead);
}

// yaml 显式开启时，BatchSchedulerConfig 与 RuntimeConfig 均应带上 enable_executor_build_ahead。
TEST_F(ScheduleConfigParserTest, ExecutorBuildAheadYamlPropagatesToRuntimeConfig) {
  std::string yaml_path = CreateTempYamlConfigWithExecutorBuildAhead(/*enable=*/true, /*enable_async=*/true);
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  std::string model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  ModelConfig model_config = LoadModelConfig(model_dir, false);
  model_config.num_key_value_heads = 1;
  model_config.max_training_seq_len = 4096;

  ASSERT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());
  ASSERT_TRUE(parser_->UpdateModelConfig(model_config).OK());

  BatchSchedulerConfig batch_config;
  parser_->GetBatchSchedulerConfig(batch_config);
  EXPECT_TRUE(batch_config.enable_executor_build_ahead);
  EXPECT_TRUE(batch_config.enable_async);

  RuntimeConfig runtime_config;
  ASSERT_TRUE(parser_->GetRuntimeConfig(runtime_config).OK());
  EXPECT_TRUE(runtime_config.enable_executor_build_ahead);
  EXPECT_TRUE(runtime_config.enable_async);
}

// --- PdV2RuntimeConfig tests ---
//
// Build a yaml fixture with backend=v2 + EVENT_DRIVEN scheduler (the latter
// is required by InitConnectorConfig's fail-fast at L662-669). pd_v2 fields
// (metadata_addr, redis_password, transfer_layer_chunk_size,
// pd_v2_executor_rpc_base_port, pd_v2_local_complete_threshold) and shared
// fields (group_role, cluster_name, inference_addr, router_addr,
// coordinator_addr, heartbeat_interval_ms) are populated.
namespace {
std::string WritePdV2YamlConfig(std::vector<std::string> *temp_files, std::vector<std::string> *temp_dirs,
                                int *test_file_counter, const std::string &group_role, const std::string &backend) {
  std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
  std::string filename = "pd_v2_runtime_config_test_" + std::to_string((*test_file_counter)++) + ".yaml";
  std::filesystem::path temp_file_path = temp_dir / filename;
  temp_files->push_back(temp_file_path.string());

  std::filesystem::path model_dir = temp_dir / ("test_model_dir_" + std::to_string(*test_file_counter));
  temp_dirs->push_back(model_dir.string());
  if (std::filesystem::exists(model_dir)) {
    std::filesystem::remove_all(model_dir);
  }
  std::filesystem::create_directory(model_dir);

  std::ofstream file(temp_file_path);
  if (!file.is_open()) return "";
  file << "setting:\n";
  file << "  global:\n";
  file << "    tensor_para_size: 1\n";
  file << "  batch_scheduler:\n";
  file << "    scheduler_type: 1\n";  // EVENT_DRIVEN — required for backend=v2.
  file << "    max_token_len: 4096\n";
  file << "    max_step_tokens: 4096\n";
  file << "    max_batch_size: 32\n";
  file << "  block_manager:\n";
  file << "    block_token_num: 16\n";
  file << "    reserved_device_memory_ratio: 0.01\n";
  file << "  connector:\n";
  file << "    group_role: " << group_role << "\n";
  file << "    backend: " << backend << "\n";
  file << "    cluster_name: pd_v2_test_cluster\n";
  file << "    inference_addr: \"127.0.0.1:18091\"\n";
  file << "    router_addr: \"http://127.0.0.1:9080\"\n";
  file << "    coordinator_addr: \"127.0.0.1:11092\"\n";
  file << "    metadata_addr: \"redis://127.0.0.1:6379\"\n";
  file << "    redis_password: \"test_redis_pw\"\n";
  file << "    transfer_layer_chunk_size: 4\n";
  file << "    pd_v2_executor_rpc_base_port: 21000\n";
  file << "    pd_v2_local_complete_threshold: 16\n";
  file << "    heartbeat_interval_ms: 7000\n";
  file << "model_spec:\n";
  file << "  base_model:\n";
  file << "    model_dir: \"" << model_dir.string() << "\"\n";
  file.close();
  return temp_file_path.string();
}
}  // namespace

TEST_F(ScheduleConfigParserTest, PdV2RuntimeConfigPopulatedOnBackendV2) {
  std::string yaml_path = WritePdV2YamlConfig(&temp_files_, &temp_dirs_, &test_file_counter_, "decode", "v2");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;
  ASSERT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());

  PdV2RuntimeConfig pd_v2;
  ASSERT_TRUE(parser_->GetPdV2RuntimeConfig(pd_v2).OK());
  EXPECT_TRUE(pd_v2.is_pd_v2);
  EXPECT_EQ(pd_v2.group_role, GroupRole::DECODE);
  EXPECT_EQ(pd_v2.cluster_name, "pd_v2_test_cluster");
  EXPECT_EQ(pd_v2.inference_addr, "127.0.0.1:18091");
  EXPECT_EQ(pd_v2.router_addr, "http://127.0.0.1:9080");
  EXPECT_EQ(pd_v2.coordinator_addr, "127.0.0.1:11092");
  EXPECT_EQ(pd_v2.metadata_addr, "redis://127.0.0.1:6379");
  EXPECT_EQ(pd_v2.redis_password, "test_redis_pw");
  EXPECT_EQ(pd_v2.transfer_layer_chunk_size, 4u);
  EXPECT_EQ(pd_v2.pd_v2_executor_rpc_base_port, 21000u);
  EXPECT_EQ(pd_v2.pd_v2_local_complete_threshold, 16);
  EXPECT_EQ(pd_v2.heartbeat_interval_ms, 7000);

  // ConnectorConfig must mirror shared fields for back-compat (parser
  // double-writes during the deprecation window).
  ConnectorConfig cc;
  ASSERT_TRUE(parser_->GetConnectorConfigs(cc).OK());
  EXPECT_TRUE(cc.is_pd_v2);
  EXPECT_EQ(cc.group_role, GroupRole::DECODE);
  EXPECT_EQ(cc.cluster_name, pd_v2.cluster_name);
  EXPECT_EQ(cc.inference_addr, pd_v2.inference_addr);
}

TEST_F(ScheduleConfigParserTest, PdV2RuntimeConfigReturnsNotFoundWhenV1) {
  // V1 yaml — connector.backend defaults to empty (= V1). GetPdV2RuntimeConfig
  // must signal not-initialised so callers downstream can skip pd_v2 paths.
  std::string yaml_path = CreateConnectorYamlConfig("decode", "10.0.0.1:8080");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;
  ASSERT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());

  PdV2RuntimeConfig pd_v2;
  EXPECT_FALSE(parser_->GetPdV2RuntimeConfig(pd_v2).OK());
}

TEST_F(ScheduleConfigParserTest, PdV2RuntimeConfigReturnsNotFoundWhenNoConnector) {
  // Standalone yaml (group_role=none) — pd_v2 must stay un-initialised.
  std::string yaml_path = CreateConnectorYamlConfig("none", "10.0.0.1:8080");
  ASSERT_FALSE(yaml_path.empty());

  YamlReader yaml_reader;
  ASSERT_TRUE(yaml_reader.LoadFile(yaml_path).OK());
  ModelConfig model_config;
  ASSERT_TRUE(parser_->ParseScheduleConfig(yaml_reader, model_config).OK());

  PdV2RuntimeConfig pd_v2;
  EXPECT_FALSE(parser_->GetPdV2RuntimeConfig(pd_v2).OK());
}

}  // namespace ksana_llm
