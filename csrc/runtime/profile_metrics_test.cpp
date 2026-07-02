/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/profile_metrics.h"

#include <cstring>

#include "gtest/gtest.h"

namespace ksana_llm {
namespace {

ForwardProfileOpId RegisterTestOp(const char* suffix) {
  return ForwardProfileOpRegistry::Register(suffix);
}

void ExpectMetricsEqual(const ProfileMetrics& expected, const ProfileMetrics& actual) {
  EXPECT_EQ(actual.schedule_forward_interval_ns, expected.schedule_forward_interval_ns);
  ASSERT_EQ(actual.forward_metrics.size(), expected.forward_metrics.size());
  for (const auto& [kind, expected_sub] : expected.forward_metrics) {
    const ForwardPerfMetrics* actual_sub = actual.Get(kind);
    ASSERT_NE(actual_sub, nullptr) << "missing forward kind " << static_cast<int>(kind);
    EXPECT_EQ(actual_sub->forward_time_ns, expected_sub.forward_time_ns);
    ASSERT_EQ(actual_sub->ops.size(), expected_sub.ops.size());
    for (const auto& [op_id, expected_op] : expected_sub.ops) {
      const LayerPerfMetrics* actual_op = actual_sub->GetOp(op_id);
      ASSERT_NE(actual_op, nullptr) << "missing op_id " << op_id;
      EXPECT_EQ(actual_op->cumulative_time_ns, expected_op.cumulative_time_ns);
      EXPECT_EQ(actual_op->count, expected_op.count);
    }
  }
}

void AppendI64(int64_t v, std::vector<int>& out) {
  out.push_back(static_cast<int>(static_cast<uint64_t>(v) >> 32));
  out.push_back(static_cast<int>(static_cast<uint32_t>(v)));
}

void AppendOpWire(ForwardProfileOpId op_id, const char* name, int64_t cumulative_time_ns, int32_t count,
                  std::vector<int>& out) {
  out.push_back(static_cast<int>(op_id));
  const int name_len = static_cast<int>(std::strlen(name));
  out.push_back(name_len);
  for (int i = 0; i < name_len; ++i) {
    out.push_back(static_cast<int>(static_cast<unsigned char>(name[i])));
  }
  AppendI64(cumulative_time_ns, out);
  out.push_back(count);
}

ProfileMetrics MakeSampleProfileMetrics() {
  ProfileMetrics metrics;
  metrics.schedule_forward_interval_ns = 1234;

  ForwardPerfMetrics& main = metrics.GetOrCreate(ForwardMetricsKind::kMain);
  main.forward_time_ns = 5678;
  main.GetOrCreateOp(ForwardProfileOp::kAttn).cumulative_time_ns = 100;
  main.GetOrCreateOp(ForwardProfileOp::kAttn).count = 2;
  main.GetOrCreateOp(ForwardProfileOp::kFfn).cumulative_time_ns = 200;
  main.GetOrCreateOp(ForwardProfileOp::kFfn).count = 3;

  const ForwardProfileOpId custom_op = RegisterTestOp("profile_metrics_test_mhc_pre");
  main.GetOrCreateOp(custom_op).cumulative_time_ns = 42;
  main.GetOrCreateOp(custom_op).count = 1;

  ForwardPerfMetrics& speculative = metrics.GetOrCreate(ForwardMetricsKind::kSpeculative);
  speculative.forward_time_ns = 900;
  speculative.GetOrCreateOp(ForwardProfileOp::kAttn).cumulative_time_ns = 11;
  speculative.GetOrCreateOp(ForwardProfileOp::kAttn).count = 1;

  return metrics;
}

}  // namespace

TEST(ForwardProfileOpRegistryTest, BuiltinOpsHaveStableIds) {
  EXPECT_EQ(ForwardProfileOpRegistry::Name(ForwardProfileOp::kAttn), std::string("attn"));
  EXPECT_EQ(ForwardProfileOpRegistry::Name(ForwardProfileOp::kFfn), std::string("ffn"));
  EXPECT_GE(ForwardProfileOpRegistry::RegisteredCount(), 2u);
}

TEST(ForwardProfileOpRegistryTest, RegisterRejectsDuplicateName) {
  const ForwardProfileOpId first = RegisterTestOp("profile_metrics_test_custom_op");
  const ForwardProfileOpId second = RegisterTestOp("profile_metrics_test_custom_op");
  EXPECT_NE(first, ForwardProfileOp::kInvalid);
  EXPECT_EQ(second, ForwardProfileOp::kInvalid);
  EXPECT_STREQ(ForwardProfileOpRegistry::Name(first), "profile_metrics_test_custom_op");
  EXPECT_STREQ(ForwardProfileOpRegistry::Name(second), "unknown");
}

TEST(ForwardProfileOpRegistryTest, RegisterRejectsBuiltinName) {
  EXPECT_EQ(RegisterTestOp("attn"), ForwardProfileOp::kInvalid);
  EXPECT_EQ(RegisterTestOp("ffn"), ForwardProfileOp::kInvalid);
}

TEST(ForwardProfileOpRegistryTest, IsValidRejectsInvalidId) {
  EXPECT_FALSE(ForwardProfileOpRegistry::IsValid(ForwardProfileOp::kInvalid));
  EXPECT_TRUE(ForwardProfileOpRegistry::IsValid(ForwardProfileOp::kAttn));
  EXPECT_TRUE(ForwardProfileOpRegistry::IsValid(ForwardProfileOp::kFfn));
}

TEST(ForwardProfileOpRegistryTest, DuplicateRegistrationDoesNotIncreaseCount) {
  const size_t count_before = ForwardProfileOpRegistry::RegisteredCount();
  const ForwardProfileOpId first = RegisterTestOp("profile_metrics_test_dup_count");
  ASSERT_NE(first, ForwardProfileOp::kInvalid);
  EXPECT_EQ(ForwardProfileOpRegistry::RegisteredCount(), count_before + 1);

  const ForwardProfileOpId duplicate = RegisterTestOp("profile_metrics_test_dup_count");
  EXPECT_EQ(duplicate, ForwardProfileOp::kInvalid);
  EXPECT_EQ(ForwardProfileOpRegistry::RegisteredCount(), count_before + 1);
}

TEST(ForwardProfileOpRegistryTest, DuplicateRegistrationKeepsFirstIdUsable) {
  const ForwardProfileOpId first = RegisterTestOp("profile_metrics_test_dup_keep_first");
  ASSERT_NE(first, ForwardProfileOp::kInvalid);

  ForwardPerfMetrics metrics;
  metrics.Accumulate(first, 100);
  EXPECT_EQ(RegisterTestOp("profile_metrics_test_dup_keep_first"), ForwardProfileOp::kInvalid);

  const LayerPerfMetrics* op_metrics = metrics.GetOp(first);
  ASSERT_NE(op_metrics, nullptr);
  EXPECT_EQ(op_metrics->cumulative_time_ns, 100);
  EXPECT_EQ(op_metrics->count, 1);
}

TEST(ForwardPerfMetricsTest, AccumulateIgnoresInvalidOpId) {
  ForwardPerfMetrics metrics;
  metrics.Accumulate(ForwardProfileOp::kInvalid, 100);
  metrics.Accumulate(ForwardProfileOp::kInvalid, 50);

  EXPECT_TRUE(metrics.ops.empty());
  EXPECT_EQ(metrics.GetOp(ForwardProfileOp::kInvalid), nullptr);
}

TEST(ForwardPerfMetricsTest, GetOrCreateOpWithInvalidIdDoesNotPersistInOps) {
  ForwardPerfMetrics metrics;
  metrics.GetOrCreateOp(ForwardProfileOp::kInvalid).cumulative_time_ns = 999;
  metrics.GetOrCreateOp(ForwardProfileOp::kInvalid).count = 3;

  EXPECT_TRUE(metrics.ops.empty());
}

TEST(ModelPerfMacroTest, EndNoSyncIgnoresInvalidOpId) {
  ForwardPerfMetrics metrics;
  metrics.Accumulate(ForwardProfileOp::kAttn, 1);

  MODEL_PERF_OP_BEGIN(&metrics, timer);
  MODEL_PERF_OP_END_NOSYNC(&metrics, ForwardProfileOp::kInvalid, timer);

  EXPECT_EQ(metrics.ops.size(), 1u);
  const LayerPerfMetrics* attn = metrics.GetOp(ForwardProfileOp::kAttn);
  ASSERT_NE(attn, nullptr);
  EXPECT_EQ(attn->count, 1);
  EXPECT_EQ(metrics.GetOp(ForwardProfileOp::kInvalid), nullptr);
}

TEST(ModelPerfMacroTest, EndNoSyncIgnoresDuplicateRegisteredOpId) {
  const ForwardProfileOpId valid_op = RegisterTestOp("profile_metrics_test_dup_macro_op");
  ASSERT_NE(valid_op, ForwardProfileOp::kInvalid);
  const ForwardProfileOpId duplicate = RegisterTestOp("profile_metrics_test_dup_macro_op");
  ASSERT_EQ(duplicate, ForwardProfileOp::kInvalid);

  ForwardPerfMetrics metrics;
  MODEL_PERF_OP_BEGIN(&metrics, timer);
  MODEL_PERF_OP_END_NOSYNC(&metrics, duplicate, timer);

  EXPECT_TRUE(metrics.ops.empty());
}

TEST(ForwardProfileOpRegistryTest, RegisterAssignsUniqueIdsForNewOps) {
  const ForwardProfileOpId op_a = RegisterTestOp("profile_metrics_test_op_a");
  const ForwardProfileOpId op_b = RegisterTestOp("profile_metrics_test_op_b");

  EXPECT_NE(op_a, op_b);
  EXPECT_GT(op_a, ForwardProfileOp::kFfn);
  EXPECT_GT(op_b, ForwardProfileOp::kFfn);
  EXPECT_STREQ(ForwardProfileOpRegistry::Name(op_a), "profile_metrics_test_op_a");
  EXPECT_STREQ(ForwardProfileOpRegistry::Name(op_b), "profile_metrics_test_op_b");
}

TEST(ForwardProfileOpRegistryTest, NewRegisteredOpUsableInMetrics) {
  const ForwardProfileOpId shared_expert = RegisterTestOp("profile_metrics_test_shared_expert");

  ForwardPerfMetrics metrics;
  metrics.Accumulate(shared_expert, 321);
  metrics.Accumulate(shared_expert, 9);

  const LayerPerfMetrics* op_metrics = metrics.GetOp(shared_expert);
  ASSERT_NE(op_metrics, nullptr);
  EXPECT_EQ(op_metrics->cumulative_time_ns, 330);
  EXPECT_EQ(op_metrics->count, 2);
}

TEST(ForwardPerfMetricsTest, AccumulateUpdatesOpMetrics) {
  ForwardPerfMetrics metrics;
  metrics.Accumulate(ForwardProfileOp::kAttn, 100);
  metrics.Accumulate(ForwardProfileOp::kAttn, 50);

  const LayerPerfMetrics* attn = metrics.GetOp(ForwardProfileOp::kAttn);
  ASSERT_NE(attn, nullptr);
  EXPECT_EQ(attn->cumulative_time_ns, 150);
  EXPECT_EQ(attn->count, 2);
  EXPECT_EQ(metrics.GetOp(ForwardProfileOp::kFfn), nullptr);
}

TEST(ProfileMetricsSerializationTest, SerializeEmptyProfileMetrics) {
  ProfileMetrics metrics;
  std::vector<int> buffer;
  metrics.SerializeTo(buffer);

  // schedule_forward_interval_ns(2) + forward_metrics_count(1)
  ASSERT_EQ(buffer.size(), 3u);
  EXPECT_EQ(buffer[2], 0);

  ProfileMetrics decoded;
  const size_t next_idx = ProfileMetrics::DeserializeFrom(buffer, 0, decoded);
  EXPECT_EQ(next_idx, buffer.size());
  EXPECT_EQ(decoded.schedule_forward_interval_ns, 0);
  EXPECT_TRUE(decoded.forward_metrics.empty());
}

TEST(ProfileMetricsSerializationTest, SerializeAndDeserializeMainAndSpeculative) {
  const ProfileMetrics original = MakeSampleProfileMetrics();

  std::vector<int> buffer;
  original.SerializeTo(buffer);
  ASSERT_FALSE(buffer.empty());

  ProfileMetrics decoded;
  const size_t next_idx = ProfileMetrics::DeserializeFrom(buffer, 0, decoded);
  EXPECT_EQ(next_idx, buffer.size());
  ExpectMetricsEqual(original, decoded);
}

TEST(ProfileMetricsSerializationTest, DeserializeReplacesExistingData) {
  const ProfileMetrics original = MakeSampleProfileMetrics();
  std::vector<int> buffer;
  original.SerializeTo(buffer);

  ProfileMetrics decoded;
  decoded.schedule_forward_interval_ns = 999;
  decoded.GetOrCreate(ForwardMetricsKind::kMain).forward_time_ns = 888;
  decoded.GetOrCreate(ForwardMetricsKind::kMain).GetOrCreateOp(ForwardProfileOp::kAttn).count = 777;

  const size_t next_idx = ProfileMetrics::DeserializeFrom(buffer, 0, decoded);
  EXPECT_EQ(next_idx, buffer.size());
  ExpectMetricsEqual(original, decoded);
}

TEST(ProfileMetricsSerializationTest, RoundTripMultipleTimes) {
  ProfileMetrics metrics = MakeSampleProfileMetrics();

  for (int i = 0; i < 3; ++i) {
    std::vector<int> buffer;
    metrics.SerializeTo(buffer);

    ProfileMetrics decoded;
    const size_t next_idx = ProfileMetrics::DeserializeFrom(buffer, 0, decoded);
    EXPECT_EQ(next_idx, buffer.size());
    ExpectMetricsEqual(metrics, decoded);
    metrics = decoded;
  }
}

TEST(ProfileMetricsSerializationTest, RegisteredCustomOpSurvivesCodec) {
  const ForwardProfileOpId custom_op = RegisterTestOp("profile_metrics_test_codec_custom");

  ProfileMetrics original;
  ForwardPerfMetrics& main = original.GetOrCreate(ForwardMetricsKind::kMain);
  main.GetOrCreateOp(custom_op).cumulative_time_ns = 4096;
  main.GetOrCreateOp(custom_op).count = 7;

  std::vector<int> buffer;
  original.SerializeTo(buffer);

  ProfileMetrics decoded;
  ASSERT_EQ(ProfileMetrics::DeserializeFrom(buffer, 0, decoded), buffer.size());

  const ForwardPerfMetrics* decoded_main = decoded.Get(ForwardMetricsKind::kMain);
  ASSERT_NE(decoded_main, nullptr);
  const LayerPerfMetrics* decoded_custom = decoded_main->GetOp(custom_op);
  ASSERT_NE(decoded_custom, nullptr);
  EXPECT_EQ(decoded_custom->cumulative_time_ns, 4096);
  EXPECT_EQ(decoded_custom->count, 7);
  EXPECT_STREQ(ForwardProfileOpRegistry::Name(custom_op), "profile_metrics_test_codec_custom");
}

TEST(ProfileMetricsSerializationTest, WireFormatCarriesOpName) {
  const ForwardProfileOpId custom_op = RegisterTestOp("profile_metrics_test_wire_name");

  ProfileMetrics original;
  original.GetOrCreate(ForwardMetricsKind::kMain).GetOrCreateOp(custom_op).cumulative_time_ns = 1;

  std::vector<int> buffer;
  original.SerializeTo(buffer);

  const char* expected_name = "profile_metrics_test_wire_name";
  bool found_name = false;
  for (size_t i = 0; i + static_cast<size_t>(std::strlen(expected_name)) < buffer.size(); ++i) {
    if (buffer[i] == static_cast<int>(std::strlen(expected_name))) {
      bool match = true;
      for (size_t j = 0; j < std::strlen(expected_name); ++j) {
        if (buffer[i + 1 + j] != static_cast<int>(static_cast<unsigned char>(expected_name[j]))) {
          match = false;
          break;
        }
      }
      if (match) {
        found_name = true;
        break;
      }
    }
  }
  EXPECT_TRUE(found_name);
}

TEST(ProfileMetricsSerializationTest, DeserializeAlignsRegistryFromWireName) {
  const ForwardProfileOpId custom_op = RegisterTestOp("profile_metrics_test_align_registry");
  const char* custom_name = ForwardProfileOpRegistry::Name(custom_op);
  ASSERT_NE(custom_op, ForwardProfileOp::kInvalid);

  ProfileMetrics original;
  original.GetOrCreate(ForwardMetricsKind::kMain).GetOrCreateOp(custom_op).cumulative_time_ns = 512;
  original.GetOrCreate(ForwardMetricsKind::kMain).GetOrCreateOp(custom_op).count = 2;

  std::vector<int> buffer;
  original.SerializeTo(buffer);

  ProfileMetrics decoded;
  ASSERT_EQ(ProfileMetrics::DeserializeFrom(buffer, 0, decoded), buffer.size());
  EXPECT_STREQ(ForwardProfileOpRegistry::Name(custom_op), custom_name);
}

TEST(ProfileMetricsSerializationTest, DeserializeOutOfOrderOpsAlignsRegistry) {
  const size_t base = ForwardProfileOpRegistry::RegisteredCount();
  const ForwardProfileOpId op_a_id = static_cast<ForwardProfileOpId>(base);
  const ForwardProfileOpId op_b_id = static_cast<ForwardProfileOpId>(base + 1);
  const char* name_a = "profile_metrics_test_out_of_order_a";
  const char* name_b = "profile_metrics_test_out_of_order_b";

  std::vector<int> buffer;
  AppendI64(0, buffer);
  buffer.push_back(1);
  buffer.push_back(static_cast<int>(ForwardMetricsKind::kMain));
  AppendI64(1000, buffer);
  buffer.push_back(2);
  // 较大 op_id 先到，会触发 gap 填充 unknown；较小 op_id 后到时应覆盖占位符。
  AppendOpWire(op_b_id, name_b, 300, 3, buffer);
  AppendOpWire(op_a_id, name_a, 200, 2, buffer);

  ProfileMetrics decoded;
  ASSERT_EQ(ProfileMetrics::DeserializeFrom(buffer, 0, decoded), buffer.size());

  EXPECT_STREQ(ForwardProfileOpRegistry::Name(op_a_id), name_a);
  EXPECT_STREQ(ForwardProfileOpRegistry::Name(op_b_id), name_b);

  const ForwardPerfMetrics* main = decoded.Get(ForwardMetricsKind::kMain);
  ASSERT_NE(main, nullptr);
  EXPECT_EQ(main->forward_time_ns, 1000);

  const LayerPerfMetrics* decoded_a = main->GetOp(op_a_id);
  const LayerPerfMetrics* decoded_b = main->GetOp(op_b_id);
  ASSERT_NE(decoded_a, nullptr);
  ASSERT_NE(decoded_b, nullptr);
  EXPECT_EQ(decoded_a->cumulative_time_ns, 200);
  EXPECT_EQ(decoded_a->count, 2);
  EXPECT_EQ(decoded_b->cumulative_time_ns, 300);
  EXPECT_EQ(decoded_b->count, 3);
}

TEST(ModelPerfMacroTest, EndNoSyncAccumulatesRegisteredOp) {
  const ForwardProfileOpId custom_op = RegisterTestOp("profile_metrics_test_nosync_op");
  ForwardPerfMetrics metrics;

  MODEL_PERF_OP_BEGIN(&metrics, timer);
  ModelPerfOpEndNoSync(&metrics, custom_op, timer);

  const LayerPerfMetrics* op_metrics = metrics.GetOp(custom_op);
  ASSERT_NE(op_metrics, nullptr);
  EXPECT_EQ(op_metrics->count, 1);
  EXPECT_GE(op_metrics->cumulative_time_ns, 0);
}

TEST(ModelPerfMacroTest, EndNoSyncSkipsWhenMetricsNull) {
  MODEL_PERF_OP_BEGIN(nullptr, timer);
  MODEL_PERF_OP_END_NOSYNC(nullptr, ForwardProfileOp::kAttn, timer);
  SUCCEED();
}

#ifdef ENABLE_CUDA
TEST(ModelPerfMacroTest, EndStreamIgnoresInvalidOpId) {
  SetDevice(0);
  Stream stream;

  ForwardPerfMetrics metrics;
  MODEL_PERF_OP_BEGIN(&metrics, timer);
  MODEL_PERF_OP_END_STREAM(&metrics, ForwardProfileOp::kInvalid, timer, stream);

  EXPECT_TRUE(metrics.ops.empty());
  stream.Destroy();
}

TEST(ModelPerfMacroTest, EndStreamAccumulatesRegisteredOp) {
  SetDevice(0);
  Stream stream;

  const ForwardProfileOpId shared_expert = RegisterTestOp("profile_metrics_test_end_stream_op");
  ForwardPerfMetrics metrics;

  MODEL_PERF_OP_BEGIN(&metrics, timer);
  MODEL_PERF_OP_END_STREAM(&metrics, shared_expert, timer, stream);

  const LayerPerfMetrics* op_metrics = metrics.GetOp(shared_expert);
  ASSERT_NE(op_metrics, nullptr);
  EXPECT_EQ(op_metrics->count, 1);
  EXPECT_GE(op_metrics->cumulative_time_ns, 0);

  stream.Destroy();
}

TEST(ModelPerfMacroTest, EndStreamSkipsWhenMetricsNull) {
  SetDevice(0);
  Stream stream;

  MODEL_PERF_OP_BEGIN(nullptr, timer);
  MODEL_PERF_OP_END_STREAM(nullptr, ForwardProfileOp::kAttn, timer, stream);

  stream.Destroy();
  SUCCEED();
}

TEST(ModelPerfMacroTest, EndStreamHelperMatchesMacro) {
  SetDevice(0);
  Stream stream;

  const ForwardProfileOpId custom_op = RegisterTestOp("profile_metrics_test_end_stream_helper");
  ForwardPerfMetrics metrics_by_macro;
  ForwardPerfMetrics metrics_by_helper;

  MODEL_PERF_OP_BEGIN(&metrics_by_macro, macro_timer);
  MODEL_PERF_OP_END_STREAM(&metrics_by_macro, custom_op, macro_timer, stream);

  MODEL_PERF_OP_BEGIN(&metrics_by_helper, helper_timer);
  ModelPerfOpEndStream(&metrics_by_helper, custom_op, helper_timer, stream);

  const LayerPerfMetrics* macro_metrics = metrics_by_macro.GetOp(custom_op);
  const LayerPerfMetrics* helper_metrics = metrics_by_helper.GetOp(custom_op);
  ASSERT_NE(macro_metrics, nullptr);
  ASSERT_NE(helper_metrics, nullptr);
  EXPECT_EQ(macro_metrics->count, helper_metrics->count);
  EXPECT_EQ(macro_metrics->count, 1);

  stream.Destroy();
}
#endif

}  // namespace ksana_llm
