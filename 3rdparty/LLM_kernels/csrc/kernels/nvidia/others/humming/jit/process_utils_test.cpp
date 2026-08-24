/* Copyright 2026 Tencent Inc.  All rights reserved.

===============================================================================*/
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/others/humming/jit/process_utils.h"

using llm_kernels::nvidia::humming::CommandResult;
using llm_kernels::nvidia::humming::ExecuteCommand;
using llm_kernels::nvidia::humming::MkdirP;

TEST(ExecuteCommandTest, CapturesStdoutStderrAndExitCodeSeparately) {
  CommandResult r = ExecuteCommand({"bash", "-c", "echo out; echo err >&2; exit 3"}, 30);
  ASSERT_TRUE(r.spawned);
  EXPECT_FALSE(r.timed_out);
  EXPECT_EQ(r.exit_code, 3);
  EXPECT_FALSE(r.Ok());
  EXPECT_EQ(r.stdout_data, "out\n");
  EXPECT_EQ(r.stderr_data, "err\n");
}

TEST(ExecuteCommandTest, OkOnZeroExit) { EXPECT_TRUE(ExecuteCommand({"true"}, 30).Ok()); }

TEST(ExecuteCommandTest, SpawnFailureOnMissingBinary) {
  CommandResult r = ExecuteCommand({"/nonexistent/binary/xyz"}, 30);
  EXPECT_FALSE(r.spawned);
  EXPECT_FALSE(r.Ok());
}

TEST(ExecuteCommandTest, TimeoutKillsProcessGroup) {
  // Two sleeps: one backgrounded (same process group) — both must be killed.
  auto t0 = std::chrono::steady_clock::now();
  CommandResult r = ExecuteCommand({"bash", "-c", "sleep 500 & sleep 500"}, 2);
  auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t0).count();
  EXPECT_TRUE(r.spawned);
  EXPECT_TRUE(r.timed_out);
  EXPECT_FALSE(r.Ok());
  EXPECT_LT(elapsed_s, 60);  // returned promptly after the 2s deadline
}

TEST(ExecuteCommandTest, SignalDeathReportsNegativeSignal) {
  CommandResult r = ExecuteCommand({"bash", "-c", "kill -TERM $$"}, 30);
  ASSERT_TRUE(r.spawned);
  EXPECT_FALSE(r.timed_out);
  EXPECT_EQ(r.exit_code, -SIGTERM);
  EXPECT_FALSE(r.Ok());
}

TEST(ExecuteCommandTest, LargeConcurrentOutputDoesNotDeadlock) {
  // 200 KB on BOTH streams (pipe buffer is 64 KB) — sequential draining would
  // deadlock; the poll loop must read both concurrently.
  CommandResult r = ExecuteCommand(
      {"bash", "-c", "head -c 200000 /dev/zero | tr '\\0' 'a'; head -c 200000 /dev/zero | tr '\\0' 'b' >&2"}, 60);
  EXPECT_TRUE(r.Ok());
  EXPECT_EQ(r.stdout_data.size(), 200000);
  EXPECT_EQ(r.stderr_data.size(), 200000);
}

TEST(MkdirPTest, CreatesNestedAndToleratesExisting) {
  std::string base = "/tmp/humming_mkdirp_test_" + std::to_string(getpid());
  std::string deep = base + "/a/b/c";
  ASSERT_TRUE(MkdirP(deep));
  EXPECT_TRUE(MkdirP(deep));  // idempotent
  struct stat st;
  ASSERT_EQ(stat(deep.c_str(), &st), 0);
  EXPECT_TRUE(S_ISDIR(st.st_mode));
}

TEST(MkdirPTest, FailsWhenComponentIsRegularFile) {
  std::string base = "/tmp/humming_mkdirp_file_" + std::to_string(getpid());
  ASSERT_TRUE(MkdirP(base));
  std::string file = base + "/regular";
  {
    std::ofstream f(file);
    f << "x";
  }
  EXPECT_FALSE(MkdirP(file + "/sub"));
}
