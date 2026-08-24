/* Copyright 2026 Tencent Inc.  All rights reserved.

===============================================================================*/
// Process utilities for the humming JIT compiler.
//
// Replaces the previous popen("/bin/sh -c ... 2>&1") approach which:
//   1. doubled the process count (one /bin/sh wrapper per child),
//   2. merged stderr into the stdout JSON stream parsed by QueryKernelConfigs,
//   3. could block forever in pclose (holding the config-dir flock) when a
//      child hung, stalling all sibling ranks waiting on the same flock.
#pragma once

#include <string>
#include <vector>

namespace llm_kernels {
namespace nvidia {
namespace humming {

struct CommandResult {
  /// false when the process could not be spawned (binary not found, pipe fail).
  bool spawned = false;
  /// true when the process group was SIGKILLed after exceeding the timeout.
  bool timed_out = false;
  /// WEXITSTATUS when the process exited normally; -WTERMSIG when killed by a
  /// signal. Undefined when spawned == false.
  int exit_code = -1;
  std::string stdout_data;
  std::string stderr_data;

  /// Process ran to completion and returned 0.
  bool Ok() const { return spawned && !timed_out && exit_code == 0; }
};

/// Execute argv[0] with argv[1..] directly (no shell wrapper).
///
/// - stdout/stderr captured separately (concurrent poll drain: children writing
///   more than the pipe buffer to both streams cannot deadlock).
/// - Child runs in its own process group; on timeout the whole group
///   (including grandchildren, e.g. nvcc's cicc/ptxas) is SIGKILLed and reaped.
/// - stdin is redirected from /dev/null.
/// - timeout_sec must be > 0.
CommandResult ExecuteCommand(const std::vector<std::string>& argv, int timeout_sec);

/// Create path plus missing parents (mkdir -p semantics) via mkdir(2).
/// Returns true when the directory already exists; false when a path component
/// exists as a non-directory or creation fails (errno holds the last error).
///
/// Replaces the previous one-shell-process-per-call "mkdir -p" (~1500 spawned
/// shells during an 8-rank cold start).
bool MkdirP(const std::string& path);

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
