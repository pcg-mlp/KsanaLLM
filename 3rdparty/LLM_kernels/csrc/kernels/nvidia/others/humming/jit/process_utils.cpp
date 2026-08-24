/* Copyright 2026 Tencent Inc.  All rights reserved.

===============================================================================*/
#include "csrc/kernels/nvidia/others/humming/jit/process_utils.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>

extern char** environ;

namespace llm_kernels {
namespace nvidia {
namespace humming {

namespace {

/// Read everything currently available on fd (must be O_NONBLOCK).
/// Sets *eof when read() returns 0.
void DrainFd(int fd, std::string* out, bool* eof) {
  char buf[65536];
  for (;;) {
    ssize_t n = read(fd, buf, sizeof(buf));
    if (n > 0) {
      out->append(buf, static_cast<size_t>(n));
      continue;
    }
    if (n == 0) {
      *eof = true;
      return;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) return;
    if (errno == EINTR) continue;
    *eof = true;  // unrecoverable error: treat as EOF so the loop can finish
    return;
  }
}

void SetNonBlocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags >= 0) fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

}  // namespace

CommandResult ExecuteCommand(const std::vector<std::string>& argv, int timeout_sec) {
  CommandResult result;
  if (argv.empty() || timeout_sec <= 0) {
    result.stderr_data = "ExecuteCommand: empty argv or non-positive timeout";
    return result;
  }

  std::vector<char*> cargv;
  cargv.reserve(argv.size() + 1);
  for (const auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
  cargv.push_back(nullptr);

  int out_pipe[2], err_pipe[2];
  if (pipe(out_pipe) != 0) {
    result.stderr_data = std::string("pipe() failed: ") + strerror(errno);
    return result;
  }
  if (pipe(err_pipe) != 0) {
    close(out_pipe[0]);
    close(out_pipe[1]);
    result.stderr_data = std::string("pipe() failed: ") + strerror(errno);
    return result;
  }

  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, out_pipe[1], STDOUT_FILENO);
  posix_spawn_file_actions_adddup2(&actions, err_pipe[1], STDERR_FILENO);
  posix_spawn_file_actions_addclose(&actions, out_pipe[0]);
  posix_spawn_file_actions_addclose(&actions, err_pipe[0]);
  posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);

  posix_spawnattr_t attr;
  posix_spawnattr_init(&attr);
  // Own process group (pgid == child pid): on timeout the whole group —
  // including grandchildren like nvcc's cicc/ptxas — can be killed at once.
  posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
  posix_spawnattr_setpgroup(&attr, 0);

  pid_t pid = -1;
  // posix_spawnp (not posix_spawn): argv[0] may be a bare name like "nvcc",
  // which must be resolved through PATH (execvp semantics). posix_spawn only
  // accepts pathnames and returns ENOENT for slash-less names.
  int rc = posix_spawnp(&pid, cargv[0], &actions, &attr, cargv.data(), environ);

  posix_spawn_file_actions_destroy(&actions);
  posix_spawnattr_destroy(&attr);
  close(out_pipe[1]);
  close(err_pipe[1]);

  if (rc != 0) {
    close(out_pipe[0]);
    close(err_pipe[0]);
    result.stderr_data = std::string("posix_spawnp failed: ") + strerror(rc);
    return result;
  }
  result.spawned = true;

  SetNonBlocking(out_pipe[0]);
  SetNonBlocking(err_pipe[0]);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
  bool out_eof = false, err_eof = false;
  while (!out_eof || !err_eof) {
    if (!result.timed_out && std::chrono::steady_clock::now() >= deadline) {
      result.timed_out = true;
      kill(-pid, SIGKILL);  // whole process group
      kill(pid, SIGKILL);   // group leader fallback
    }
    struct pollfd fds[2];
    nfds_t n = 0;
    int out_idx = -1, err_idx = -1;
    if (!out_eof) {
      fds[n] = {out_pipe[0], POLLIN, 0};
      out_idx = static_cast<int>(n++);
    }
    if (!err_eof) {
      fds[n] = {err_pipe[0], POLLIN, 0};
      err_idx = static_cast<int>(n++);
    }
    int pr = poll(fds, n, 50);
    if (pr < 0 && errno != EINTR) break;
    if (out_idx >= 0 && fds[out_idx].revents != 0) DrainFd(out_pipe[0], &result.stdout_data, &out_eof);
    if (err_idx >= 0 && fds[err_idx].revents != 0) DrainFd(err_pipe[0], &result.stderr_data, &err_eof);
    // After a timeout kill, keep draining until both pipes hit EOF so the
    // reap below cannot race a still-dying group member.
  }

  close(out_pipe[0]);
  close(err_pipe[0]);

  int status = 0;
  pid_t wr;
  while ((wr = waitpid(pid, &status, 0)) < 0 && errno == EINTR) {
  }
  if (wr == pid) {
    if (WIFEXITED(status)) {
      result.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
      result.exit_code = -WTERMSIG(status);
    }
  }
  return result;
}

bool MkdirP(const std::string& path) {
  if (path.empty()) return false;
  std::string cur;
  size_t i = 0;
  if (path[0] == '/') {
    cur = "/";
    i = 1;
  }
  while (i <= path.size()) {
    size_t next = path.find('/', i);
    if (next == std::string::npos) next = path.size();
    if (next > i) {
      cur += path.substr(i, next - i);
      if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
      cur += "/";
    }
    i = next + 1;
  }
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
