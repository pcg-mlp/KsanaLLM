/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/core/ipc_process_watchdog.h"

#include <signal.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include "utils/logger.h"

namespace ksana_llm {

static std::vector<int64_t> s_children_pids;

void SignalHandlerWhenParentExit(int signum, siginfo_t* info, void* context) {
  pid_t parent_pid = info->si_pid;
  KLLM_LOG_ERROR << fmt::format("Process {} terminated by parent process {} with sigal {} ({}).", getpid(), parent_pid,
                                signum, strsignal(signum));

  // Force exit now, change to graceful exiting in the future.
  _exit(128 + signum);
}

void SignalHandlerWhenChildExit(int signum, siginfo_t* info, void* context) {
  pid_t child_pid = info->si_pid;
  // If s_children_pids is empty, that means, executors is not regestered yet.
  if (s_children_pids.empty() ||
      std::find(s_children_pids.begin(), s_children_pids.end(), child_pid) != std::end(s_children_pids)) {
    KLLM_LOG_ERROR << fmt::format("Process {} terminated by child process {} with sigal {} ({}).", getpid(), child_pid,
                                  signum, strsignal(signum));

    // Force exit now, change to graceful exiting in the future.
    _exit(128 + signum);
  } else {
    int status;
    while (waitpid(-1, &status, WNOHANG) > 0) {
    }
  }
}

void SetupSignalHandler(int signum, void (*handler)(int, siginfo_t*, void*)) {
  struct sigaction sa;

  memset(&sa, 0, sizeof(sa));
  sa.sa_sigaction = handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART | SA_SIGINFO;

  // Mask some signal.
  sigaddset(&sa.sa_mask, SIGQUIT);
  sigaddset(&sa.sa_mask, SIGHUP);

  if (sigaction(signum, &sa, nullptr) < 0) {
    KLLM_LOG_ERROR << "SetupSignalHandler error, sigaction failed.";
    _exit(EXIT_FAILURE);
  }
}

void SetMonitoredChildPids(const std::vector<int64_t>& pids) { s_children_pids = pids; }

void MonitorParent() {
  // Receive SIGTERM if parent exited.
  if (prctl(PR_SET_PDEATHSIG, SIGTERM) == -1) {
    KLLM_LOG_ERROR << "MonitorParent error, prctl failed.";
    _exit(EXIT_FAILURE);
  }

  // If parent already exited.
  if (getppid() == 1) {
    _exit(EXIT_FAILURE);
  }

  SetupSignalHandler(SIGTERM, SignalHandlerWhenParentExit);
}

void MonitorChildren() {
  // Receive SIGCHLD if any child process exited.
  SetupSignalHandler(SIGCHLD, SignalHandlerWhenChildExit);
}

}  // namespace ksana_llm
