/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/runtime/ipcSocket.cpp
 *
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "device/nvidia/ipc_socket.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <memory.h>
#include <nccl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include "device/nvidia/nccl_utils.h"
#include "fmt/core.h"
#include "utils/logger.h"

// Check system calls
#define SYSCHECK_SYNC(statement, name, retval)                                                 \
  do {                                                                                         \
    retval = (statement);                                                                      \
    if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) {         \
      KLLM_LOG_INFO << fmt::format("Call to {} returned {}, retrying", name, strerror(errno)); \
    } else {                                                                                   \
      break;                                                                                   \
    }                                                                                          \
  } while (true)

#define SYSCHECK(statement, name)                                                      \
  do {                                                                                 \
    int retval;                                                                        \
    SYSCHECK_SYNC((statement), name, retval);                                          \
    if (retval == -1) {                                                                \
      KLLM_LOG_WARNING << fmt::format("Call to {} failed: {}", name, strerror(errno)); \
      NCCL_CHECK(ncclSystemError);                                                     \
    }                                                                                  \
  } while (false)

namespace ksana_llm {

// Enable Linux abstract socket naming
#define USE_ABSTRACT_SOCKET
#define NCCL_IPC_SOCKNAME_LEN 64
#define NCCL_IPC_SOCKNAME_STR "/tmp/ub-socket-%d-%lx"

struct NcclIpcSocket {
  int fd;
  char socket_name[NCCL_IPC_SOCKNAME_LEN];
  uint32_t volatile* abort_flag;
  uint64_t hash;
};

/*
 * Create a Unix Domain Socket
 */
std::shared_ptr<NcclIpcSocket> NcclIpcSocketInit(int rank, uint64_t hash, uint32_t volatile* abort_flag) {
  int fd = -1;
  struct sockaddr_un cliaddr;
  char temp[NCCL_IPC_SOCKNAME_LEN] = "";
  auto handle = std::make_shared<NcclIpcSocket>();

  if (handle == NULL) {
    NCCL_CHECK(ncclInternalError);  // throws
  }

  handle->hash = hash;
  handle->fd = -1;
  handle->socket_name[0] = '\0';
  if ((fd = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
    KLLM_LOG_WARNING << fmt::format("UDS: Socket creation error : %s (%d)", strerror(errno), errno);
    NCCL_CHECK(ncclSystemError);  // throws
  }

  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;

  // Create unique name for the socket.
  int len = snprintf(temp, NCCL_IPC_SOCKNAME_LEN, NCCL_IPC_SOCKNAME_STR, rank, hash);
  if (len > static_cast<int>(sizeof(cliaddr.sun_path) - 1)) {
    KLLM_LOG_WARNING << "UDS: Cannot bind provided name to socket. Name too large";
    NCCL_CHECK(ncclInternalError);  // throws
  }
#ifndef USE_ABSTRACT_SOCKET
  unlink(temp);
#endif

  KLLM_LOG_DEBUG << fmt::format("UDS: Creating socket {}", temp);

  strncpy(cliaddr.sun_path, temp, len);
#ifdef USE_ABSTRACT_SOCKET
  cliaddr.sun_path[0] = '\0';  // Linux abstract socket trick
#endif
  if (bind(fd, (struct sockaddr*)&cliaddr, sizeof(cliaddr)) < 0) {
    KLLM_LOG_WARNING << fmt::format("UDS: Binding to socket {} failed : {} ({})", temp, strerror(errno), errno);
    close(fd);
    NCCL_CHECK(ncclSystemError);  // throws
  }

  handle->fd = fd;
  strcpy(handle->socket_name, temp);

  handle->abort_flag = abort_flag;
  // Mark socket as non-blocking
  if (handle->abort_flag) {
    int flags = fcntl(fd, F_GETFL);
    KLLM_CHECK_WITH_INFO(flags != -1, fmt::format("fcntl failed with error: {}", strerror(errno)));
    SYSCHECK(fcntl(fd, F_SETFL, flags | O_NONBLOCK), "fcntl");
  }

  return handle;
}

void NcclIpcSocketClose(std::shared_ptr<NcclIpcSocket> handle) {
  KLLM_CHECK_WITH_INFO(handle != nullptr, "handle is null.");
  if (handle->fd <= 0) {
    return;  // nothing to do
  }
#ifndef USE_ABSTRACT_SOCKET
  if (handle->socket_name[0] != '\0') {
    unlink(handle->socket_name);
  }
#endif
  close(handle->fd);
}

void NcclIpcSocketRecvMsg(std::shared_ptr<NcclIpcSocket> handle, void* hdr, int hdrLen, int* recv_fd) {
  struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
  struct iovec iov[1];

  // Union to guarantee alignment requirements for control array
  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  struct cmsghdr* cmptr;
  char dummy_buffer[1];
  int ret;

  msg.msg_control = control_un.control;
  msg.msg_controllen = sizeof(control_un.control);

  if (hdr == NULL) {
    iov[0].iov_base = static_cast<void*>(dummy_buffer);
    iov[0].iov_len = sizeof(dummy_buffer);
  } else {
    iov[0].iov_base = hdr;
    iov[0].iov_len = hdrLen;
  }

  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  while ((ret = recvmsg(handle->fd, &msg, 0)) <= 0) {
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      KLLM_LOG_WARNING << fmt::format("UDS: Receiving data over socket failed : {}", errno);
      NCCL_CHECK(ncclSystemError);  // throws
    }
    if (handle->abort_flag && *handle->abort_flag) {
      NCCL_CHECK(ncclInternalError);  // throws
    }
  }

  if (recv_fd != NULL) {
    if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) && (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
      if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
        KLLM_LOG_WARNING << "UDS: Receiving data over socket failed";
        NCCL_CHECK(ncclSystemError);  // throws
      }

      memmove(recv_fd, CMSG_DATA(cmptr), sizeof(*recv_fd));
    } else {
      KLLM_LOG_WARNING << fmt::format("UDS: Receiving data over socket {} failed", handle->socket_name);
      NCCL_CHECK(ncclSystemError);  // throws
    }
    KLLM_LOG_DEBUG << fmt::format("UDS: Got recv_fd {} from socket {}", *recv_fd, handle->socket_name);
  }
}

int NcclIpcSocketRecvFd(std::shared_ptr<NcclIpcSocket> handle) {
  int recv_fd = -1;
  NcclIpcSocketRecvMsg(handle, NULL, 0, &recv_fd);
  return recv_fd;
}

void NcclIpcSocketSendMsg(std::shared_ptr<NcclIpcSocket> handle, void* hdr, int hdrLen, int send_fd, int rank,
                          uint64_t hash) {
  struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
  struct iovec iov[1];
  char temp[NCCL_IPC_SOCKNAME_LEN];

  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  struct cmsghdr* cmptr;
  char dummy_buffer[1];
  struct sockaddr_un cliaddr;

  // Construct client address to send this shareable handle to
  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;

  int len = snprintf(temp, NCCL_IPC_SOCKNAME_LEN, NCCL_IPC_SOCKNAME_STR, rank, hash);
  if (len > static_cast<int>(sizeof(cliaddr.sun_path) - 1)) {
    KLLM_LOG_WARNING << "UDS: Cannot connect to provided name for socket. Name too large";
    NCCL_CHECK(ncclInternalError);  // throws
  }
  static_cast<void>(strncpy(cliaddr.sun_path, temp, len));

#ifdef USE_ABSTRACT_SOCKET
  cliaddr.sun_path[0] = '\0';  // Linux abstract socket trick
#endif

  KLLM_LOG_DEBUG << fmt::format("UDS: Sending hdr {} len {} to UDS socket {}", hdr, hdrLen, temp);

  if (send_fd != -1) {
    KLLM_LOG_DEBUG << fmt::format("UDS: Sending fd {} to UDS socket {}", send_fd, temp);

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;
    memmove(CMSG_DATA(cmptr), &send_fd, sizeof(send_fd));
  }

  msg.msg_name = static_cast<void*>(&cliaddr);
  msg.msg_namelen = sizeof(struct sockaddr_un);

  if (hdr == NULL) {
    iov[0].iov_base = static_cast<void*>(dummy_buffer);
    iov[0].iov_len = sizeof(dummy_buffer);
  } else {
    iov[0].iov_base = hdr;
    iov[0].iov_len = hdrLen;
  }
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;
  msg.msg_flags = 0;

  ssize_t sendResult;
  while ((sendResult = sendmsg(handle->fd, &msg, 0)) < 0) {
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      KLLM_LOG_WARNING << fmt::format("UDS: Sending data over socket {} failed : {} ({})", temp, strerror(errno),
                                      errno);
      NCCL_CHECK(ncclSystemError);  // throws
    }
    if (handle->abort_flag && *handle->abort_flag) {
      NCCL_CHECK(ncclInternalError);  // throws
    }
  }
}

void NcclIpcSocketSendFd(std::shared_ptr<NcclIpcSocket> handle, int send_fd, int rank) {
  NcclIpcSocketSendMsg(handle, NULL, 0, send_fd, rank, handle->hash);
}

}  // namespace ksana_llm
