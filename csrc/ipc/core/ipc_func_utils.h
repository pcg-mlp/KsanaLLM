/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <memory>

#include "ipc/core/ipc_func_types.h"
#include "utils/waiter.h"

namespace ksana_llm {

class IpcWaiter {
 public:
  // Wait until all executors finished.
  void Wait() { waiter->Wait(); }

  // Reset status and waiter;
  void Reset(size_t wait_num) {
    if (!waiter) {
      waiter = std::make_shared<Waiter>(wait_num);
    } else {
      waiter->Reset(wait_num);
    }
  }

  // Notify the waiter.
  void Notify() {
    if (waiter) {
      waiter->Notify();
    }
  }

  // Wait all executor finished.
  std::shared_ptr<Waiter> waiter = nullptr;
};

void SetIpcContext(std::shared_ptr<IpcContext> context, int func_id, bool is_async, bool is_oneway,
                   bool share_callable = false, bool share_callback = false, bool is_heavy = false);

std::shared_ptr<IpcContext> CreateBroadcastContext(int func_id, bool is_async, bool is_oneway,
                                                   bool share_callable = false, bool share_callback = false,
                                                   bool is_heavy = false);

std::shared_ptr<IpcContext> CreateEngine2ExecutorContext(int func_id, int executor_id, bool is_async, bool is_oneway,
                                                         bool share_callable = false, bool share_callback = false,
                                                         bool is_heavy = false);

std::shared_ptr<IpcContext> CreateExecutor2EngineContext(int func_id, int executor_id, bool is_async, bool is_oneway,
                                                         bool share_callable = false, bool share_callback = false,
                                                         bool is_heavy = false);

}  // namespace ksana_llm
