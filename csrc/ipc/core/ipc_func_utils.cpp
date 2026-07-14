/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/core/ipc_func_utils.h"

namespace ksana_llm {

void SetIpcContext(std::shared_ptr<IpcContext> context, int func_id, bool is_async, bool is_oneway,
                   bool share_callable /* = false*/, bool share_callback /* = false*/, bool is_heavy /* = false*/) {
  context->func_id = func_id;

  // Force async if in oneway mode.
  context->is_async = is_oneway ? true : is_async;
  context->is_oneway = is_oneway;
  context->share_callable = share_callable;
  context->share_callback = share_callback;
  context->is_heavy = is_heavy;
}

std::shared_ptr<IpcContext> CreateBroadcastContext(int func_id, bool is_async, bool is_oneway,
                                                   bool share_callable /* = false*/, bool share_callback /* = false*/,
                                                   bool is_heavy /* = false*/) {
  std::shared_ptr<IpcContext> context = std::make_shared<IpcContext>();

  // For share callable if in broadcast mode.
  SetIpcContext(context, func_id, is_async, is_oneway, true, share_callback, is_heavy);
  context->from_rank = 0;
  context->to_rank = -1;

  return context;
}

std::shared_ptr<IpcContext> CreateEngine2ExecutorContext(int func_id, int executor_id, bool is_async, bool is_oneway,
                                                         bool share_callable /* = false*/,
                                                         bool share_callback /* = false*/, bool is_heavy /* = false*/) {
  std::shared_ptr<IpcContext> context = std::make_shared<IpcContext>();
  SetIpcContext(context, func_id, is_async, is_oneway, share_callable, share_callback, is_heavy);

  context->from_rank = 0;
  context->to_rank = executor_id;

  return context;
}

std::shared_ptr<IpcContext> CreateExecutor2EngineContext(int func_id, int executor_id, bool is_async, bool is_oneway,
                                                         bool share_callable /* = false*/,
                                                         bool share_callback /* = false*/, bool is_heavy /* = false*/) {
  std::shared_ptr<IpcContext> context = std::make_shared<IpcContext>();
  SetIpcContext(context, func_id, is_async, is_oneway, share_callable, share_callback, is_heavy);
  context->from_rank = executor_id;
  context->to_rank = 0;

  return context;
}

}  // namespace ksana_llm
