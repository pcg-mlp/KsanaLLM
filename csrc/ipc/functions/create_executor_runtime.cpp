/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/create_executor_runtime.h"

#include "ipc/core/ipc_func_serialization.h"
#include "utils/logger.h"
#include "utils/memory_allocator.h"
#include "utils/ret_code.h"
#include "utils/singleton.h"
#include "utils/status.h"

namespace ksana_llm {

Status CreateExecutorRuntimeRequest::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);

  // Serialize vocab_info
  SerializePod<bool>(vocab_info.initialized, ptr, *bytes);
  if (vocab_info.initialized) {
    SerializePod<int>(vocab_info.vocab_size, ptr, *bytes);
    SerializePod<size_t>(vocab_info.vocab.size(), ptr, *bytes);
    for (const auto& word : vocab_info.vocab) {
      SerializeString(word, ptr, *bytes);
    }
    SerializePod<size_t>(vocab_info.stop_token_ids.size(), ptr, *bytes);
    for (int token_id : vocab_info.stop_token_ids) {
      SerializePod<int>(token_id, ptr, *bytes);
    }
    SerializePod<int>(vocab_info.vocab_type, ptr, *bytes);
    SerializePod<bool>(vocab_info.add_prefix_space, ptr, *bytes);
    SerializePod<int>(vocab_info.think_end_token_id, ptr, *bytes);
  }

  return Status();
}

Status CreateExecutorRuntimeRequest::Deserialize(const void* ptr) {
  size_t offset = 0;

  // Deserialize vocab_info
  vocab_info.initialized = DeserializePod<bool>(ptr, offset);
  if (vocab_info.initialized) {
    vocab_info.vocab_size = DeserializePod<int>(ptr, offset);
    size_t vocab_count = DeserializePod<size_t>(ptr, offset);
    vocab_info.vocab.resize(vocab_count);
    for (size_t i = 0; i < vocab_count; ++i) {
      vocab_info.vocab[i] = DeserializeString(ptr, offset);
    }
    size_t stop_token_count = DeserializePod<size_t>(ptr, offset);
    vocab_info.stop_token_ids.resize(stop_token_count);
    for (size_t i = 0; i < stop_token_count; ++i) {
      vocab_info.stop_token_ids[i] = DeserializePod<int>(ptr, offset);
    }
    vocab_info.vocab_type = DeserializePod<int>(ptr, offset);
    vocab_info.add_prefix_space = DeserializePod<bool>(ptr, offset);
    vocab_info.think_end_token_id = DeserializePod<int>(ptr, offset);
  }

  return Status();
}

Status CreateExecutorRuntimeResponse::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeStatus(result_status, ptr, *bytes);
  return Status();
}

Status CreateExecutorRuntimeResponse::Deserialize(const void* ptr) {
  size_t offset = 0;
  result_status = DeserializeStatus(ptr, offset);
  return Status();
}

void CreateExecutorRuntimeCallable::SetExecutorRuntime(std::shared_ptr<ExecutorRuntime> executor_runtime) {
  executor_runtime_ = executor_runtime;
}

void CreateExecutorRuntimeCallable::SetEnvironment(std::shared_ptr<Environment> env) { env_ = env; }

Status CreateExecutorRuntimeCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  // Get vocab info from request and set to Environment
  const CreateExecutorRuntimeRequest* req = reinterpret_cast<const CreateExecutorRuntimeRequest*>(request);
  if (req->vocab_info.initialized && env_) {
    KLLM_LOG_INFO << "Vocab info received via IPC, vocab_size: " << req->vocab_info.vocab_size
                  << ", think_end_token_id: " << req->vocab_info.think_end_token_id;
    env_->SetVocabInfo(req->vocab_info);
  }

  executor_runtime_->Initialize();
  return Status();
}

Status CreateExecutorRuntimeCallback::Call(const IpcResponse* response, IpcContext* context) {
  const CreateExecutorRuntimeResponse* rsp = reinterpret_cast<const CreateExecutorRuntimeResponse*>(response);
  if (!rsp->result_status.OK()) {
    KLLM_LOG_ERROR << "Create executor runtime error: " << rsp->result_status.GetMessage();
    create_status = rsp->result_status;
  }
  Notify();
  return Status();
}

}  // namespace ksana_llm
