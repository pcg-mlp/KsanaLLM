/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "ipc/core/ipc_func_proxy.h"
#include "ipc/core/ipc_func_types.h"
#include "tests/test.h"
#include "utils/logger.h"
#include "utils/ret_code.h"

namespace ksana_llm {

// ============================================================================
// Test Helper Classes and Mock Objects
// ============================================================================

// Function IDs for testing
constexpr int TEST_FUNC_SIMPLE = 1000;
constexpr int TEST_FUNC_COMPLEX = 1001;
constexpr int TEST_FUNC_ERROR = 1002;
constexpr int TEST_FUNC_BROADCAST = 1003;

// Global state for synchronization and verification
struct TestState {
  std::mutex mutex;
  std::condition_variable cv;
  bool invoked = false;
  int invoke_count = 0;
  int expected_value = 0;
  int actual_value = 0;
  std::string message;
  bool error_occurred = false;

  void Reset() {
    std::lock_guard<std::mutex> lock(mutex);
    invoked = false;
    invoke_count = 0;
    expected_value = 0;
    actual_value = 0;
    message.clear();
    error_occurred = false;
  }

  void WaitForInvocation(int expected_count = 1, int timeout_ms = 5000) {
    std::unique_lock<std::mutex> lock(mutex);
    auto result = cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                              [this, expected_count] { return invoke_count >= expected_count; });
    if (!result) {
      KLLM_LOG_WARNING << "WaitForInvocation timeout, expected: " << expected_count << ", actual: " << invoke_count;
    }
  }

  void Notify() {
    std::lock_guard<std::mutex> lock(mutex);
    invoked = true;
    invoke_count++;
    cv.notify_all();
  }
};

static TestState g_test_state;

// Simple test request
class SimpleTestRequest : public IpcRequest {
 public:
  int value = 0;
  std::string message;

  Status Serialize(void* ptr, size_t* bytes) const override {
    char* buffer = static_cast<char*>(ptr);
    size_t offset = 0;

    memcpy(buffer + offset, &value, sizeof(value));
    offset += sizeof(value);

    size_t msg_len = message.length();
    memcpy(buffer + offset, &msg_len, sizeof(msg_len));
    offset += sizeof(msg_len);

    if (msg_len > 0) {
      memcpy(buffer + offset, message.c_str(), msg_len);
      offset += msg_len;
    }

    *bytes = offset;
    return Status();
  }

  Status Deserialize(const void* ptr) override {
    const char* buffer = static_cast<const char*>(ptr);
    size_t offset = 0;

    memcpy(&value, buffer + offset, sizeof(value));
    offset += sizeof(value);

    size_t msg_len = 0;
    memcpy(&msg_len, buffer + offset, sizeof(msg_len));
    offset += sizeof(msg_len);

    if (msg_len > 0) {
      message.assign(buffer + offset, msg_len);
    }

    return Status();
  }
};

// Simple test response
class SimpleTestResponse : public IpcResponse {
 public:
  int result = 0;
  bool success = false;

  Status Serialize(void* ptr, size_t* bytes) const override {
    char* buffer = static_cast<char*>(ptr);
    size_t offset = 0;

    memcpy(buffer + offset, &result, sizeof(result));
    offset += sizeof(result);

    memcpy(buffer + offset, &success, sizeof(success));
    offset += sizeof(success);

    *bytes = offset;
    return Status();
  }

  Status Deserialize(const void* ptr) override {
    const char* buffer = static_cast<const char*>(ptr);
    size_t offset = 0;

    memcpy(&result, buffer + offset, sizeof(result));
    offset += sizeof(result);

    memcpy(&success, buffer + offset, sizeof(success));
    offset += sizeof(success);

    return Status();
  }
};

// Simple test callable
class SimpleTestCallable : public IpcCallable {
 public:
  Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override {
    const SimpleTestRequest* req = static_cast<const SimpleTestRequest*>(request);
    SimpleTestResponse* rsp = static_cast<SimpleTestResponse*>(response);

    // Process request
    rsp->result = req->value * 2;
    rsp->success = true;

    // Update test state
    {
      std::lock_guard<std::mutex> lock(g_test_state.mutex);
      g_test_state.actual_value = req->value;
      g_test_state.message = req->message;
      g_test_state.invoked = true;
      g_test_state.invoke_count++;
      g_test_state.cv.notify_all();
    }

    return Status();
  }
};

// Simple test callback
class SimpleTestCallback : public IpcCallback {
 public:
  Status Call(const IpcResponse* response, IpcContext* context) override {
    const SimpleTestResponse* rsp = static_cast<const SimpleTestResponse*>(response);

    // Update test state
    {
      std::lock_guard<std::mutex> lock(g_test_state.mutex);
      g_test_state.actual_value += rsp->result;
      g_test_state.invoked = true;
      g_test_state.invoke_count++;
      g_test_state.cv.notify_all();
    }

    return Status();
  }
};

// Error test callable that simulates errors
class ErrorTestCallable : public IpcCallable {
 public:
  Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override {
    {
      std::lock_guard<std::mutex> lock(g_test_state.mutex);
      g_test_state.error_occurred = true;
      g_test_state.invoke_count++;
      g_test_state.cv.notify_all();
    }
    return Status(RET_RUNTIME_FAILED, "Simulated error in callable");
  }
};

// Error test callback
class ErrorTestCallback : public IpcCallback {
 public:
  Status Call(const IpcResponse* response, IpcContext* context) override {
    {
      std::lock_guard<std::mutex> lock(g_test_state.mutex);
      g_test_state.error_occurred = true;
      g_test_state.invoke_count++;
      g_test_state.cv.notify_all();
    }
    return Status(RET_RUNTIME_FAILED, "Simulated error in callback");
  }
};

// ============================================================================
// Test Fixture
// ============================================================================

class IpcFuncProxyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_test_state.Reset();

    // Generate unique shared memory name for this test
    shm_name_ = "test_ipc_" + std::to_string(getpid()) + "_" + std::to_string(test_counter_++);
    setenv("KSANA_IPC_NAME", shm_name_.c_str(), 1);
  }

  void TearDown() override {
    // Cleanup shared memory
    std::string full_shm_name = "/" + shm_name_;
    shm_unlink(full_shm_name.c_str());
    unsetenv("KSANA_IPC_NAME");
  }

  // Helper function to wait for child process
  bool WaitForChild(pid_t pid, int timeout_sec = 10) {
    int status;
    int count = 0;
    while (count < timeout_sec * 10) {
      pid_t result = waitpid(pid, &status, WNOHANG);
      if (result == pid) {
        return WIFEXITED(status) && WEXITSTATUS(status) == 0;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      count++;
    }
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    return false;
  }

 protected:
  static int test_counter_;
  std::string shm_name_;
  const size_t executor_num_ = 2;
};

int IpcFuncProxyTest::test_counter_ = 0;

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(IpcFuncProxyTest, ConstructorAndInitialization) {
  // Test Engine mode construction
  EXPECT_NO_THROW({
    IpcFuncProxy engine_proxy(executor_num_, true, 0, false);
    engine_proxy.UnlinkIpcResource();
  });

  // Test Executor mode construction
  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    // Child process: Executor
    try {
      IpcFuncProxy executor_proxy(executor_num_, false, 0, false);
      executor_proxy.Destroy();
      exit(0);
    } catch (...) {
      exit(1);
    }
  } else {
    IpcFuncProxy engine_proxy(executor_num_, true, 0, false);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine_proxy.Destroy();

    // Parent process: wait for child
    EXPECT_TRUE(WaitForChild(pid));
  }
}

TEST_F(IpcFuncProxyTest, SetFuncInfo) {
  IpcFuncProxy proxy(executor_num_, true, 0, false);

  // Test setting function info
  EXPECT_NO_THROW({
    (proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
        TEST_FUNC_SIMPLE));
  });

  // Test duplicate function ID should throw
  EXPECT_THROW(
      {
        (proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
            TEST_FUNC_SIMPLE));
      },
      std::exception);

  proxy.UnlinkIpcResource();
}

TEST_F(IpcFuncProxyTest, GetIpcFuncCallable) {
  IpcFuncProxy proxy(executor_num_, true, 0, false);
  proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(TEST_FUNC_SIMPLE);

  // Test getting callable for specific executor
  auto callable = proxy.GetIpcFuncCallable<SimpleTestCallable>(TEST_FUNC_SIMPLE, 0);
  EXPECT_NE(callable, nullptr);

  // Test getting shared callable
  auto shared_callable = proxy.GetSharedIpcFuncCallable<SimpleTestCallable>(TEST_FUNC_SIMPLE);
  EXPECT_NE(shared_callable, nullptr);

  // Test getting non-existent function should throw
  EXPECT_THROW({ proxy.GetIpcFuncCallable<SimpleTestCallable>(9999, 0); }, std::exception);

  proxy.UnlinkIpcResource();
}

TEST_F(IpcFuncProxyTest, GetIpcFuncCallback) {
  IpcFuncProxy proxy(executor_num_, true, 0, false);
  proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(TEST_FUNC_SIMPLE);

  // Test getting callback for specific executor
  auto callback = proxy.GetIpcFuncCallback<SimpleTestCallback>(TEST_FUNC_SIMPLE, 0);
  EXPECT_NE(callback, nullptr);

  // Test getting shared callback
  auto shared_callback = proxy.GetSharedIpcFuncCallback<SimpleTestCallback>(TEST_FUNC_SIMPLE);
  EXPECT_NE(shared_callback, nullptr);

  proxy.UnlinkIpcResource();
}

// ============================================================================
// Synchronous Invocation Tests
// ============================================================================

TEST_F(IpcFuncProxyTest, SyncInvocationEngine2Executor) {
  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    // Child process: Executor
    try {
      IpcFuncProxy executor_proxy(executor_num_, false, 0, false);
      executor_proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
          TEST_FUNC_SIMPLE);

      // Wait for invocation
      g_test_state.WaitForInvocation(1);

      // Verify request was received
      EXPECT_EQ(g_test_state.actual_value, 42);
      EXPECT_EQ(g_test_state.message, "test message");

      executor_proxy.Destroy();
      exit(0);
    } catch (...) {
      exit(1);
    }
  } else {
    // Parent process: Engine
    IpcFuncProxy engine_proxy(executor_num_, true, 0, false);
    engine_proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
        TEST_FUNC_SIMPLE);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Prepare request
    SimpleTestRequest request;
    request.value = 42;
    request.message = "test message";

    SimpleTestResponse response;

    IpcContext context;
    context.func_id = TEST_FUNC_SIMPLE;
    context.from_rank = 0;
    context.to_rank = 0;
    context.is_async = false;
    context.is_oneway = false;

    // Invoke synchronously
    Status status = engine_proxy.Invoke(&request, &response, &context);
    EXPECT_TRUE(status.OK());
    EXPECT_EQ(response.result, 84);
    EXPECT_TRUE(response.success);

    engine_proxy.Destroy();
    EXPECT_TRUE(WaitForChild(pid));
  }
}

TEST_F(IpcFuncProxyTest, SyncInvocationExecutor2Engine) {
  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    // Child process: Executor
    try {
      IpcFuncProxy executor_proxy(executor_num_, false, 0, false);
      executor_proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
          TEST_FUNC_SIMPLE);

      std::this_thread::sleep_for(std::chrono::milliseconds(200));

      // Prepare request
      SimpleTestRequest request;
      request.value = 100;
      request.message = "executor to engine";

      SimpleTestResponse response;

      IpcContext context;
      context.func_id = TEST_FUNC_SIMPLE;
      context.from_rank = 0;
      context.to_rank = 0;
      context.is_async = false;
      context.is_oneway = false;

      // Invoke synchronously
      Status status = executor_proxy.Invoke(&request, &response, &context);
      EXPECT_TRUE(status.OK());
      EXPECT_EQ(response.result, 200);

      executor_proxy.Destroy();

      exit(0);
    } catch (...) {
      exit(1);
    }
  } else {
    // Parent process: Engine
    IpcFuncProxy engine_proxy(executor_num_, true, 0, false);
    engine_proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
        TEST_FUNC_SIMPLE);

    // Wait for invocation
    g_test_state.WaitForInvocation(1);

    EXPECT_EQ(g_test_state.actual_value, 100);
    EXPECT_EQ(g_test_state.message, "executor to engine");

    // Waiting util executor receiving response.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine_proxy.Destroy();
    EXPECT_TRUE(WaitForChild(pid));
  }
}

// ============================================================================
// Asynchronous Invocation Tests
// ============================================================================

TEST_F(IpcFuncProxyTest, AsyncInvocationEngine2Executor) {
  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    // Child process: Executor
    try {
      IpcFuncProxy executor_proxy(executor_num_, false, 0, false);
      executor_proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
          TEST_FUNC_SIMPLE);

      // Wait for invocation
      g_test_state.WaitForInvocation(1);

      EXPECT_EQ(g_test_state.actual_value, 55);

      executor_proxy.Destroy();
      exit(0);
    } catch (...) {
      exit(1);
    }
  } else {
    // Parent process: Engine
    IpcFuncProxy engine_proxy(executor_num_, true, 0);
    engine_proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
        TEST_FUNC_SIMPLE);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Prepare request
    SimpleTestRequest request;
    request.value = 55;
    request.message = "async test";

    IpcContext context;
    context.func_id = TEST_FUNC_SIMPLE;
    context.from_rank = 0;
    context.to_rank = 0;
    context.is_async = true;
    context.is_oneway = false;

    // Invoke asynchronously
    Status status = engine_proxy.Invoke(&request, nullptr, &context);
    EXPECT_TRUE(status.OK());

    // Wait for callback
    g_test_state.WaitForInvocation(2);          // 1 for callable, 1 for callback
    EXPECT_EQ(g_test_state.actual_value, 110);  // 55 * 2

    engine_proxy.Destroy();
    EXPECT_TRUE(WaitForChild(pid));
  }
}

// ============================================================================
// Oneway Invocation Tests
// ============================================================================

TEST_F(IpcFuncProxyTest, OnewayInvocation) {
  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    // Child process: Executor
    try {
      IpcFuncProxy executor_proxy(executor_num_, false, 0, false);
      executor_proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
          TEST_FUNC_SIMPLE);

      // Wait for invocation
      g_test_state.WaitForInvocation(1);

      EXPECT_EQ(g_test_state.actual_value, 77);
      EXPECT_EQ(g_test_state.message, "oneway message");

      executor_proxy.Destroy();
      exit(0);
    } catch (...) {
      exit(1);
    }
  } else {
    // Parent process: Engine
    IpcFuncProxy engine_proxy(executor_num_, true, 0, false);
    engine_proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
        TEST_FUNC_SIMPLE);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Prepare request
    SimpleTestRequest request;
    request.value = 77;
    request.message = "oneway message";

    IpcContext context;
    context.func_id = TEST_FUNC_SIMPLE;
    context.from_rank = 0;
    context.to_rank = 0;
    context.is_async = true;
    context.is_oneway = true;

    // Invoke oneway
    Status status = engine_proxy.Invoke(&request, nullptr, &context);
    EXPECT_TRUE(status.OK());

    // Give some time for message to be processed
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    engine_proxy.Destroy();
    EXPECT_TRUE(WaitForChild(pid));
  }
}

// ============================================================================
// Broadcast Tests
// ============================================================================

TEST_F(IpcFuncProxyTest, BroadcastInvocation) {
  std::vector<pid_t> child_pids;

  // Create multiple executors
  for (size_t i = 0; i < executor_num_; ++i) {
    pid_t pid = fork();
    ASSERT_NE(pid, -1);

    if (pid == 0) {
      // Child process: Executor
      try {
        // Waiting sometime for engine initialization.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        IpcFuncProxy executor_proxy(executor_num_, false, i, false);
        executor_proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
            TEST_FUNC_BROADCAST);

        // Wait for invocation
        g_test_state.WaitForInvocation(1);

        EXPECT_EQ(g_test_state.actual_value, 99);

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        executor_proxy.Destroy();
        exit(0);
      } catch (...) {
        exit(1);
      }
    } else {
      child_pids.push_back(pid);
    }
  }

  // Parent process: Engine
  IpcFuncProxy engine_proxy(executor_num_, true, 0, false);
  engine_proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
      TEST_FUNC_BROADCAST);

  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  // Prepare broadcast request
  SimpleTestRequest request;
  request.value = 99;
  request.message = "broadcast message";

  IpcContext context;
  context.func_id = TEST_FUNC_BROADCAST;
  context.from_rank = 0;
  context.to_rank = -1;  // Broadcast to all
  context.is_async = true;
  context.is_oneway = false;

  // Invoke broadcast
  Status status = engine_proxy.Invoke(&request, nullptr, &context);
  EXPECT_TRUE(status.OK());

  // Wait for all callbacks
  g_test_state.WaitForInvocation(executor_num_ * 2);  // callable + callback for each executor

  // Verify accumulated result
  EXPECT_EQ(g_test_state.actual_value, 99 * 2 * executor_num_);

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  engine_proxy.Destroy();

  // Wait for all children
  for (pid_t pid : child_pids) {
    EXPECT_TRUE(WaitForChild(pid));
  }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(IpcFuncProxyTest, InvalidRankError) {
  IpcFuncProxy proxy(executor_num_, true, 0, false);
  proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(TEST_FUNC_SIMPLE);

  SimpleTestRequest request;
  SimpleTestResponse response;

  IpcContext context;
  context.func_id = TEST_FUNC_SIMPLE;
  context.from_rank = 0;
  context.to_rank = 999;  // Invalid rank
  context.is_async = false;

  Status status = proxy.Invoke(&request, &response, &context);
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RET_INVALID_ARGUMENT);

  proxy.Destroy();
}

TEST_F(IpcFuncProxyTest, SyncBroadcastError) {
  IpcFuncProxy proxy(executor_num_, true, 0, false);
  proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(TEST_FUNC_SIMPLE);

  SimpleTestRequest request;
  SimpleTestResponse response;

  IpcContext context;
  context.func_id = TEST_FUNC_SIMPLE;
  context.from_rank = 0;
  context.to_rank = -1;      // Broadcast
  context.is_async = false;  // Sync mode not supported for broadcast

  Status status = proxy.Invoke(&request, &response, &context);
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RET_INVALID_ARGUMENT);

  proxy.Destroy();
}

TEST_F(IpcFuncProxyTest, NullResponseInSyncMode) {
  IpcFuncProxy proxy(executor_num_, true, 0, false);
  proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(TEST_FUNC_SIMPLE);

  SimpleTestRequest request;

  IpcContext context;
  context.func_id = TEST_FUNC_SIMPLE;
  context.from_rank = 0;
  context.to_rank = 0;
  context.is_async = false;

  Status status = proxy.Invoke(&request, nullptr, &context);  // nullptr response in sync mode
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RET_INVALID_ARGUMENT);

  proxy.Destroy();
}

// ============================================================================
// Shared Callable/Callback Tests
// ============================================================================

TEST_F(IpcFuncProxyTest, SharedCallableInvocation) {
  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    // Child process: Executor
    try {
      IpcFuncProxy executor_proxy(executor_num_, false, 0, false);
      executor_proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
          TEST_FUNC_SIMPLE);

      g_test_state.WaitForInvocation(1);

      executor_proxy.Destroy();
      exit(0);
    } catch (...) {
      exit(1);
    }
  } else {
    // Parent process: Engine
    IpcFuncProxy engine_proxy(executor_num_, true, 0, false);
    engine_proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
        TEST_FUNC_SIMPLE);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    SimpleTestRequest request;
    request.value = 123;

    IpcContext context;
    context.func_id = TEST_FUNC_SIMPLE;
    context.from_rank = 0;
    context.to_rank = 0;
    context.is_async = true;
    context.is_oneway = true;
    context.share_callable = true;

    Status status = engine_proxy.Invoke(&request, nullptr, &context);
    EXPECT_TRUE(status.OK());

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    engine_proxy.Destroy();
    EXPECT_TRUE(WaitForChild(pid));
  }
}

// ============================================================================
// Stress and Performance Tests
// ============================================================================

TEST_F(IpcFuncProxyTest, MultipleSequentialInvocations) {
  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    // Child process: Executor
    try {
      IpcFuncProxy executor_proxy(executor_num_, false, 0, false);
      executor_proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
          TEST_FUNC_SIMPLE);

      g_test_state.WaitForInvocation(10);

      executor_proxy.Destroy();
      exit(0);
    } catch (...) {
      exit(1);
    }
  } else {
    // Parent process: Engine
    IpcFuncProxy engine_proxy(executor_num_, true, 0, false);
    engine_proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
        TEST_FUNC_SIMPLE);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Send multiple requests
    for (int i = 0; i < 10; ++i) {
      SimpleTestRequest request;
      request.value = i;
      SimpleTestResponse response;

      IpcContext context;
      context.func_id = TEST_FUNC_SIMPLE;
      context.from_rank = 0;
      context.to_rank = 0;
      context.is_async = false;

      Status status = engine_proxy.Invoke(&request, &response, &context);
      EXPECT_TRUE(status.OK());
      EXPECT_EQ(response.result, i * 2);
    }

    engine_proxy.Destroy();
    EXPECT_TRUE(WaitForChild(pid));
  }
}

TEST_F(IpcFuncProxyTest, DestroyAndCleanup) {
  IpcFuncProxy* proxy = new IpcFuncProxy(executor_num_, true, 0, false);
  proxy->SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(TEST_FUNC_SIMPLE);

  // Test destroy
  EXPECT_NO_THROW({ proxy->Destroy(); });

  delete proxy;
}

// ============================================================================
// Long Task Tests
// ============================================================================

// Heavy task callable that simulates long-running operations
class HeavyTaskCallable : public IpcCallable {
 public:
  Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override {
    const SimpleTestRequest* req = static_cast<const SimpleTestRequest*>(request);
    SimpleTestResponse* rsp = static_cast<SimpleTestResponse*>(response);

    // Simulate heavy computation
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    rsp->result = req->value * 3;
    rsp->success = true;

    // Update test state
    {
      std::lock_guard<std::mutex> lock(g_test_state.mutex);
      g_test_state.actual_value = req->value;  // Use assignment instead of accumulation
      g_test_state.invoked = true;
      g_test_state.invoke_count++;
      g_test_state.cv.notify_all();
    }

    return Status();
  }
};

// Heavy task callback
class HeavyTaskCallback : public IpcCallback {
 public:
  Status Call(const IpcResponse* response, IpcContext* context) override {
    const SimpleTestResponse* rsp = static_cast<const SimpleTestResponse*>(response);

    {
      std::lock_guard<std::mutex> lock(g_test_state.mutex);
      g_test_state.actual_value += rsp->result;
      g_test_state.invoked = true;
      g_test_state.invoke_count++;
      g_test_state.cv.notify_all();
    }

    return Status();
  }
};

constexpr int TEST_FUNC_HEAVY = 2000;

TEST_F(IpcFuncProxyTest, HeavyTaskSyncInvocation) {
  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    // Child process: Executor
    try {
      IpcFuncProxy executor_proxy(executor_num_, false, 0, false);
      executor_proxy.SetFuncInfo<HeavyTaskCallable, SimpleTestRequest, SimpleTestResponse, HeavyTaskCallback>(
          TEST_FUNC_HEAVY);

      // Wait for invocation
      g_test_state.WaitForInvocation(1, 10000);

      EXPECT_EQ(g_test_state.actual_value, 50);

      executor_proxy.Destroy();
      exit(0);
    } catch (...) {
      exit(1);
    }
  } else {
    // Parent process: Engine
    IpcFuncProxy engine_proxy(executor_num_, true, 0, false);
    engine_proxy.SetFuncInfo<HeavyTaskCallable, SimpleTestRequest, SimpleTestResponse, HeavyTaskCallback>(
        TEST_FUNC_HEAVY);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Prepare heavy task request
    SimpleTestRequest request;
    request.value = 50;
    request.message = "heavy task";

    SimpleTestResponse response;

    IpcContext context;
    context.func_id = TEST_FUNC_HEAVY;
    context.from_rank = 0;
    context.to_rank = 0;
    context.is_async = false;
    context.is_oneway = false;
    context.is_heavy = true;  // Mark as heavy task

    // Invoke synchronously
    Status status = engine_proxy.Invoke(&request, &response, &context);
    EXPECT_TRUE(status.OK());
    EXPECT_EQ(response.result, 150);  // 50 * 3
    EXPECT_TRUE(response.success);

    engine_proxy.Destroy();
    EXPECT_TRUE(WaitForChild(pid));
  }
}

TEST_F(IpcFuncProxyTest, HeavyTaskAsyncInvocation) {
  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    // Child process: Executor
    try {
      IpcFuncProxy executor_proxy(executor_num_, false, 0, false);
      executor_proxy.SetFuncInfo<HeavyTaskCallable, SimpleTestRequest, SimpleTestResponse, HeavyTaskCallback>(
          TEST_FUNC_HEAVY);

      // Wait for invocation
      g_test_state.WaitForInvocation(1, 10000);

      EXPECT_EQ(g_test_state.actual_value, 60);

      executor_proxy.Destroy();
      exit(0);
    } catch (...) {
      exit(1);
    }
  } else {
    // Parent process: Engine
    IpcFuncProxy engine_proxy(executor_num_, true, 0, false);
    engine_proxy.SetFuncInfo<HeavyTaskCallable, SimpleTestRequest, SimpleTestResponse, HeavyTaskCallback>(
        TEST_FUNC_HEAVY);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Prepare heavy task request
    SimpleTestRequest request;
    request.value = 60;
    request.message = "async heavy task";

    IpcContext context;
    context.func_id = TEST_FUNC_HEAVY;
    context.from_rank = 0;
    context.to_rank = 0;
    context.is_async = true;
    context.is_oneway = false;
    context.is_heavy = true;  // Mark as heavy task

    // Invoke asynchronously
    Status status = engine_proxy.Invoke(&request, nullptr, &context);
    EXPECT_TRUE(status.OK());

    // Wait for callback (only callback runs in parent process)
    g_test_state.WaitForInvocation(1, 10000);
    EXPECT_EQ(g_test_state.actual_value, 180);  // 60*3 (only callback result in parent process)

    engine_proxy.Destroy();
    EXPECT_TRUE(WaitForChild(pid));
  }
}

TEST_F(IpcFuncProxyTest, MultipleHeavyTasksConcurrent) {
  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    // Child process: Executor
    try {
      IpcFuncProxy executor_proxy(executor_num_, false, 0, false);
      executor_proxy.SetFuncInfo<HeavyTaskCallable, SimpleTestRequest, SimpleTestResponse, HeavyTaskCallback>(
          TEST_FUNC_HEAVY);

      // Wait for multiple invocations
      g_test_state.WaitForInvocation(5, 10000);

      // Verify all tasks were executed
      EXPECT_EQ(g_test_state.actual_value, 10 + 20 + 30 + 40 + 50);

      executor_proxy.Destroy();
      exit(0);
    } catch (...) {
      exit(1);
    }
  } else {
    // Parent process: Engine
    IpcFuncProxy engine_proxy(executor_num_, true, 0, false);
    engine_proxy.SetFuncInfo<HeavyTaskCallable, SimpleTestRequest, SimpleTestResponse, HeavyTaskCallback>(
        TEST_FUNC_HEAVY);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Send multiple heavy tasks
    std::vector<int> values = {10, 20, 30, 40, 50};
    for (int val : values) {
      SimpleTestRequest request;
      request.value = val;
      SimpleTestResponse response;

      IpcContext context;
      context.func_id = TEST_FUNC_HEAVY;
      context.from_rank = 0;
      context.to_rank = 0;
      context.is_async = false;
      context.is_oneway = false;
      context.is_heavy = true;

      Status status = engine_proxy.Invoke(&request, &response, &context);
      EXPECT_TRUE(status.OK());
      EXPECT_EQ(response.result, val * 3);
    }

    engine_proxy.Destroy();
    EXPECT_TRUE(WaitForChild(pid));
  }
}

TEST_F(IpcFuncProxyTest, MixedHeavyAndNormalTasks) {
  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    // Child process: Executor
    try {
      IpcFuncProxy executor_proxy(executor_num_, false, 0, false);
      executor_proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
          TEST_FUNC_SIMPLE);
      executor_proxy.SetFuncInfo<HeavyTaskCallable, SimpleTestRequest, SimpleTestResponse, HeavyTaskCallback>(
          TEST_FUNC_HEAVY);

      // Wait for both normal and heavy tasks
      g_test_state.WaitForInvocation(6, 10000);  // 3 normal + 3 heavy

      executor_proxy.Destroy();
      exit(0);
    } catch (...) {
      exit(1);
    }
  } else {
    // Parent process: Engine
    IpcFuncProxy engine_proxy(executor_num_, true, 0, false);
    engine_proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
        TEST_FUNC_SIMPLE);
    engine_proxy.SetFuncInfo<HeavyTaskCallable, SimpleTestRequest, SimpleTestResponse, HeavyTaskCallback>(
        TEST_FUNC_HEAVY);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Send mixed tasks
    for (int i = 0; i < 3; ++i) {
      // Normal task
      {
        SimpleTestRequest request;
        request.value = i;
        SimpleTestResponse response;

        IpcContext context;
        context.func_id = TEST_FUNC_SIMPLE;
        context.from_rank = 0;
        context.to_rank = 0;
        context.is_async = false;
        context.is_heavy = false;

        Status status = engine_proxy.Invoke(&request, &response, &context);
        EXPECT_TRUE(status.OK());
      }

      // Heavy task
      {
        SimpleTestRequest request;
        request.value = i + 10;
        SimpleTestResponse response;

        IpcContext context;
        context.func_id = TEST_FUNC_HEAVY;
        context.from_rank = 0;
        context.to_rank = 0;
        context.is_async = false;
        context.is_heavy = true;

        Status status = engine_proxy.Invoke(&request, &response, &context);
        EXPECT_TRUE(status.OK());
      }
    }

    engine_proxy.Destroy();
    EXPECT_TRUE(WaitForChild(pid));
  }
}

TEST_F(IpcFuncProxyTest, HeavyTaskOnewayInvocation) {
  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    // Child process: Executor
    try {
      IpcFuncProxy executor_proxy(executor_num_, false, 0, false);
      executor_proxy.SetFuncInfo<HeavyTaskCallable, SimpleTestRequest, SimpleTestResponse, HeavyTaskCallback>(
          TEST_FUNC_HEAVY);

      // Wait for invocation
      g_test_state.WaitForInvocation(1, 10000);

      EXPECT_EQ(g_test_state.actual_value, 88);

      executor_proxy.Destroy();
      exit(0);
    } catch (...) {
      exit(1);
    }
  } else {
    // Parent process: Engine
    IpcFuncProxy engine_proxy(executor_num_, true, 0, false);
    engine_proxy.SetFuncInfo<HeavyTaskCallable, SimpleTestRequest, SimpleTestResponse, HeavyTaskCallback>(
        TEST_FUNC_HEAVY);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Prepare heavy task request
    SimpleTestRequest request;
    request.value = 88;
    request.message = "oneway heavy task";

    IpcContext context;
    context.func_id = TEST_FUNC_HEAVY;
    context.from_rank = 0;
    context.to_rank = 0;
    context.is_async = true;
    context.is_oneway = true;
    context.is_heavy = true;  // Mark as heavy task

    // Invoke oneway
    Status status = engine_proxy.Invoke(&request, nullptr, &context);
    EXPECT_TRUE(status.OK());

    // Give time for task to be processed
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    engine_proxy.Destroy();
    EXPECT_TRUE(WaitForChild(pid));
  }
}

TEST_F(IpcFuncProxyTest, HeavyTaskBroadcast) {
  std::vector<pid_t> child_pids;

  // Create multiple executors
  for (size_t i = 0; i < executor_num_; ++i) {
    pid_t pid = fork();
    ASSERT_NE(pid, -1);

    if (pid == 0) {
      // Child process: Executor
      try {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        IpcFuncProxy executor_proxy(executor_num_, false, i, false);
        executor_proxy.SetFuncInfo<HeavyTaskCallable, SimpleTestRequest, SimpleTestResponse, HeavyTaskCallback>(
            TEST_FUNC_HEAVY);

        // Wait for invocation
        g_test_state.WaitForInvocation(1, 10000);

        EXPECT_EQ(g_test_state.actual_value, 75);

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        executor_proxy.Destroy();
        exit(0);
      } catch (...) {
        exit(1);
      }
    } else {
      child_pids.push_back(pid);
    }
  }

  // Parent process: Engine
  IpcFuncProxy engine_proxy(executor_num_, true, 0, false);
  engine_proxy.SetFuncInfo<HeavyTaskCallable, SimpleTestRequest, SimpleTestResponse, HeavyTaskCallback>(
      TEST_FUNC_HEAVY);

  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  // Prepare broadcast heavy task request
  SimpleTestRequest request;
  request.value = 75;
  request.message = "broadcast heavy task";

  IpcContext context;
  context.func_id = TEST_FUNC_HEAVY;
  context.from_rank = 0;
  context.to_rank = -1;  // Broadcast to all
  context.is_async = true;
  context.is_oneway = false;
  context.is_heavy = true;  // Mark as heavy task

  // Invoke broadcast
  Status status = engine_proxy.Invoke(&request, nullptr, &context);
  EXPECT_TRUE(status.OK());

  // Wait for all callbacks
  g_test_state.WaitForInvocation(executor_num_, 10000);  // Only callbacks run in parent process

  // Verify accumulated result (only callback results in parent process)
  EXPECT_EQ(g_test_state.actual_value, 75 * 3 * executor_num_);  // 75*3*2 = 450

  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  engine_proxy.Destroy();

  // Wait for all children
  for (pid_t pid : child_pids) {
    EXPECT_TRUE(WaitForChild(pid));
  }
}

TEST_F(IpcFuncProxyTest, HeavyTaskStressTest) {
  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    // Child process: Executor
    try {
      IpcFuncProxy executor_proxy(executor_num_, false, 0, false);
      executor_proxy.SetFuncInfo<HeavyTaskCallable, SimpleTestRequest, SimpleTestResponse, HeavyTaskCallback>(
          TEST_FUNC_HEAVY);

      // Wait for many invocations
      g_test_state.WaitForInvocation(20, 30000);

      executor_proxy.Destroy();
      exit(0);
    } catch (...) {
      exit(1);
    }
  } else {
    // Parent process: Engine
    IpcFuncProxy engine_proxy(executor_num_, true, 0, false);
    engine_proxy.SetFuncInfo<HeavyTaskCallable, SimpleTestRequest, SimpleTestResponse, HeavyTaskCallback>(
        TEST_FUNC_HEAVY);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Send many heavy tasks rapidly
    for (int i = 0; i < 20; ++i) {
      SimpleTestRequest request;
      request.value = i;
      SimpleTestResponse response;

      IpcContext context;
      context.func_id = TEST_FUNC_HEAVY;
      context.from_rank = 0;
      context.to_rank = 0;
      context.is_async = false;
      context.is_heavy = true;

      Status status = engine_proxy.Invoke(&request, &response, &context);
      EXPECT_TRUE(status.OK());
      EXPECT_EQ(response.result, i * 3);
    }

    engine_proxy.Destroy();
    EXPECT_TRUE(WaitForChild(pid));
  }
}

// Test to verify that normal and heavy tasks can truly execute in parallel
// Note: Due to IpcFuncInfo mutex, same func_id calls are serialized.
// This test verifies that DIFFERENT func_ids (normal vs heavy) can execute in parallel.
TEST_F(IpcFuncProxyTest, ParallelExecutionOfNormalAndHeavyTasks) {
  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    // Child process: Executor
    try {
      IpcFuncProxy executor_proxy(executor_num_, false, 0, false);
      executor_proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
          TEST_FUNC_SIMPLE);
      executor_proxy.SetFuncInfo<HeavyTaskCallable, SimpleTestRequest, SimpleTestResponse, HeavyTaskCallback>(
          TEST_FUNC_HEAVY);

      // Wait for all tasks to complete
      // 1 heavy task + 5 normal tasks = 6 total
      g_test_state.WaitForInvocation(6, 15000);

      executor_proxy.Destroy();
      exit(0);
    } catch (...) {
      exit(1);
    }
  } else {
    // Parent process: Engine
    IpcFuncProxy engine_proxy(executor_num_, true, 0, false);
    engine_proxy.SetFuncInfo<SimpleTestCallable, SimpleTestRequest, SimpleTestResponse, SimpleTestCallback>(
        TEST_FUNC_SIMPLE);
    engine_proxy.SetFuncInfo<HeavyTaskCallable, SimpleTestRequest, SimpleTestResponse, HeavyTaskCallback>(
        TEST_FUNC_HEAVY);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto start_time = std::chrono::steady_clock::now();

    // Send 1 heavy task (takes ~100ms) - will be executed by long task thread
    std::thread heavy_task_thread([&engine_proxy]() {
      SimpleTestRequest request;
      request.value = 100;
      SimpleTestResponse response;

      IpcContext context;
      context.func_id = TEST_FUNC_HEAVY;
      context.from_rank = 0;
      context.to_rank = 0;
      context.is_async = false;
      context.is_heavy = true;

      Status status = engine_proxy.Invoke(&request, &response, &context);
      EXPECT_TRUE(status.OK());
    });

    // Immediately send 5 normal tasks sequentially (should complete quickly)
    // These should NOT be blocked by the heavy task because they use different func_id
    std::this_thread::sleep_for(std::chrono::milliseconds(10));  // Small delay to ensure heavy task is queued

    for (int i = 0; i < 5; ++i) {
      SimpleTestRequest request;
      request.value = i;
      SimpleTestResponse response;

      IpcContext context;
      context.func_id = TEST_FUNC_SIMPLE;
      context.from_rank = 0;
      context.to_rank = 0;
      context.is_async = false;
      context.is_heavy = false;

      Status status = engine_proxy.Invoke(&request, &response, &context);
      EXPECT_TRUE(status.OK());
      EXPECT_EQ(response.result, i * 2);
    }

    auto normal_tasks_end_time = std::chrono::steady_clock::now();
    auto normal_tasks_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(normal_tasks_end_time - start_time).count();

    // Normal tasks should complete quickly (< 100ms) even though heavy task is running
    // If normal tasks were blocked by heavy task, it would take > 100ms
    // Since same func_id is serialized, we send normal tasks sequentially, so they should be very fast
    EXPECT_LT(normal_tasks_duration, 100)
        << "Normal tasks took " << normal_tasks_duration
        << "ms, suggesting they were blocked by heavy task. Expected < 100ms for parallel execution.";

    KLLM_LOG_INFO << "Normal tasks completed in " << normal_tasks_duration << "ms (expected < 100ms)";

    // Wait for heavy task to complete
    heavy_task_thread.join();

    auto total_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count();

    KLLM_LOG_INFO << "Total execution time: " << total_duration << "ms";

    // Total time should be dominated by heavy task (~100ms)
    // If tasks were sequential, it would be much longer
    EXPECT_LT(total_duration, 200) << "Total execution took " << total_duration
                                   << "ms, suggesting poor parallelism. Expected < 200ms.";

    engine_proxy.Destroy();
    EXPECT_TRUE(WaitForChild(pid));
  }
}

}  // namespace ksana_llm
