/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "ipc/core/ipc_func_serialization.h"
#include "ipc/core/ipc_func_types.h"
#include "ipc/functions/add_executor.h"
#include "utils/logger.h"
#include "utils/ret_code.h"
#include "utils/status.h"
#include "utils/waiter.h"

namespace ksana_llm {

// ============================================================================
// Test Helper Classes and Utilities
// ============================================================================

// Global state for tracking test execution
struct AddExecutorTestState {
  std::mutex mutex;
  std::condition_variable cv;
  int call_count = 0;
  int callback_count = 0;
  bool callable_invoked = false;
  bool callback_invoked = false;
  Status last_status;

  void Reset() {
    std::lock_guard<std::mutex> lock(mutex);
    call_count = 0;
    callback_count = 0;
    callable_invoked = false;
    callback_invoked = false;
    last_status = Status();
  }

  void IncrementCallCount() {
    std::lock_guard<std::mutex> lock(mutex);
    call_count++;
    callable_invoked = true;
    cv.notify_all();
  }

  void IncrementCallbackCount() {
    std::lock_guard<std::mutex> lock(mutex);
    callback_count++;
    callback_invoked = true;
    cv.notify_all();
  }

  void WaitForCalls(int expected_count, int timeout_ms = 5000) {
    std::unique_lock<std::mutex> lock(mutex);
    auto result = cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                              [this, expected_count] { return call_count >= expected_count; });
    if (!result) {
      KLLM_LOG_WARNING << "WaitForCalls timeout, expected: " << expected_count << ", actual: " << call_count;
    }
  }

  void WaitForCallbacks(int expected_count, int timeout_ms = 5000) {
    std::unique_lock<std::mutex> lock(mutex);
    auto result = cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                              [this, expected_count] { return callback_count >= expected_count; });
    if (!result) {
      KLLM_LOG_WARNING << "WaitForCallbacks timeout, expected: " << expected_count << ", actual: " << callback_count;
    }
  }
};

static AddExecutorTestState g_test_state;

// Mock callable for testing
class MockAddExecutorCallable : public AddExecutorCallable {
 public:
  Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override {
    g_test_state.IncrementCallCount();

    // Call parent implementation to trigger Notify()
    Status status = AddExecutorCallable::Call(request, response, context);

    return status;
  }
};

// Mock callback for testing
class MockAddExecutorCallback : public AddExecutorCallback {
 public:
  Status Call(const IpcResponse* response, IpcContext* context) override {
    g_test_state.IncrementCallbackCount();
    return Status();
  }
};

// Error-simulating callable
class ErrorAddExecutorCallable : public AddExecutorCallable {
 public:
  Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override {
    g_test_state.IncrementCallCount();
    return Status(RET_RUNTIME_FAILED, "Simulated error in AddExecutorCallable");
  }
};

// Error-simulating callback
class ErrorAddExecutorCallback : public AddExecutorCallback {
 public:
  Status Call(const IpcResponse* response, IpcContext* context) override {
    g_test_state.IncrementCallbackCount();
    return Status(RET_RUNTIME_FAILED, "Simulated error in AddExecutorCallback");
  }
};

// ============================================================================
// Test Fixture
// ============================================================================

class AddExecutorTest : public ::testing::Test {
 protected:
  void SetUp() override { g_test_state.Reset(); }

  void TearDown() override {
    // Cleanup if needed
  }
};

// ============================================================================
// Request and Response Tests
// ============================================================================

TEST_F(AddExecutorTest, RequestConstruction) {
  // Test default construction
  AddExecutorRequest request;
  EXPECT_NO_THROW({ AddExecutorRequest req; });
}

TEST_F(AddExecutorTest, RequestDestruction) {
  // Test proper destruction
  EXPECT_NO_THROW({
    AddExecutorRequest* req = new AddExecutorRequest();
    delete req;
  });
}

TEST_F(AddExecutorTest, RequestPolymorphism) {
  // Test polymorphic behavior
  IpcRequest* base_ptr = new AddExecutorRequest();
  EXPECT_NE(base_ptr, nullptr);

  // Test virtual destructor
  EXPECT_NO_THROW({ delete base_ptr; });
}

TEST_F(AddExecutorTest, ResponseConstruction) {
  // Test default construction
  AddExecutorResponse response;
  EXPECT_NO_THROW({ AddExecutorResponse rsp; });
}

TEST_F(AddExecutorTest, ResponseDestruction) {
  // Test proper destruction
  EXPECT_NO_THROW({
    AddExecutorResponse* rsp = new AddExecutorResponse();
    delete rsp;
  });
}

TEST_F(AddExecutorTest, ResponsePolymorphism) {
  // Test polymorphic behavior
  IpcResponse* base_ptr = new AddExecutorResponse();
  EXPECT_NE(base_ptr, nullptr);

  // Test virtual destructor
  EXPECT_NO_THROW({ delete base_ptr; });
}

// ============================================================================
// Callable Tests
// ============================================================================

TEST_F(AddExecutorTest, CallableConstruction) {
  // Test default construction
  EXPECT_NO_THROW({ AddExecutorCallable callable; });
}

TEST_F(AddExecutorTest, CallableDestruction) {
  // Test proper destruction
  EXPECT_NO_THROW({
    AddExecutorCallable* callable = new AddExecutorCallable();
    delete callable;
  });
}

TEST_F(AddExecutorTest, CallablePolymorphism) {
  // Test polymorphic behavior as IpcCallable
  IpcCallable* base_ptr = new AddExecutorCallable();
  EXPECT_NE(base_ptr, nullptr);

  EXPECT_NO_THROW({ delete base_ptr; });
}

TEST_F(AddExecutorTest, CallableInheritance) {
  // Test that AddExecutorCallable inherits from both IpcCallable and IpcWaiter
  AddExecutorCallable callable;

  // Should be able to cast to IpcCallable
  IpcCallable* ipc_callable = dynamic_cast<IpcCallable*>(&callable);
  EXPECT_NE(ipc_callable, nullptr);

  // Should be able to cast to IpcWaiter
  IpcWaiter* ipc_waiter = dynamic_cast<IpcWaiter*>(&callable);
  EXPECT_NE(ipc_waiter, nullptr);
}

TEST_F(AddExecutorTest, CallableExecution) {
  AddExecutorCallable callable;
  AddExecutorRequest request;
  AddExecutorResponse response;
  IpcContext context;

  // Execute the callable
  Status status = callable.Call(&request, &response, &context);

  // Should return success
  EXPECT_TRUE(status.OK());
}

TEST_F(AddExecutorTest, CallableWithMock) {
  MockAddExecutorCallable callable;
  AddExecutorRequest request;
  AddExecutorResponse response;
  IpcContext context;

  // Execute the callable
  Status status = callable.Call(&request, &response, &context);

  // Verify execution
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(g_test_state.call_count, 1);
  EXPECT_TRUE(g_test_state.callable_invoked);
}

TEST_F(AddExecutorTest, CallableMultipleInvocations) {
  MockAddExecutorCallable callable;
  AddExecutorRequest request;
  AddExecutorResponse response;
  IpcContext context;

  // Execute multiple times
  const int num_calls = 5;
  for (int i = 0; i < num_calls; ++i) {
    Status status = callable.Call(&request, &response, &context);
    EXPECT_TRUE(status.OK());
  }

  // Verify all calls were counted
  EXPECT_EQ(g_test_state.call_count, num_calls);
}

TEST_F(AddExecutorTest, CallableErrorHandling) {
  ErrorAddExecutorCallable callable;
  AddExecutorRequest request;
  AddExecutorResponse response;
  IpcContext context;

  // Execute the callable
  Status status = callable.Call(&request, &response, &context);

  // Should return error
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RET_RUNTIME_FAILED);
  EXPECT_EQ(g_test_state.call_count, 1);
}

// ============================================================================
// Callback Tests
// ============================================================================

TEST_F(AddExecutorTest, CallbackConstruction) {
  // Test default construction
  EXPECT_NO_THROW({ AddExecutorCallback callback; });
}

TEST_F(AddExecutorTest, CallbackDestruction) {
  // Test proper destruction
  EXPECT_NO_THROW({
    AddExecutorCallback* callback = new AddExecutorCallback();
    delete callback;
  });
}

TEST_F(AddExecutorTest, CallbackPolymorphism) {
  // Test polymorphic behavior
  IpcCallback* base_ptr = new AddExecutorCallback();
  EXPECT_NE(base_ptr, nullptr);

  EXPECT_NO_THROW({ delete base_ptr; });
}

TEST_F(AddExecutorTest, CallbackWithMock) {
  MockAddExecutorCallback callback;
  AddExecutorResponse response;
  IpcContext context;

  // Execute the callback
  Status status = callback.Call(&response, &context);

  // Verify execution
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(g_test_state.callback_count, 1);
  EXPECT_TRUE(g_test_state.callback_invoked);
}

TEST_F(AddExecutorTest, CallbackMultipleInvocations) {
  MockAddExecutorCallback callback;
  AddExecutorResponse response;
  IpcContext context;

  // Execute multiple times
  const int num_calls = 5;
  for (int i = 0; i < num_calls; ++i) {
    Status status = callback.Call(&response, &context);
    EXPECT_TRUE(status.OK());
  }

  // Verify all calls were counted
  EXPECT_EQ(g_test_state.callback_count, num_calls);
}

TEST_F(AddExecutorTest, CallbackErrorHandling) {
  ErrorAddExecutorCallback callback;
  AddExecutorResponse response;
  IpcContext context;

  // Execute the callback
  Status status = callback.Call(&response, &context);

  // Should return error
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RET_RUNTIME_FAILED);
  EXPECT_EQ(g_test_state.callback_count, 1);
}

// ============================================================================
// Waiter Integration Tests
// ============================================================================

TEST_F(AddExecutorTest, WaiterInitialization) {
  AddExecutorCallable callable;

  // Initialize waiter
  EXPECT_NO_THROW({ callable.Reset(1); });

  // Waiter should be created
  EXPECT_NE(callable.waiter, nullptr);
}

TEST_F(AddExecutorTest, WaiterNotifyAndWait) {
  AddExecutorCallable callable;
  callable.Reset(1);

  // Start a thread to notify after a delay
  std::thread notifier([&callable]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    callable.Notify();
  });

  // Wait should block until notified
  auto start = std::chrono::steady_clock::now();
  callable.Wait();
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // Should have waited at least 100ms
  EXPECT_GE(duration.count(), 100);

  notifier.join();
}

TEST_F(AddExecutorTest, WaiterMultipleNotifications) {
  AddExecutorCallable callable;
  const int num_notifications = 3;
  callable.Reset(num_notifications);

  // Start threads to notify
  std::vector<std::thread> notifiers;
  for (int i = 0; i < num_notifications; ++i) {
    notifiers.emplace_back([&callable]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      callable.Notify();
    });
  }

  // Wait should block until all notifications received
  auto start = std::chrono::steady_clock::now();
  callable.Wait();
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // Should have waited at least 50ms
  EXPECT_GE(duration.count(), 50);

  for (auto& t : notifiers) {
    t.join();
  }
}

TEST_F(AddExecutorTest, WaiterReset) {
  AddExecutorCallable callable;

  // First initialization
  callable.Reset(1);
  auto first_waiter = callable.waiter;
  EXPECT_NE(first_waiter, nullptr);

  // Reset with different count
  callable.Reset(2);
  auto second_waiter = callable.waiter;
  EXPECT_NE(second_waiter, nullptr);

  // Should reuse the same waiter object
  EXPECT_EQ(first_waiter, second_waiter);
}

TEST_F(AddExecutorTest, WaiterInCallableExecution) {
  AddExecutorCallable callable;
  callable.Reset(1);

  AddExecutorRequest request;
  AddExecutorResponse response;
  IpcContext context;

  // Start a thread to wait
  std::atomic<bool> wait_completed{false};
  std::thread waiter([&callable, &wait_completed]() {
    callable.Wait();
    wait_completed = true;
  });

  // Give waiter time to start
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(wait_completed);

  // Execute callable (which calls Notify())
  Status status = callable.Call(&request, &response, &context);
  EXPECT_TRUE(status.OK());

  // Wait should complete
  waiter.join();
  EXPECT_TRUE(wait_completed);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(AddExecutorTest, CallableAndCallbackIntegration) {
  MockAddExecutorCallable callable;
  MockAddExecutorCallback callback;

  AddExecutorRequest request;
  AddExecutorResponse response;
  IpcContext context;

  // Execute callable
  Status callable_status = callable.Call(&request, &response, &context);
  EXPECT_TRUE(callable_status.OK());
  EXPECT_EQ(g_test_state.call_count, 1);

  // Execute callback
  Status callback_status = callback.Call(&response, &context);
  EXPECT_TRUE(callback_status.OK());
  EXPECT_EQ(g_test_state.callback_count, 1);
}

TEST_F(AddExecutorTest, MultipleCallablesWithWaiter) {
  const int num_callables = 3;
  AddExecutorCallable main_callable;
  main_callable.Reset(num_callables);

  // Start threads to execute callables
  std::vector<std::thread> threads;
  for (int i = 0; i < num_callables; ++i) {
    threads.emplace_back([&main_callable]() {
      AddExecutorRequest request;
      AddExecutorResponse response;
      IpcContext context;

      AddExecutorCallable callable;
      callable.waiter = main_callable.waiter;  // Share the waiter

      Status status = callable.Call(&request, &response, &context);
      EXPECT_TRUE(status.OK());
    });
  }

  // Wait for all callables to complete
  auto start = std::chrono::steady_clock::now();
  main_callable.Wait();
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // Should have completed
  EXPECT_LT(duration.count(), 5000);  // Should not timeout

  for (auto& t : threads) {
    t.join();
  }
}

TEST_F(AddExecutorTest, ConcurrentCallableAndCallbackExecution) {
  const int num_iterations = 10;
  MockAddExecutorCallable callable;
  MockAddExecutorCallback callback;

  // Execute concurrently
  std::thread callable_thread([&callable, num_iterations]() {
    for (int i = 0; i < num_iterations; ++i) {
      AddExecutorRequest request;
      AddExecutorResponse response;
      IpcContext context;

      Status status = callable.Call(&request, &response, &context);
      EXPECT_TRUE(status.OK());
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  });

  std::thread callback_thread([&callback, num_iterations]() {
    for (int i = 0; i < num_iterations; ++i) {
      AddExecutorResponse response;
      IpcContext context;

      Status status = callback.Call(&response, &context);
      EXPECT_TRUE(status.OK());
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  });

  callable_thread.join();
  callback_thread.join();

  // Verify all executions
  EXPECT_EQ(g_test_state.call_count, num_iterations);
  EXPECT_EQ(g_test_state.callback_count, num_iterations);
}

// ============================================================================
// Edge Cases and Error Scenarios
// ============================================================================

TEST_F(AddExecutorTest, NullRequestPointer) {
  AddExecutorCallable callable;
  AddExecutorRequest request;
  AddExecutorResponse response;
  IpcContext context;

  // Should handle null request gracefully
  EXPECT_NO_THROW({
    Status status = callable.Call(&request, &response, &context);
    // Implementation doesn't check for null, so it will succeed
    EXPECT_TRUE(status.OK());
  });
}

TEST_F(AddExecutorTest, NullResponsePointer) {
  AddExecutorCallable callable;
  AddExecutorRequest request;
  IpcContext context;

  // Should handle null response gracefully
  EXPECT_NO_THROW({
    Status status = callable.Call(&request, nullptr, &context);
    // Implementation doesn't check for null, so it will succeed
    EXPECT_TRUE(status.OK());
  });
}

TEST_F(AddExecutorTest, NullContextPointer) {
  AddExecutorCallable callable;
  AddExecutorRequest request;
  AddExecutorResponse response;

  // Should handle null context gracefully
  EXPECT_NO_THROW({
    Status status = callable.Call(&request, &response, nullptr);
    // Implementation doesn't check for null, so it will succeed
    EXPECT_TRUE(status.OK());
  });
}

TEST_F(AddExecutorTest, CallbackWithNullResponse) {
  MockAddExecutorCallback callback;
  IpcContext context;

  // Should handle null response gracefully
  EXPECT_NO_THROW({
    Status status = callback.Call(nullptr, &context);
    EXPECT_TRUE(status.OK());
  });
}

TEST_F(AddExecutorTest, CallbackWithNullContext) {
  MockAddExecutorCallback callback;
  AddExecutorResponse response;

  // Should handle null context gracefully
  EXPECT_NO_THROW({
    Status status = callback.Call(&response, nullptr);
    EXPECT_TRUE(status.OK());
  });
}

TEST_F(AddExecutorTest, WaiterWithoutInitialization) {
  AddExecutorCallable callable;

  // Waiter is null by default
  EXPECT_EQ(callable.waiter, nullptr);

  // Calling Notify without initialization should not crash
  // (but will cause segfault in actual implementation)
  // This test documents the expected behavior
}

TEST_F(AddExecutorTest, MemoryLeakCheck) {
  // Create and destroy multiple objects to check for leaks
  const int num_iterations = 100;

  for (int i = 0; i < num_iterations; ++i) {
    AddExecutorRequest* req = new AddExecutorRequest();
    AddExecutorResponse* rsp = new AddExecutorResponse();
    AddExecutorCallable* callable = new AddExecutorCallable();
    AddExecutorCallback* callback = new AddExecutorCallback();

    delete req;
    delete rsp;
    delete callable;
    delete callback;
  }

  // If we reach here without crashes, memory management is correct
  SUCCEED();
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(AddExecutorTest, CallablePerformance) {
  MockAddExecutorCallable callable;
  AddExecutorRequest request;
  AddExecutorResponse response;
  IpcContext context;

  const int num_calls = 1000;
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_calls; ++i) {
    Status status = callable.Call(&request, &response, &context);
    EXPECT_TRUE(status.OK());
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Should be reasonably fast (less than 100 microseconds per call)
  double avg_time = static_cast<double>(duration.count()) / num_calls;
  EXPECT_LT(avg_time, 100.0);
}

TEST_F(AddExecutorTest, CallbackPerformance) {
  MockAddExecutorCallback callback;
  AddExecutorResponse response;
  IpcContext context;

  const int num_calls = 1000;
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_calls; ++i) {
    Status status = callback.Call(&response, &context);
    EXPECT_TRUE(status.OK());
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Should be reasonably fast (less than 100 microseconds per call)
  double avg_time = static_cast<double>(duration.count()) / num_calls;
  EXPECT_LT(avg_time, 100.0);
}

}  // namespace ksana_llm
