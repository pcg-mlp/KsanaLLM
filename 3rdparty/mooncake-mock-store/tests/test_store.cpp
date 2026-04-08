/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "mooncake/store.h"

using mooncake::MockStore;

static const char* TEST_DIR = "/tmp/mooncake_test";

static MockStore make() {
  MockStore s;
  s.setup("test", TEST_DIR);
  return s;
}

static void test_basic() {
  std::cout << "test_basic ... ";
  auto s = make();
  s.remove_all();

  const char* data = "Hello, Mooncake!";
  assert(s.put("k1", data, strlen(data)) == 0);
  assert(s.is_exist("k1") == 1);
  assert(s.get_size("k1") == (int64_t)strlen(data));

  void* out = nullptr;
  int64_t n = s.get("k1", &out);
  assert(n == (int64_t)strlen(data));
  assert(memcmp(out, data, n) == 0);
  free(out);

  assert(s.remove("k1") == 0);
  assert(s.is_exist("k1") == 0);
  s.close();
  std::cout << "ok\n";
}

static void test_zero_copy() {
  std::cout << "test_zero_copy ... ";
  auto s = make();

  float src[1000];
  for (int i = 0; i < 1000; i++) src[i] = (float)i;

  assert(s.put_from("tensor", (uintptr_t)src, sizeof(src)) == 0);

  float dst[1000] = {};
  int64_t n = s.get_into("tensor", (uintptr_t)dst, sizeof(dst));
  assert(n == sizeof(src));
  assert(memcmp(src, dst, sizeof(src)) == 0);

  s.close();
  std::cout << "ok\n";
}

static void test_batch() {
  std::cout << "test_batch ... ";
  auto s = make();

  const char* keys[] = {"a", "b", "c"};
  const char* vals[] = {"111", "222", "333"};
  const void* vptrs[] = {vals[0], vals[1], vals[2]};
  size_t sizes[] = {3, 3, 3};

  assert(s.put_batch(keys, vptrs, sizes, 3) == 0);

  void* outs[3] = {};
  int64_t osizes[3] = {};
  s.get_batch(keys, 3, outs, osizes);
  for (int i = 0; i < 3; i++) {
    assert(osizes[i] == 3);
    assert(memcmp(outs[i], vals[i], 3) == 0);
    free(outs[i]);
  }

  s.close();
  std::cout << "ok\n";
}

static void test_multithread() {
  std::cout << "test_multithread ... ";
  constexpr int N_THREADS = 4, N_ITEMS = 50;

  std::vector<MockStore> stores(N_THREADS);
  for (auto& s : stores) s.setup("node", TEST_DIR);

  std::vector<std::thread> writers;
  for (int t = 0; t < N_THREADS; t++) {
    writers.emplace_back([&, t] {
      for (int i = 0; i < N_ITEMS; i++) {
        std::string key = "t" + std::to_string(t) + "_" + std::to_string(i);
        std::string val = "v" + std::to_string(t) + "_" + std::to_string(i);
        assert(stores[t].put(key.c_str(), val.data(), val.size()) == 0);
      }
    });
  }
  for (auto& w : writers) w.join();

  for (int t = 0; t < N_THREADS; t++) {
    for (int i = 0; i < N_ITEMS; i++) {
      std::string key = "t" + std::to_string(t) + "_" + std::to_string(i);
      std::string expected = "v" + std::to_string(t) + "_" + std::to_string(i);
      void* out = nullptr;
      int64_t n = stores[0].get(key.c_str(), &out);
      assert(n == (int64_t)expected.size());
      assert(memcmp(out, expected.data(), n) == 0);
      free(out);
    }
  }

  for (auto& s : stores) s.close();
  std::cout << "ok\n";
}

static void test_large_tensor() {
  std::cout << "test_large_tensor ... ";
  auto s = make();

  constexpr size_t SZ = 1000 * 1000;
  std::vector<float> tensor(SZ);
  for (size_t i = 0; i < SZ; i++) tensor[i] = (float)i * 0.001f;

  assert(s.put("big", tensor.data(), SZ * sizeof(float)) == 0);

  std::vector<float> recv(SZ, 0);
  int64_t n = s.get_into("big", (uintptr_t)recv.data(), SZ * sizeof(float));
  assert(n == (int64_t)(SZ * sizeof(float)));
  assert(memcmp(tensor.data(), recv.data(), n) == 0);

  s.close();
  std::cout << "ok\n";
}

int main() {
  test_basic();
  test_zero_copy();
  test_batch();
  test_multithread();
  test_large_tensor();
  std::cout << "\nAll tests passed.\n";
  return 0;
}
