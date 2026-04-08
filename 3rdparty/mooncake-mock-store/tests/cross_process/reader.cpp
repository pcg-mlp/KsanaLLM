/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mooncake/store.h"

int main(int argc, char** argv) {
  const char* dir = (argc > 1) ? argv[1] : "/tmp/mooncake_mock";

  mooncake::MockStore store;
  store.setup("reader", dir, 0, 0, "tcp", "", "");

  // Wait for writer to finish
  printf("[reader] waiting for writer...\n");
  for (int i = 0; i < 100; i++) {
    if (store.is_exist("__done__") == 1) break;
    usleep(50000);  // 50ms
  }
  if (store.is_exist("__done__") != 1) {
    fprintf(stderr, "[reader] FAIL: writer did not finish in time\n");
    return 1;
  }
  printf("[reader] writer done, verifying data...\n");

  int errors = 0;

  // 1. String
  {
    void* out = nullptr;
    int64_t n = store.get("greeting", &out);
    const char* expected = "Hello from writer process!";
    if (n != (int64_t)strlen(expected) || memcmp(out, expected, n) != 0) {
      fprintf(stderr, "[reader] FAIL: greeting mismatch\n");
      errors++;
    } else {
      printf("[reader] greeting: OK\n");
    }
    free(out);
  }

  // 2. Float tensor
  {
    constexpr size_t ROWS = 1000, COLS = 100, COUNT = ROWS * COLS;
    std::vector<float> buf(COUNT);
    int64_t n = store.get_into("tensor_1000x100", (uintptr_t)buf.data(), COUNT * sizeof(float));
    if (n != (int64_t)(COUNT * sizeof(float))) {
      fprintf(stderr, "[reader] FAIL: tensor size mismatch (got %lld)\n", (long long)n);
      errors++;
    } else {
      bool ok = true;
      for (size_t i = 0; i < COUNT && ok; i++) {
        float expected = (float)i * 0.001f;
        if (std::fabs(buf[i] - expected) > 1e-6f) ok = false;
      }
      if (!ok) {
        fprintf(stderr, "[reader] FAIL: tensor data mismatch\n");
        errors++;
      } else {
        printf("[reader] tensor_1000x100: OK\n");
      }
    }
  }

  // 3. KV cache blocks
  {
    constexpr int N_BLOCKS = 10;
    constexpr size_t BLOCK_SZ = 128 * 1024;
    bool all_ok = true;
    for (int i = 0; i < N_BLOCKS; i++) {
      std::string key = "kv_block_" + std::to_string(i);
      void* out = nullptr;
      int64_t n = store.get(key.c_str(), &out);
      if (n != (int64_t)BLOCK_SZ) {
        all_ok = false;
        free(out);
        break;
      }
      auto* p = (uint8_t*)out;
      for (size_t j = 0; j < BLOCK_SZ; j++) {
        if (p[j] != (uint8_t)((i * 37 + j) % 256)) {
          all_ok = false;
          break;
        }
      }
      free(out);
      if (!all_ok) break;
    }
    if (!all_ok) {
      fprintf(stderr, "[reader] FAIL: kv_block data mismatch\n");
      errors++;
    } else {
      printf("[reader] kv_block_0..9: OK\n");
    }
  }

  store.close();

  if (errors == 0) {
    printf("[reader] ALL PASSED\n");
    return 0;
  } else {
    fprintf(stderr, "[reader] %d error(s)\n", errors);
    return 1;
  }
}
