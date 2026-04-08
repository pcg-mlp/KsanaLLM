/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "mooncake/store.h"

int main(int argc, char** argv) {
  const char* dir = (argc > 1) ? argv[1] : "/tmp/mooncake_mock";

  mooncake::MockStore store;
  store.setup("writer", dir, 0, 0, "tcp", "", "");

  // 1. Simple string
  const char* msg = "Hello from writer process!";
  store.put("greeting", msg, strlen(msg));
  printf("[writer] put 'greeting' (%zu bytes)\n", strlen(msg));

  // 2. Float tensor (1000 x 100)
  constexpr size_t ROWS = 1000, COLS = 100;
  std::vector<float> tensor(ROWS * COLS);
  for (size_t i = 0; i < tensor.size(); i++) tensor[i] = (float)i * 0.001f;
  store.put("tensor_1000x100", tensor.data(), tensor.size() * sizeof(float));
  printf("[writer] put 'tensor_1000x100' (%zu bytes)\n", tensor.size() * sizeof(float));

  // 3. Batch of KV cache blocks
  constexpr int N_BLOCKS = 10;
  constexpr size_t BLOCK_SZ = 128 * 1024;  // 128KB
  for (int i = 0; i < N_BLOCKS; i++) {
    std::string key = "kv_block_" + std::to_string(i);
    std::vector<uint8_t> block(BLOCK_SZ);
    for (size_t j = 0; j < BLOCK_SZ; j++) block[j] = (uint8_t)((i * 37 + j) % 256);
    store.put(key.c_str(), block.data(), block.size());
  }
  printf("[writer] put %d kv_block_* (%zu bytes each)\n", N_BLOCKS, BLOCK_SZ);

  // 4. Write a "done" flag
  uint8_t flag = 1;
  store.put("__done__", &flag, 1);
  printf("[writer] done.\n");

  store.close();
  return 0;
}
