/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/kernel_registry.h"

#include <iostream>

#include "test.h"

void Fun(int i) { std::cout << i << std::endl; }
void Fun2(int i, float j) { std::cout << i + j << std::endl; }

// A demo for kernel invoke with blocked input.
//
// Let's assume we have tow sequence, with length 14 and 8, every block could fill 5 tokens.
//
// seq_1:  block1(5 tokens)       block2(5 tokens)       block3(4 tokens)
// seq_2:  block4(5 tokens)       block5(3 tokens)
//
// The argument is:
//
// seq_num:    2
// seq_len:    [14, 8]
// block_ptr:  [[addr1, addr2, addr3], [addr4, addr5]]
// block_num:  [3, 2]
// block_len:  5
//
void kernel_func(int seq_num, int seq_len[], void *block_ptr[], int block_num[], int block_len) {}

TEST(KernelRegistryTest, CommonTest) {
  REGISTER_NVIDIA_KERNEL(a, Fun);
  REGISTER_NVIDIA_KERNEL(b, Fun2);

  int i = 1;
  EXECUTE_NVIDIA_KERNEL(a, i);
  EXECUTE_NVIDIA_KERNEL(b, 1, 0.5);
}
