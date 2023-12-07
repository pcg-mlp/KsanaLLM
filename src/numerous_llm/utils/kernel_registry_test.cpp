/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "numerous_llm/utils/kernel_registry.h"

#include <iostream>

void Fun(int i) { std::cout << i << endl; }
void Fun2(int i, float j) { std::cout << i + j << endl; }

int main() {
  REGISTER_NVIDIA_KERNEL(a, Fun);
  REGISTER_NVIDIA_KERNEL(b, Fun2);

  int i = 1;
  EXECUTE_NVIDIA_KERNEL(a, i);
  EXECUTE_NVIDIA_KERNEL(b, 1, 0.5);

  return 0;
}

