/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace numerous_llm {

// Some basic fields of memory block.
class BaseBlock {
 public:
  // block index, zero based.
  int block_index;

  // block size, in number of tokens.
  int block_size;
};

// A block that stores a contiguous chunk of tokens from left to right.
// Logical blocks are used to represent the states of the corresponding
// physical blocks in the KV cache.
class KvCacheBlock : public BaseBlock {
 public:
  // The token ids this block contains.
  std::vector<int> token_ids;

  // The generated token numbers.
  size_t token_num = 0;

  // Whether current block is empty.
  bool IsEmpty();

  // Whether current block is full.
  bool IsFull();

  // Get the avail slot number.
  size_t GetEmptySlotNum();

  // The last token id.
  int GetLastTokenId();

  // Get the token id list.
  std::vector<int> GetTokenIds();
};

// A block that store a chunk of model's lora weights.
class LoraWeightBlock : public BaseBlock {};

// The physical memory block information.
class PhysicalBlock : public BaseBlock {
 public:
  // The reference count of current block.
  int ref_count = 0;

  // /The device of this block, CPU or GPU or NPU.
  std::string device;

  // The address of this block.
  void *address = nullptr;
};

}  // namespace numerous_llm
