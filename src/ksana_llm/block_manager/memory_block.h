/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once
#include <cstdint>
#include "ksana_llm/utils/environment.h"

namespace ksana_llm {

// The memory block information.
struct MemoryBlock {
    // block id, unique in global.
    int64_t block_id;

    // block size, in bytes.
    int64_t block_size;

    // The reference count of current block.
    uint64_t ref_count = 0;

    // /The device of this block, CPU or GPU or NPU.
    MemoryDevice device;

    // The physical address of this block.
    void *address = nullptr;
};

}  // namespace ksana_llm
