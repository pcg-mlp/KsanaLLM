/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "cache_manager/block_allocator/executor_block_allocator.h"

#include "configure/context.h"
#include "configure/environment.h"
#include "connector/config.h"
#include "device/device_types.h"
#include "device/device_utils.h"
#include "fmt/core.h"
#include "pd_v2/configure/pd_v2_runtime_config.h"
#include "utils/logger.h"
#include "utils/singleton.h"
#include "utils/string_utils.h"

namespace ksana_llm {

namespace {

// pd_v2 requires the executor KV pool to be a single contiguous allocation so
// peer Prefill nodes can compute target_offset = block_id * block_size when
// writing via Mooncake. Gate on YAML `setting.connector.backend == "v2"`
// (parsed into PdV2RuntimeConfig::is_pd_v2). No env vars, no compile flags.
inline bool IsPdV2ContiguousKvEnabled() {
  PdV2RuntimeConfig pd;
  if (!Singleton<Environment>::GetInstance()->GetPdV2RuntimeConfig(pd).OK()) {
    return false;
  }
  return pd.is_pd_v2;
}

}  // namespace

ExecutorBlockAllocator::ExecutorBlockAllocator(MemoryLocation location, size_t block_num, size_t block_size, int rank,
                                               std::shared_ptr<MemoryAllocatorInterface> memory_allocator,
                                               std::shared_ptr<Context> context)
    : location_(location),
      block_num_(block_num),
      block_size_(block_size),
      rank_(rank),
      memory_allocator_(memory_allocator),
      context_(context) {
  using namespace std::placeholders;
  if (location_ == MemoryLocation::LOCATION_HOST) {
    malloc_fn_ = std::bind(&MemoryAllocatorInterface::HostAlloc, memory_allocator.get(), _1, _2);
    free_fn_ = std::bind(&MemoryAllocatorInterface::HostFree, memory_allocator.get(), _1);
  } else if (location_ == MemoryLocation::LOCATION_DEVICE) {
    // pd_v2 needs the KV pool registered with the RDMA HCA via
    // ibv_reg_mr (GPU Direct RDMA). cudaMallocAsync pool sub-allocations
    // are NOT recognized by the nv_peer_mem kernel module — registering
    // them returns EFAULT (Bad address). Use sync cudaMalloc / cudaFree
    // on pd_v2 paths so nv_peer_mem can map the buffer.
    //
    // History: P0.4 (commit 1708716e, 2026-05-15) introduced this gated
    // sync-alloc path. Commit e8b4811c reverted it 2026-05-16 because at
    // that point the test outputs were still degenerate — but the actual
    // root cause was K7 (`is_decode_only` flag wrong for pd_v2 Decode in
    // local-bypass mode), which was independently fixed in commit
    // ae7cf355. With K7 fixed, sync alloc can be re-enabled without
    // re-introducing the "token 2+ stutter" symptom; sync alloc remains
    // the right solution because it's the only way nv_peer_mem can
    // register the buffer for GDR.
    //
    // Runtime impact: ~zero. The KV pool is preallocated once at boot
    // and freed at process exit; the stream-ordered async free
    // optimization only matters for high-frequency alloc/free traffic
    // which the KV pool doesn't have. Boot adds a one-shot sync
    // cudaMalloc of the contiguous slab (~6-15 s on 96GB H20 pools).
    if (IsPdV2ContiguousKvEnabled()) {
      malloc_fn_ = std::bind(&MemoryAllocatorInterface::Malloc, memory_allocator.get(), _1, _2);
      free_fn_ = std::bind(&MemoryAllocatorInterface::Free, memory_allocator.get(), _1);
    } else {
      malloc_fn_ = std::bind(&MemoryAllocatorInterface::MallocAsync, memory_allocator.get(), _1, _2,
                             context_->device->GetMemoryManageStream());
      free_fn_ = std::bind(&MemoryAllocatorInterface::FreeAsync, memory_allocator.get(), _1,
                           context_->device->GetMemoryManageStream());
    }
  } else {
    KLLM_THROW("The MemoryLocation is not supported.");
  }
}

ExecutorBlockAllocator::~ExecutorBlockAllocator() { Clear(); }

void ExecutorBlockAllocator::PreAllocateBlocks() {
  bool use_continuous_memory = false;
  void* base_mem_ptr = nullptr;

  if (location_ == MemoryLocation::LOCATION_DEVICE) {
    SetDevice(rank_);
  }

#if defined(ENABLE_ACL) || defined(ENABLE_FLASH_ATTN_WITH_CACHE) || defined(ENABLE_KUNLUN)
  if (location_ == MemoryLocation::LOCATION_DEVICE) {
    use_continuous_memory = true;
    size_t dev_bytes = (block_num_ + 1) * block_size_;
    malloc_fn_(reinterpret_cast<void**>(&base_mem_ptr), dev_bytes);
    blocks_base_ptr_ = base_mem_ptr;
  }
#endif

  // pd_v2 path: same contiguous layout as the gated branch above, regardless
  // of attn backend. Required so peer Prefill nodes can write directly into
  // KV pool slots via Mooncake using target_offset = block_id * block_size.
  if (!use_continuous_memory && location_ == MemoryLocation::LOCATION_DEVICE && IsPdV2ContiguousKvEnabled()) {
    use_continuous_memory = true;
    size_t dev_bytes = (block_num_ + 1) * block_size_;
    malloc_fn_(reinterpret_cast<void**>(&base_mem_ptr), dev_bytes);
    // The constructor above bound `malloc_fn_` to sync `cudaMalloc` for
    // the pd_v2 path so nv_peer_mem can register the buffer for GPU
    // Direct RDMA. Sync alloc returns only after physical backing is
    // committed, so no StreamSynchronize is needed here. (Async path
    // doesn't reach this branch — it's gated on
    // `IsPdV2ContiguousKvEnabled()`.)
    blocks_base_ptr_ = base_mem_ptr;
  }

  // Host contiguous path: single cudaHostAlloc + pointer slicing.
  if (!use_continuous_memory && location_ == MemoryLocation::LOCATION_HOST) {
    use_continuous_memory = true;
    size_t dev_bytes = (block_num_ + 1) * block_size_;
    malloc_fn_(reinterpret_cast<void**>(&base_mem_ptr), dev_bytes);
    blocks_base_ptr_ = base_mem_ptr;
  }

  // NOTE: Make sure block ids on all worker nodes have same id range.
  allocated_blocks_.reserve(block_num_);
  void* memory_ptr = nullptr;
  for (size_t block_id = 0; block_id < block_num_; ++block_id) {
    if (use_continuous_memory) {
      memory_ptr = reinterpret_cast<uint8_t*>(base_mem_ptr) + block_id * block_size_;
    } else {
      malloc_fn_(&memory_ptr, block_size_);
    }
    allocated_blocks_.emplace(block_id, memory_ptr);
  }
}

void ExecutorBlockAllocator::Clear() {
  if (location_ == MemoryLocation::LOCATION_DEVICE) {
    SetDevice(rank_);
  }

#if defined(ENABLE_ACL_ATB) || defined(ENABLE_FLASH_ATTN_WITH_CACHE) || defined(ENABLE_KUNLUN)
  // 与 PreAllocateBlocks() 的连续内存分配保持对称:那里对 ACL/FLASH_ATTN_WITH_CACHE/KUNLUN
  // 只做一次 malloc 得到 base_mem_ptr,各 block 是它的偏移指针;此处必须只 free 这一个 base
  // 指针。若漏掉 ENABLE_KUNLUN 而走下面的通用逐指针释放,会对偏移指针调用 xpu_free,触发
  // XPU -707 "free a non-base allocation device pointer"。
  if (location_ == MemoryLocation::LOCATION_DEVICE) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (blocks_base_ptr_ != nullptr) {
      free_fn_(blocks_base_ptr_);
      blocks_base_ptr_ = nullptr;
    }
    allocated_blocks_.clear();
    return;
  }
#endif

  // pd_v2 contiguous-KV path mirrors the gated branch above for the
  // single-allocation free.
  if (blocks_base_ptr_ != nullptr) {
    std::unique_lock<std::mutex> lock(mutex_);
    free_fn_(blocks_base_ptr_);
    blocks_base_ptr_ = nullptr;
    allocated_blocks_.clear();
    return;
  }

  {
    auto clear_fn = [&](std::unordered_map<int, void*>& blocks) -> void {
      for (auto it = blocks.begin(); it != blocks.end();) {
        free_fn_(it->second);
        it = blocks.erase(it);
      }
    };

    std::unique_lock<std::mutex> lock(mutex_);
    clear_fn(allocated_blocks_);
  }
}

Status ExecutorBlockAllocator::AllocateBlocks(size_t block_num, std::vector<int>& blocks) {
  KLLM_THROW("ExecutorBlockAllocator::AllocateBlocks is not implemented.");
}

Status ExecutorBlockAllocator::FreeBlocks(const std::vector<int>& blocks) {
  KLLM_THROW("ExecutorBlockAllocator::FreeBlocks is not implemented.");
}

Status ExecutorBlockAllocator::GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) {
  std::unique_lock<std::mutex> lock(mutex_);

  addrs.clear();
  addrs.reserve(blocks.size());
  for (auto block_id : blocks) {
    auto it = allocated_blocks_.find(block_id);
    if (it != allocated_blocks_.end()) {
      addrs.push_back(it->second);
      continue;
    }

    KLLM_LOG_ERROR << "Get block id " << block_id << " address error on device " << rank_;
    return Status(RET_SEGMENT_FAULT, FormatStr("Get block address error, block id %d", block_id));
  }
  return Status();
}

void* ExecutorBlockAllocator::GetBlocksBasePtr() { return blocks_base_ptr_; }

int ExecutorBlockAllocator::GetBlocksBaseId() { return blocks_base_id_; }

size_t ExecutorBlockAllocator::GetFreeBlockNumber() {
  KLLM_THROW("ExecutorBlockAllocator::GetFreeBlockNumber is not implemented.");
}

size_t ExecutorBlockAllocator::GetUsedBlockNumber() {
  KLLM_THROW("ExecutorBlockAllocator::GetUsedBlockNumber is not implemented.");
}

}  // namespace ksana_llm
