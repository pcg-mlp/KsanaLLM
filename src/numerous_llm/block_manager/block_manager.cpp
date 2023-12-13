/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/


#include "numerous_llm/block_manager/block_manager.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/singleton.h"

namespace numerous_llm {

// 构造函数，根据Singleton Instance的 BlockManagerConfig 配置 BlockManager
BlockManager::BlockManager() : BlockManager(GetBlockManagerConfig()) {
}

// 获取配置
BlockManagerConfig BlockManager::GetBlockManagerConfig() {
    BlockManagerConfig block_manager_config;
    Singleton<Environment>::GetInstance()->GetBlockManagerConfig(block_manager_config);
    return block_manager_config;
}

// 构造函数，根据给定的 BlockManagerConfig 配置 BlockManager
BlockManager::BlockManager(const BlockManagerConfig &block_manager_config) : gpu_allocator(block_manager_config.gpu_allocator_config), cpu_allocator(block_manager_config.cpu_allocator_config){
    block_size_ = block_manager_config.gpu_allocator_config.block_size;
    NLLM_LOG_INFO << "BlockManager Init Success";
}

// 析构函数，释放BlockManager分配的所有内存
BlockManager::~BlockManager() {
}

// 根据给定的block_ids，获取对应的内存指针，存储在addrs中
Status BlockManager::GetGpuBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) {
    return gpu_allocator.GetBlockPtrs(blocks, addrs);
}

// 分配block_num个块，将分配成功的块的id存储在blocks中
Status BlockManager::AllocateGpuBlocks(int64_t block_num, std::vector<int>& blocks){
    return gpu_allocator.Allocate(block_num, blocks);
}

// 分配指定大小的显存空间
Status BlockManager::Allocate(void*& gpu_memory, int64_t size){
    std::unique_lock<std::mutex> lock(contiguous_memory_mutex_);
    CUDA_CHECK(cudaHostAlloc(&gpu_memory, size, cudaHostAllocDefault));
    used_contiguous_memory_map_.insert({gpu_memory, size});
    return Status();
}

// 释放给定的blocks，将它们从used_gpu_block_map_移动到free_gpu_block_map_
Status BlockManager::FreeGpuBlocks(std::vector<int>& blocks) {
    return gpu_allocator.Free(blocks);
}

// 释放连续显存
Status BlockManager::Free(void* gpu_memory){
    std::unique_lock<std::mutex> lock(contiguous_memory_mutex_);
    auto it = used_contiguous_memory_map_.find(gpu_memory);
    if (it != used_contiguous_memory_map_.end()) {
        CUDA_CHECK(cudaFreeHost(gpu_memory));
        used_contiguous_memory_map_.erase(it);
    } else {
        return Status(RET_FREE_FAIL, fmt::format("free error, gpu_memory {}" , gpu_memory));
    }
    return Status();
}

Status BlockManager::SwapGpuToCpu(std::vector<int>& gpu_blocks, cudaStream_t stream) {
    std::unique_lock<std::mutex> lock(swap_mutex_);
    std::vector<void*> gpu_addrs;
    STATUS_CHECK_RETURN(GetGpuBlockPtrs(gpu_blocks, gpu_addrs));
    std::vector<int> cpu_blocks;
    std::vector<void*> cpu_addrs;
    STATUS_CHECK_RETURN(cpu_allocator.Allocate(gpu_blocks.size(), cpu_blocks));
    STATUS_CHECK_RETURN(cpu_allocator.GetBlockPtrs(cpu_blocks, cpu_addrs));
    for (int i = 0; i < gpu_blocks.size(); i++) {
        swap_map_[gpu_blocks[i]] = cpu_blocks[i];
        CUDA_CHECK(cudaMemcpyAsync(cpu_addrs[i], gpu_addrs[i], block_size_, cudaMemcpyDeviceToHost, stream));
    }
    return Status();
}

Status BlockManager::SwapCpuToGpu(std::vector<int>& gpu_blocks, cudaStream_t stream) {
    std::unique_lock<std::mutex> lock(swap_mutex_);
    std::vector<void*> gpu_addrs;
    STATUS_CHECK_RETURN(GetGpuBlockPtrs(gpu_blocks, gpu_addrs));
    std::vector<int> cpu_blocks(1);
    std::vector<void*> cpu_addrs(1);
    for (int i = 0; i < gpu_blocks.size(); i++) {
        auto it = swap_map_.find(gpu_blocks[i]);
        if (it != swap_map_.end()) {
            cpu_blocks[0] = it->second;
            STATUS_CHECK_RETURN(cpu_allocator.GetBlockPtrs(cpu_blocks, cpu_addrs));
            CUDA_CHECK(cudaMemcpyAsync(gpu_addrs[i], cpu_addrs[0], block_size_, cudaMemcpyHostToDevice, stream));
            STATUS_CHECK_RETURN(cpu_allocator.Free(cpu_blocks));
            swap_map_.erase(it);
        } else {
          return Status(RET_SEGMENT_FAULT);
        }
    }
    return Status();
}

// 函数：获取指定设备类型的空闲内存块数量
// 参数：device - 设备类型
// 返回值：空闲内存块数量
int64_t BlockManager::GetFreeBlockNumber(MemoryDevice device){
    switch (device){
        case MEMORY_CPU_PINNED:
            return cpu_allocator.GetFreeBlockNumber();

        case MEMORY_GPU:
            return gpu_allocator.GetFreeBlockNumber();
        
        default:
            return 0;
    }
}


}  // namespace numerous_llm
