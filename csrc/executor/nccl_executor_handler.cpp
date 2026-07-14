/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "executor/nccl_executor_handler.h"

#ifdef ENABLE_CUDA
#  include "cache_manager/block_allocator/block_allocator_manager.h"
#  include "connector/task_key.h"
#  include "device/nvidia/cuda_utils.h"
#  include "device/nvidia/nccl_utils.h"

namespace ksana_llm {

NcclExecutorHandler::NcclExecutorHandler(int device_id, ConnectorConfig connector_config)
    : device_id_(device_id), connector_config_(connector_config) {
  // 初始化GPU显存池(Send/Recv)
  size_t buffer_pool_size = connector_config.send_thread_num << 1;
  gpu_buffer_pool_ = std::make_unique<BlockingQueue<void*>>(buffer_pool_size);
  size_t allocated_buffer_size = connector_config.transfer_batch * sizeof(TaskKey);

  // 预分配显存块并放入池中
  for (size_t i = 0; i < buffer_pool_size; ++i) {
    void* gpu_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&gpu_buf, allocated_buffer_size));
    gpu_buffer_pool_->Put(gpu_buf);
    KLLM_LOG_INFO << "Allocated GPU buffer " << i << " of size " << allocated_buffer_size << " bytes at " << gpu_buf;
  }

  KLLM_LOG_INFO << "NcclExecutorHandler created for device " << device_id_;
}

NcclExecutorHandler::~NcclExecutorHandler() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [group_key, group_info] : comm_groups_) {
    if (group_info) {
      if (group_info->comm) {
        ncclCommDestroy(group_info->comm);
      }
      if (group_info->send_stream) {
        cudaStreamDestroy(group_info->send_stream);
      }
      if (group_info->recv_stream) {
        cudaStreamDestroy(group_info->recv_stream);
      }
    }
  }
  comm_groups_.clear();

  // 释放GPU显存池
  if (gpu_buffer_pool_) {
    size_t buffer_pool_size = connector_config_.send_thread_num << 1;
    while (buffer_pool_size != 0) {
      void* gpu_buf = gpu_buffer_pool_->Get();
      if (gpu_buf != nullptr) {
        CUDA_CHECK(cudaFree(gpu_buf));
        buffer_pool_size--;
      }
    }
    gpu_buffer_pool_->Stop();
    KLLM_LOG_INFO << "Freed " << (connector_config_.send_thread_num << 1) << " GPU buffers";
  }

  KLLM_LOG_INFO << "NcclExecutorHandler destroyed for device " << device_id_;
}

void NcclExecutorHandler::SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> block_allocator_manager,
                                                   int rank, const RuntimeConfig& runtime_config) {
  // Store parameters for lazy initialization
  block_allocator_manager_ = block_allocator_manager;
  rank_ = rank;
  attn_tensor_parallel_size_ = runtime_config.parallel_basic_config.attn_tensor_parallel_size;

  KLLM_LOG_INFO << "BlockAllocatorManager parameters stored for NcclExecutorHandler on device " << device_id_
                << ", rank=" << rank << ", attn_tensor_parallel_size=" << attn_tensor_parallel_size_;
}

Status NcclExecutorHandler::InitializeCommGroup(const std::string& group_key, const ncclUniqueId& comm_id, int rank,
                                                int world_size) {
  std::lock_guard<std::mutex> lock(mutex_);

  // 检查是否已存在
  if (comm_groups_.find(group_key) != comm_groups_.end()) {
    KLLM_LOG_WARNING << "Communication group " << group_key << " already exists";
    return Status();
  }
  KLLM_LOG_INFO << "Initializing NCCL comm group " << group_key << " on device " << device_id_ << " with rank " << rank
                << "/" << world_size;
  auto group_info = std::make_unique<CommGroupInfo>();
  group_info->rank = rank;
  group_info->world_size = world_size;

  // 创建CUDA流
  CUDA_CHECK(cudaStreamCreate(&group_info->send_stream));
  CUDA_CHECK(cudaStreamCreate(&group_info->recv_stream));

  // 初始化NCCL通信器
  ncclResult_t result = ncclCommInitRank(&group_info->comm, world_size, comm_id, rank);
  if (result != ncclSuccess) {
    KLLM_LOG_ERROR << "Failed to initialize NCCL communicator for group " << group_key << ": "
                   << ncclGetErrorString(result);
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Failed to initialize NCCL communicator");
  }

  comm_groups_[group_key] = std::move(group_info);
  KLLM_LOG_INFO << "Initialized NCCL comm group " << group_key << " on device " << device_id_ << " with rank " << rank
                << "/" << world_size;

  return Status();
}

Status NcclExecutorHandler::Send(const std::string& group_key, int peer_rank, const void* host_buf, size_t count,
                                 DataType dtype) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = comm_groups_.find(group_key);
  if (it == comm_groups_.end() || !it->second || !it->second->comm) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Communication group not found: " + group_key);
  }

  auto& group_info = it->second;

  // 使用group_info中的send_stream
  cudaStream_t stream = group_info->send_stream;

  // 从显存池获取GPU内存
  void* gpu_buf = gpu_buffer_pool_->Get();
  if (gpu_buf == nullptr) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Failed to get GPU buffer from pool");
  }

  // 将host数据复制到GPU
  CUDA_CHECK(cudaMemcpyAsync(gpu_buf, host_buf, count, cudaMemcpyHostToDevice, stream));

  // 执行NCCL Send
  ncclDataType_t nccl_dtype;
  GetNcclDataType(dtype, nccl_dtype);
  peer_rank = (group_info->world_size / 2 + peer_rank + (group_info->rank - device_id_)) % group_info->world_size;
  CUDA_CHECK(cudaStreamSynchronize(stream));
  KLLM_LOG_DEBUG << "NCCL Send: group=" << group_key << ", peer_rank=" << peer_rank << ", rank=" << group_info->rank
                 << ", count=" << count << ", device=" << device_id_ << ", dtype=" << nccl_dtype
                 << ", stream=" << stream << ", gpu_buf=" << gpu_buf;
  NCCL_CHECK(ncclSend(gpu_buf, count, nccl_dtype, peer_rank, group_info->comm, stream));
  // 同步并归还缓冲区到池中
  CUDA_CHECK(cudaStreamSynchronize(stream));
  gpu_buffer_pool_->Put(gpu_buf);
  KLLM_LOG_DEBUG << "NCCL Send: group=" << group_key << ", peer_rank=" << peer_rank << ", count=" << count
                 << ", device=" << device_id_;
  return Status();
}

Status NcclExecutorHandler::Recv(const std::string& group_key, int peer_rank, void* host_buf, size_t count,
                                 DataType dtype) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = comm_groups_.find(group_key);
  if (it == comm_groups_.end() || !it->second || !it->second->comm) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Communication group not found: " + group_key);
  }

  auto& group_info = it->second;

  // 使用group_info中的recv_stream
  cudaStream_t stream = group_info->recv_stream;

  // 从显存池获取GPU内存
  void* gpu_buf = gpu_buffer_pool_->Get();
  if (gpu_buf == nullptr) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Failed to get GPU buffer from pool");
  }

  // 执行NCCL Recv
  ncclDataType_t nccl_dtype;
  GetNcclDataType(dtype, nccl_dtype);
  peer_rank = (group_info->world_size / 2 + peer_rank + (group_info->rank - device_id_)) % group_info->world_size;
  CUDA_CHECK(cudaStreamSynchronize(stream));
  KLLM_LOG_DEBUG << "NCCL Recv: group=" << group_key << ", peer_rank=" << peer_rank << ", rank=" << group_info->rank
                 << ", count=" << count << ", device=" << device_id_ << ", dtype=" << nccl_dtype
                 << ", stream=" << stream << ", gpu_buf=" << gpu_buf;
  NCCL_CHECK(ncclRecv(gpu_buf, count, nccl_dtype, peer_rank, group_info->comm, stream));
  // 将GPU数据复制到host
  CUDA_CHECK(cudaMemcpyAsync(host_buf, gpu_buf, count, cudaMemcpyDeviceToHost, stream));

  // 同步并归还缓冲区到池中
  CUDA_CHECK(cudaStreamSynchronize(stream));
  gpu_buffer_pool_->Put(gpu_buf);

  KLLM_LOG_DEBUG << "NCCL Recv: group=" << group_key << ", peer_rank=" << peer_rank << ", count=" << count
                 << ", device=" << device_id_;

  return Status();
}

Status NcclExecutorHandler::SendGroup(const std::string& group_key, int peer_rank, const std::vector<int>& block_ids,
                                      const std::vector<int64_t>& offsets, const std::vector<size_t>& counts,
                                      DataType dtype) {
  if (block_ids.empty() || block_ids.size() != offsets.size() || block_ids.size() != counts.size()) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Invalid block_ids, offsets or counts");
  }

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = comm_groups_.find(group_key);
  if (it == comm_groups_.end() || !it->second || !it->second->comm) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Communication group not found: " + group_key);
  }

  auto& group_info = it->second;

  // 解析GPU指针
  std::vector<void*> buffers;
  Status status = ResolveBlockPtrs(block_ids, offsets, buffers);
  if (!status.OK()) {
    return status;
  }

  // 使用group_info中的send_stream
  cudaStream_t stream = group_info->send_stream;

  ncclDataType_t nccl_dtype;
  GetNcclDataType(dtype, nccl_dtype);
  NCCL_CHECK(ncclGroupStart());
  peer_rank = (group_info->world_size / 2 + peer_rank + (group_info->rank - device_id_)) % group_info->world_size;
  for (size_t i = 0; i < buffers.size(); ++i) {
    NCCL_CHECK(ncclSend(buffers[i], counts[i], nccl_dtype, peer_rank, group_info->comm, stream));
  }
  NCCL_CHECK(ncclGroupEnd());

  // 同步
  CUDA_CHECK(cudaStreamSynchronize(stream));

  KLLM_LOG_DEBUG << "NCCL SendGroup: group=" << group_key << ", peer_rank=" << peer_rank
                 << ", buffer_count=" << buffers.size() << ", device=" << device_id_;

  return Status();
}

Status NcclExecutorHandler::RecvGroup(const std::string& group_key, int peer_rank, const std::vector<int>& block_ids,
                                      const std::vector<int64_t>& offsets, const std::vector<size_t>& counts,
                                      DataType dtype) {
  if (block_ids.empty() || block_ids.size() != offsets.size() || block_ids.size() != counts.size()) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Invalid block_ids, offsets or counts");
  }

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = comm_groups_.find(group_key);
  if (it == comm_groups_.end() || !it->second || !it->second->comm) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Communication group not found: " + group_key);
  }

  auto& group_info = it->second;

  // 解析GPU指针
  std::vector<void*> buffers;
  Status status = ResolveBlockPtrs(block_ids, offsets, buffers);
  if (!status.OK()) {
    return status;
  }

  // 使用group_info中的recv_stream
  cudaStream_t stream = group_info->recv_stream;

  ncclDataType_t nccl_dtype;
  GetNcclDataType(dtype, nccl_dtype);

  peer_rank = (group_info->world_size / 2 + peer_rank + (group_info->rank - device_id_)) % group_info->world_size;
  NCCL_CHECK(ncclGroupStart());
  for (size_t i = 0; i < buffers.size(); ++i) {
    NCCL_CHECK(ncclRecv(buffers[i], counts[i], nccl_dtype, peer_rank, group_info->comm, stream));
  }
  NCCL_CHECK(ncclGroupEnd());

  // 同步
  CUDA_CHECK(cudaStreamSynchronize(stream));

  KLLM_LOG_DEBUG << "NCCL RecvGroup: group=" << group_key << ", peer_rank=" << peer_rank
                 << ", buffer_count=" << buffers.size() << ", device=" << device_id_;

  return Status();
}

Status NcclExecutorHandler::GetCommGroupInfo(const std::string& group_key, int& rank, int& world_size) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = comm_groups_.find(group_key);
  if (it == comm_groups_.end() || !it->second) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Communication group not found: " + group_key);
  }

  rank = it->second->rank;
  world_size = it->second->world_size;
  return Status();
}

Status NcclExecutorHandler::EnsureDeviceAllocatorInitialized() {
  // Double-checked locking pattern for thread-safe lazy initialization
  if (device_allocator_) {
    return Status();
  }

  if (!block_allocator_manager_) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR,
                  "BlockAllocatorManager not set. Call SetBlockAllocatorManager first.");
  }

  if (rank_ < 0 || attn_tensor_parallel_size_ <= 0) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR,
                  "Invalid rank or attn_tensor_parallel_size. Call SetBlockAllocatorManager first.");
  }

  // Calculate attn_dp_group_id and attn_dp_rank_id (reference: model_input.cpp:52-54)
  int attn_dp_group_id = rank_ / attn_tensor_parallel_size_;
  int attn_dp_rank_id = rank_ % attn_tensor_parallel_size_;

  // Get and store device_allocator (reference: model_input.cpp:1240-1241)
  device_allocator_ =
      block_allocator_manager_->GetBlockAllocatorGroup(attn_dp_group_id)->GetDeviceBlockAllocator(attn_dp_rank_id);

  KLLM_LOG_INFO << "device_allocator initialized lazily for NcclExecutorHandler on device " << device_id_
                << ", attn_dp_group_id=" << attn_dp_group_id << ", attn_dp_rank_id=" << attn_dp_rank_id;

  return Status();
}

Status NcclExecutorHandler::ResolveBlockPtrs(const std::vector<int>& block_ids, const std::vector<int64_t>& offsets,
                                             std::vector<void*>& ptrs) {
  // Lazy initialization on first use
  Status init_status = EnsureDeviceAllocatorInitialized();
  if (!init_status.OK()) {
    return init_status;
  }

  if (block_ids.size() != offsets.size()) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "block_ids and offsets size mismatch");
  }

  // 获取block基地址
  std::vector<void*> block_base_ptrs;
  Status status = device_allocator_->GetBlockPtrs(block_ids, block_base_ptrs);
  if (!status.OK()) {
    return status;
  }

  // 计算最终指针（基地址 + 偏移量）
  ptrs.resize(block_ids.size());
  for (size_t i = 0; i < block_ids.size(); ++i) {
    ptrs[i] = static_cast<char*>(block_base_ptrs[i]) + offsets[i];
  }
  return Status();
}

}  // namespace ksana_llm

#endif  // ENABLE_CUDA