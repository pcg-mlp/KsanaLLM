/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <algorithm>
#include <atomic>
#include <fstream>
#include <numeric>
#include <random>
#include <set>
#include <thread>

#include "3rdparty/half/include/half.hpp"
#include "device/common_device.h"
#include "device/device_types.h"
#include "layers/layer_workspace_manager.h"
#include "layers/layernorm_layer.h"
#include "test.h"
#include "utils/attention_backend/attention_backend_manager.h"
#include "utils/singleton.h"

#ifdef ENABLE_CUDA
#  include "csrc/device/nvidia/cuda_utils.h"
#  include "csrc/kernels/nvidia/cast/cast.h"
#  include "csrc/kernels/nvidia/permute/permute.h"
#  include "kernels/nvidia/kernel_wrapper.h"
#endif

#ifdef ENABLE_KUNLUN
#  include "device/kunlun/bkcl_utils.h"
#endif

namespace ksana_llm {

class LayerTest : public testing::Test {
 protected:
  void SetUp() override {
    model_config.path = "/model/llama-hf/7B/";
    model_config.weight_data_type = TYPE_FP16;
    model_config.head_num = 32;
    model_config.size_per_head = 128;
    model_config.inter_size = 11008;
    model_config.num_layer = 32;
    model_config.vocab_size = 32000;
    runtime_config.parallel_basic_config.tensor_parallel_size = 1;
    model_config.layernorm_eps = 1e-6;
    runtime_config.max_batch_size = 128;
    runtime_config.max_seq_len = 1024;
    model_config.rotary_embedding = 128;
    model_config.max_position_embeddings = 2048;
    model_config.rope_theta = 10000.0f;
    model_config.num_key_value_heads = model_config.head_num;

    BlockManagerConfig block_manager_config;
    block_manager_config.host_allocator_config.blocks_num = 2;
    block_manager_config.host_allocator_config.block_token_num = 16;
    block_manager_config.host_allocator_config.block_size = block_manager_config.host_allocator_config.block_token_num *
                                                            2 * model_config.head_num * model_config.size_per_head *
                                                            model_config.num_layer * sizeof(float16);
    block_manager_config.host_allocator_config.device = MEMORY_HOST;
    block_manager_config.device_allocator_config.blocks_num = 2;
    block_manager_config.device_allocator_config.block_token_num = 16;
    block_manager_config.device_allocator_config.block_size =
        block_manager_config.host_allocator_config.block_token_num * 2 * model_config.head_num *
        model_config.size_per_head * model_config.num_layer * sizeof(float16);
    KLLM_LOG_INFO << fmt::format("block_size {}", block_manager_config.device_allocator_config.block_size);
    block_manager_config.device_allocator_config.device = MEMORY_DEVICE;

    runtime_config.attn_backend_config.block_token_num = block_manager_config.device_allocator_config.block_token_num;
    runtime_config.attn_backend_config.block_size = block_manager_config.device_allocator_config.block_size;
    runtime_config.inter_data_type = model_config.weight_data_type;
    runtime_config.w4afp8_moe_backend = W4AFP8_MOE_BACKEND::Default;

    AttentionBackendManager::GetInstance()->Initialize();

    const auto& env = Singleton<Environment>::GetInstance();
    env->SetBlockManagerConfig(block_manager_config);

    context_ = std::make_shared<Context>();

    context_->global.reset(new GlobalContext(1, 1, 1, env));
    context_->global->InitDistributedConfig();

    context_->device = std::make_shared<DeviceContext>(1);
#ifdef ENABLE_CUDA
    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context_->device->ext->InitNcclParam(nccl_uid);
#elif defined(ENABLE_KUNLUN)
    // Single-card test: mirror production's single-card path. GenerateBKCLUniqueID() is
    // intentionally skipped because bkcl_get_unique_id() blocks on RDMA rendezvous; InitBkcl
    // no-ops the actual bkcl_init_rank when world_size <= 1, so a communicator is not created.
    context_->device->ext->InitBkcl(context_->device->GetDeviceId(), /*world_size=*/1);
#endif
  }

  void TearDown() override {}

  Status CreateHalfDataTypeTensor(Tensor& tensor, const std::vector<size_t>& shape, const DataType data_type,
                                  size_t dtype_size = 2) {
    tensor = Tensor(MemoryLocation::LOCATION_DEVICE, data_type, shape);
    return Status();
  }

 protected:
  ModelConfig model_config;
  RuntimeConfig runtime_config;
  std::shared_ptr<Context> context_{nullptr};
};

}  // namespace ksana_llm
