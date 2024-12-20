/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <filesystem>
#include <numeric>

#include "ksana_llm/block_manager/block_manager.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "tensor.h"
#include "test.h"

namespace ksana_llm {

void SetTestBlockManager() {
  std::filesystem::path current_path = __FILE__;
  std::filesystem::path parent_path = current_path.parent_path();
  std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm.yaml";
  std::string config_path = std::filesystem::absolute(config_path_relate).string();

  Singleton<Environment>::GetInstance()->ParseConfig(config_path);

  BlockManagerConfig block_manager_config;
  Singleton<Environment>::GetInstance()->InitializeBlockManagerConfig();
  Singleton<Environment>::GetInstance()->GetBlockManagerConfig(block_manager_config);
  BlockManager* block_manager = new BlockManager(block_manager_config, std::make_shared<Context>(1, 1));
  SetBlockManager(block_manager);
}

TEST(WorkspaceTest, CommonTest) {
  SetTestBlockManager();
  WorkSpaceFunc f = GetWorkSpaceFunc();

  void* ws_addr_1 = nullptr;
  f(1024, &ws_addr_1);

  void* ws_addr_2 = nullptr;
  f(2048, &ws_addr_2);
#ifdef ENABLE_CUDA
  EXPECT_NE(ws_addr_1, ws_addr_2);
#endif

  void* ws_addr_3 = nullptr;
  f(1536, &ws_addr_3);
  EXPECT_EQ(ws_addr_2, ws_addr_3);
}

TEST(TensorTest, CommonTest) {
  SetTestBlockManager();

  constexpr int tensor_parallel_size = 1;
  constexpr int pipeline_parallel_size = 1;
  std::shared_ptr<Context> context = std::make_shared<Context>(tensor_parallel_size, pipeline_parallel_size);
  constexpr size_t ELEM_NUM = 16;
  constexpr int RANK = 0;

  std::vector<int32_t> src_data(ELEM_NUM, 0);

  std::iota(src_data.begin(), src_data.end(), 1);
  Tensor tensor_with_block_id_on_host;
  Tensor tensor_with_refer_ptr_on_host;
  Tensor tensor_with_block_id_on_dev;
  Tensor tensor_with_refer_ptr_on_dev;
  Tensor output_data;

  STATUS_CHECK_FAILURE(CreateTensor(tensor_with_block_id_on_host, {ELEM_NUM}, TYPE_INT32, RANK, MEMORY_HOST));
  STATUS_CHECK_FAILURE(
      CreateTensor(tensor_with_refer_ptr_on_host, {ELEM_NUM}, TYPE_INT32, RANK, MEMORY_HOST, src_data.data()));
  for (size_t src_idx = 0; src_idx < src_data.size(); ++src_idx) {
    EXPECT_EQ(tensor_with_refer_ptr_on_host.GetPtr<int32_t>()[src_idx], src_data[src_idx]);
  }
  std::memcpy(tensor_with_block_id_on_host.GetPtr<int32_t>(), tensor_with_refer_ptr_on_host.GetPtr<int32_t>(),
              tensor_with_block_id_on_host.GetTotalBytes());
  for (size_t src_idx = 0; src_idx < src_data.size(); ++src_idx) {
    EXPECT_EQ(tensor_with_block_id_on_host.GetPtr<int32_t>()[src_idx], src_data[src_idx]);
  }
  STATUS_CHECK_FAILURE(DestroyTensor(tensor_with_refer_ptr_on_host, RANK));
  STATUS_CHECK_FAILURE(DestroyTensor(tensor_with_block_id_on_host, RANK));

  STATUS_CHECK_FAILURE(CreateTensor(tensor_with_block_id_on_dev, {ELEM_NUM}, TYPE_INT32, RANK, MEMORY_DEVICE));
  MemcpyAsync(tensor_with_block_id_on_dev.GetPtr<int32_t>(), src_data.data(),
              tensor_with_block_id_on_dev.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE, context->GetH2DStreams()[RANK]);
  StreamSynchronize(context->GetH2DStreams()[RANK]);
  STATUS_CHECK_FAILURE(CreateTensor(tensor_with_refer_ptr_on_dev, {ELEM_NUM}, TYPE_INT32, RANK, MEMORY_DEVICE,
                                    tensor_with_block_id_on_dev.GetPtr<int32_t>()));
  MemcpyAsync(tensor_with_refer_ptr_on_dev.GetPtr<int32_t>(), tensor_with_block_id_on_dev.GetPtr<int32_t>(),
              tensor_with_refer_ptr_on_dev.GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE, context->GetD2DStreams()[RANK]);
  StreamSynchronize(context->GetD2DStreams()[RANK]);
  STATUS_CHECK_FAILURE(CreateTensor(output_data, {ELEM_NUM}, TYPE_INT32, RANK, MEMORY_HOST));
  MemcpyAsync(output_data.GetPtr<int32_t>(), tensor_with_refer_ptr_on_dev.GetPtr<int32_t>(),
              tensor_with_refer_ptr_on_dev.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST, context->GetD2HStreams()[RANK]);
  StreamSynchronize(context->GetD2HStreams()[RANK]);
  for (size_t src_idx = 0; src_idx < src_data.size(); ++src_idx) {
    EXPECT_EQ(output_data.GetPtr<int32_t>()[src_idx], src_data[src_idx]);
  }
  EXPECT_EQ(tensor_with_refer_ptr_on_dev.GetPtr<int32_t>(), tensor_with_block_id_on_dev.GetPtr<int32_t>());
  STATUS_CHECK_FAILURE(DestroyTensor(output_data, RANK));
  STATUS_CHECK_FAILURE(DestroyTensor(tensor_with_refer_ptr_on_dev, RANK));
  STATUS_CHECK_FAILURE(DestroyTensor(tensor_with_block_id_on_dev, RANK));
  STATUS_CHECK_FAILURE(DestroyTensor(tensor_with_block_id_on_host, RANK));
  STATUS_CHECK_FAILURE(DestroyTensor(tensor_with_refer_ptr_on_host, RANK));
}

}  // namespace ksana_llm
