/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <Python.h>
#include <filesystem>

#include "ksana_llm/block_manager/block_manager.h"
#include "ksana_llm/models/llama/llama_model.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/calc_intvec_hash.h"
#include "ksana_llm/utils/get_custom_weight_name.h"
#include "ksana_llm/utils/search_path.h"
#include "ksana_llm/utils/singleton.h"
#include "test.h"

using namespace ksana_llm;

// 定义一个 LlamaTest 类,继承自 testing::Test
class LlamaTest : public testing::Test {
 protected:
  void SetUp() override {
    context_ = std::make_shared<Context>(1, 1);

    // 解析 config.json,初始化 ModelConfig 以及 BlockManager
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    Singleton<Environment>::GetInstance()->ParseConfig(config_path);
    Singleton<Environment>::GetInstance()->GetModelConfig("", model_config);

    BlockManagerConfig block_manager_config;
    Singleton<Environment>::GetInstance()->GetBlockManagerConfig(block_manager_config);
    KLLM_LOG_DEBUG << fmt::format("block_size {}", block_manager_config.device_allocator_config.block_size);

    block_manager = new BlockManager(block_manager_config, context_);
    block_manager->PreAllocateBlocks();
    SetBlockManager(block_manager);
  }

  void TearDown() override { delete block_manager; }

 protected:
  ModelConfig model_config;
  BlockManager *block_manager = nullptr;

  std::shared_ptr<Context> context_{nullptr};

  template <typename weight_data_type>
  void TestLlamaForward() {
    int device_id = 0;
    SetDevice(device_id);
#ifdef ENABLE_FP8
    if (model_config.quant_config.method == QUANT_FP8_E4M3 && !context_->IsGemmFp8Supported()) {
      std::cout << "Cublas is insufficient to support FP8, skip test." << std::endl;
      return;
    }
#endif
    std::filesystem::path model_path(model_config.path);
    if (!std::filesystem::exists(model_path)) {
      KLLM_LOG_ERROR << fmt::format("The given model path {} does not exist.", model_config.path);
      EXPECT_TRUE(std::filesystem::exists(model_path));
    }
    Event start;
    Event stop;
    float milliseconds = 0;
    int rounds = 10;
    EventCreate(&start);
    EventCreate(&stop);

    std::shared_ptr<BaseWeight> llama_weight =
        std::make_shared<LlamaWeight<weight_data_type>>(model_config, 0, context_);
    // Start Loader Weight
    bool is_safetensors = false;
    std::vector<std::string> weights_file_list = SearchLocalPath(model_path, is_safetensors);
    for (std::string &file_name : weights_file_list) {
      std::shared_ptr<BaseFileTensorLoader> weights_loader = nullptr;
      if (is_safetensors) {
        weights_loader = std::make_shared<SafeTensorsLoader>(file_name);
      } else {
        weights_loader = std::make_shared<PytorchFileTensorLoader>(file_name);
      }
      std::vector<std::string> weight_name_list = weights_loader->GetTensorNameList();
      std::vector<std::string> custom_name_list;
      GetCustomNameList(weight_name_list, custom_name_list, model_config.path, model_config.type);

      llama_weight->LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list);
      StreamSynchronize(context_->GetMemoryManageStreams()[device_id]);
    }
    llama_weight->ProcessWeights();  // End Loader Weight
    std::shared_ptr<LlamaModel<weight_data_type>> llama =
        std::make_shared<LlamaModel<weight_data_type>>(model_config, 0, context_, llama_weight);

    // Weight Name Check
    // 正确的 weight 名称
    std::string weight_name = "lm_head.weight";
    Tensor lm_head = llama_weight->GetModelWeights(weight_name);
    EXPECT_EQ(lm_head.device, MEMORY_DEVICE);
    EXPECT_EQ(lm_head.shape, std::vector<size_t>({4096, 32000}));

    // 错误的 weight 名称
    weight_name = "wrong_name";
    Tensor wrong_tensor = llama_weight->GetModelWeights(weight_name);
    EXPECT_EQ(wrong_tensor.device, MEMORY_HOST);
    EXPECT_TRUE(wrong_tensor.shape.empty());

    // ContextDecode
    ForwardRequest forward;
    std::vector<int> input_ids = {233, 1681};
    forward.output_tokens = &input_ids;
    forward.logits_buf.resize(1);
    forward.logits_buf[0] = llama->GetLogitsPtr();
    forward.logits_offset = 0;
    std::vector<int> input_refit_pos;
    std::vector<std::vector<float>> input_refit_embedding;
    EmbeddingSlice embedding_slice;
    embedding_slice.pos = input_refit_pos;
    embedding_slice.embeddings = input_refit_embedding;
    forward.input_refit_embedding = &embedding_slice;
    std::vector<int> block_ids;
    GetBlockManager()->AllocateBlocks(1, block_ids);
    forward.kv_cache_ptrs.resize(1);
    GetBlockManager()->GetBlockPtrs(block_ids, forward.kv_cache_ptrs[0]);
#ifdef ENABLE_ACL
    // for rank_0
    forward.atb_kv_cache_base_blk_ids.clear();
    forward.atb_kv_cache_base_blk_ids.resize(1);
    // prepare base block ids in rank_0
    int32_t origin_block_id =
        (uintptr_t(forward.kv_cache_ptrs[0][0]) - uintptr_t(GetBlockManager()->GetBlockBasePtr())) /
        (2 * model_config.num_layer * model_config.block_token_num * model_config.head_num *
         model_config.size_per_head) /
        sizeof(float16);
    forward.atb_kv_cache_base_blk_ids[0].push_back(
        {static_cast<int32_t>(origin_block_id * 2 * model_config.num_layer)});
#endif
    Memset(forward.kv_cache_ptrs[0][0], 0, GetBlockManager()->GetBlockSize());
    KLLM_LOG_DEBUG << fmt::format("kv_cache_ptrs {} end {}", forward.kv_cache_ptrs[0][0],
                                  forward.kv_cache_ptrs[0][0] + (GetBlockManager()->GetBlockSize()));
    std::vector<ForwardRequest> forward_reqs = {forward};
    EXPECT_TRUE(llama->ContextDecode(llama_weight, forward_reqs).OK());

    std::vector<ForwardRequest> multi_forward_reqs = {forward, forward};
    EventRecord(start, context_->GetComputeStreams()[device_id]);
    for (int i = 0; i < rounds; ++i) {
      llama->ContextDecode(llama_weight, multi_forward_reqs);
    }
    EventRecord(stop, context_->GetComputeStreams()[device_id]);
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    std::cout << "ContextDecode milliseconds / " << rounds << " is: " << milliseconds / rounds << std::endl;

#ifdef ENABLE_CUDA
    EXPECT_TRUE((milliseconds / rounds) < 35);
#else
    // NOTE(karlluo): ACL inference is slower than CUDA
    EXPECT_TRUE((milliseconds / rounds) < 300) << "milliseconds / " << rounds << " is: " << milliseconds / rounds;
#endif

    // Sampling
    SamplingRequest sample_req;
    NgramDict ngram_dict;
    std::vector<std::vector<std::pair<int, float>>> logprobs;
    std::vector<float> prompt_probs;
    sample_req.input_tokens = &input_ids;
    sample_req.logits_offset = forward_reqs[0].logits_offset;
    sample_req.output_tokens = forward_reqs[0].output_tokens;
    sample_req.logprobs = &logprobs;
    sample_req.ngram_dict = &ngram_dict;
    std::mutex output_mutex;
    sample_req.output_mutex = &output_mutex;
    sample_req.logits_buf = forward_reqs[0].logits_buf;
    sample_req.model_config = &model_config;
    SamplingConfig sample_config;
    sample_config.num_beams = 1;
    sample_config.topk = 1;
    sample_config.topp = 0;
    sample_config.temperature = 0;
    sample_config.repetition_penalty = 1;
    sample_config.no_repeat_ngram_size = 0;
    sample_config.encoder_no_repeat_ngram_size = 0;
    sample_req.sampling_config = &sample_config;
    BatchSchedulerConfig batch_scheduler_config;
    Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);

    std::vector<SamplingRequest> sample_reqs = {sample_req};
    std::shared_ptr<Sampler> sampler = std::make_shared<Sampler>(batch_scheduler_config, device_id, context_);
    sampler->Sampling(sample_reqs, context_->GetComputeStreams()[device_id]);
    EXPECT_EQ(29871, (*forward_reqs[0].output_tokens)[2]);

    // Decode
    EXPECT_TRUE(llama->Decode(llama_weight, forward_reqs).OK());
    sampler->Sampling(sample_reqs, context_->GetComputeStreams()[device_id]);
    EXPECT_EQ(29896, (*forward_reqs[0].output_tokens)[3]);

#ifdef ENABLE_CUDA
    EXPECT_TRUE(llama->Decode(llama_weight, forward_reqs).OK());
    sampler->Sampling(sample_reqs, context_->GetComputeStreams()[device_id]);
    EXPECT_EQ(29929, (*forward_reqs[0].output_tokens)[4]);
#endif

    EventRecord(start, context_->GetComputeStreams()[device_id]);
    for (auto &forward_req : multi_forward_reqs) {
      forward_req.step = 1;
    }
    for (int i = 0; i < rounds; ++i) {
      llama->Decode(llama_weight, multi_forward_reqs);
    }
    EventRecord(stop, context_->GetComputeStreams()[device_id]);
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    std::cout << "Decode milliseconds / " << rounds << " is: " << milliseconds / rounds << std::endl;

#ifdef ENABLE_CUDA
    EXPECT_TRUE((milliseconds / rounds) < 30);

    // Test logits_custom_length
    std::vector<int> prompt_probs_input_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    forward.output_tokens = &prompt_probs_input_tokens;
    forward.logits_custom_length = 5;
    std::map<std::string, ksana_llm::TargetDescribe> request_target;
    ksana_llm::TargetDescribe target_describe;
    target_describe.slice_pos.push_back({0, 4});
    request_target["logits"] = target_describe;
    forward.request_target = &request_target;
    std::vector<ForwardRequest> prompt_probs_forward_reqs = {forward, forward};
    ModelInput model_input(model_config, 0, context_);
    model_input.ParseFromRequests(prompt_probs_forward_reqs);
    EXPECT_TRUE(model_input.use_logits_custom_length);
    std::vector<uint64_t> result(model_input.logits_custom_length_uint64_tensor.GetElementNumber());
    std::vector<uint64_t> dst = {0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14};
    Memcpy(result.data(), model_input.logits_custom_length_uint64_tensor.GetPtr<void>(),
           result.size() * sizeof(uint64_t), MEMCPY_DEVICE_TO_HOST);
    EXPECT_EQ(dst.size(), result.size());
    for (size_t i = 0; i < result.size(); i++) {
      EXPECT_EQ(result[i], dst[i]);
    }
    EXPECT_TRUE(model_input.use_logits_custom_length);
    EXPECT_TRUE(llama->ContextDecode(llama_weight, prompt_probs_forward_reqs).OK());
#else
    // NOTE(karlluo): ACL inference is slower than CUDA
    EXPECT_TRUE((milliseconds / rounds) < 300) << "milliseconds / " << rounds << " is: " << milliseconds / rounds;
#endif

    llama.reset();
    llama_weight.reset();

    StreamSynchronize(context_->GetMemoryManageStreams()[device_id]);
    EventDestroy(stop);
    EventDestroy(start);
    DeviceSynchronize();
  }
};

TEST_F(LlamaTest, ForwardTest) {
  Py_Initialize();
  // fp16 forward
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_FP16;
  model_config.quant_config.method = QUANT_NONE;
  std::cout << "Test TYPE_FP16 weight_data_type forward." << std::endl;
  TestLlamaForward<float16>();
#ifdef ENABLE_FP8
  // fp8 forward
  model_config.is_quant = true;
  model_config.quant_config.method = QUANT_FP8_E4M3;
  model_config.quant_config.is_checkpoint_fp8_serialized = false;
  std::cout << "Test TYPE_FP16 weight_data_type with QUANT_FP8_E4M3 forward" << std::endl;
  TestLlamaForward<float16>();
#endif

#ifdef ENABLE_CUDA
#  ifdef ENABLE_BFLOAT16
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_BF16;
  model_config.quant_config.method = QUANT_NONE;
  std::cout << "Test TYPE_BF16 weight_data_type forward." << std::endl;
  TestLlamaForward<bfloat16>();
#    ifdef ENABLE_FP8
  // fp8 forward
  model_config.is_quant = true;
  model_config.quant_config.method = QUANT_FP8_E4M3;
  model_config.quant_config.is_checkpoint_fp8_serialized = false;
  std::cout << "Test TYPE_BF16 weight_data_type with QUANT_FP8_E4M3 forward" << std::endl;
  TestLlamaForward<bfloat16>();
#    endif
#  endif
#endif
  Py_Finalize();
}

TEST(TorchTensorTest, TorchTensorTest) {
#ifdef ENABLE_CUDA
  int device_id = 0;
  SetDevice(device_id);
  // 设定张量的大小
  const int64_t size = 10;

  // 在GPU上分配内存
  void *a_ptr, *b_ptr, *c_ptr;
  Malloc(&a_ptr, size * sizeof(float));
  Malloc(&b_ptr, size * sizeof(float));
  Malloc(&c_ptr, size * sizeof(float));

  // 创建并初始化输入数据
  std::vector<float> a_host(size, 1.0), b_host(size, 2.0);

  // 将数据复制到GPU
  Memcpy(a_ptr, a_host.data(), size * sizeof(float), MEMCPY_HOST_TO_DEVICE);
  Memcpy(b_ptr, b_host.data(), size * sizeof(float), MEMCPY_HOST_TO_DEVICE);

  // 创建torch::Tensor，它们共享GPU内存
  auto options = torch::TensorOptions().device(torch::kCUDA, device_id).dtype(torch::kFloat32);
  torch::Tensor a = torch::from_blob(a_ptr, {size}, options);
  torch::Tensor b = torch::from_blob(b_ptr, {size}, options);
  torch::Tensor c = torch::from_blob(c_ptr, {size}, options);

  // 计算a + b = c
  c.copy_(a.add_(b));
  std::ostringstream oss;
  // 传输到cpu打印
  oss << c.to(torch::kCPU);
  EXPECT_EQ('3', oss.str()[1]);

  // 将结果复制回CPU以进行验证
  std::vector<float> c_host(size);
  Memcpy(c_host.data(), c_ptr, size * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  // 验证结果
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(c_host[i], 3.0);
  }

  // 清理GPU内存
  Free(a_ptr);
  Free(b_ptr);
  Free(c_ptr);
#endif
}
