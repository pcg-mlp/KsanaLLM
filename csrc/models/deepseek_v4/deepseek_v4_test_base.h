/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "configure/schedule_config_parser.h"
#include "executor/inflight_resource.h"
#include "helpers/block_cache_layout_helper.h"
#include "helpers/model_test_helper.h"
#include "models/deepseek_v4/attention_meta/deepseek_v4_compressor_slot_resource.h"
#include "models/deepseek_v4/deepseek_v4_model.h"
#include "runtime/hidden_unit_buffer.h"
#include "runtime/weight_instance.h"
#include "samplers/sampler.h"
#include "test.h"
#include "utils/attention_backend/attention_backend_manager.h"
#include "utils/memory_allocator.h"
#include "utils/singleton.h"

namespace ksana_llm {

class DeepSeekV4TestBase : public testing::Test {
 protected:
  static constexpr int kDeviceId = 0;
  static inline const std::vector<int> kInputIds = {
      0,     0,     128803, 2788, 3655,  5979,  3099, 32200, 7624,  7524, 19,  16,    223,  1140,  2056,
      12519, 61320, 58788,  9090, 14721, 625,   303,  8040,  1612,  1049, 410, 31946, 303,  2788,  112467,
      718,   16227, 111162, 303,  1380,  32200, 8955, 7383,  10949, 20,   16,  223,   6094, 128804};

  // Subclasses assign this in SetUp() before calling Initialize().
  std::string model_path_;

  ModelConfig model_config;
  RuntimeConfig runtime_config;
  PipelineConfig pipeline_config;
  std::shared_ptr<BlockAllocatorManager> engine_block_manager;
  std::shared_ptr<BlockAllocatorManager> executor_block_manager;
  std::shared_ptr<BlockAllocatorGroupInterface> engine_block_group;
  std::shared_ptr<BlockAllocatorGroupInterface> executor_block_group;
  std::shared_ptr<BlockAllocatorManager> swa_block_manager_;
  std::shared_ptr<BlockAllocatorGroupInterface> swa_block_group_;
  std::shared_ptr<Context> context;
  std::shared_ptr<HiddenUnitBufferHelper> hidden_unit_buffer_helper;
  HiddenUnitDeviceBuffer* hidden_unit = nullptr;
  std::shared_ptr<BaseModelCacheLayout> block_cache_layout_;
  std::shared_ptr<InflightResourceManager> inflight_resource_mgr_;

  // yaml only satisfies Environment::ParseConfig; model_path_ drives the rest.
  void Initialize() {
    InitLoguru();
    const char* skip_env = std::getenv("SKIP_TEST_IF_MODEL_NOT_FOUND");
    if (skip_env != nullptr && std::string(skip_env) == "1" && !std::filesystem::exists(model_path_)) {
      GTEST_SKIP() << "Model path " << model_path_ << " not found, skipping test.";
    }

    std::filesystem::path parent = std::filesystem::path(__FILE__).parent_path();
    const std::string config_path =
        std::filesystem::absolute(parent / "../../../examples/llama7b/ksana_llm.yaml").string();

    AttentionBackendManager::GetInstance()->Initialize();
    const auto& env = Singleton<Environment>::GetInstance();
    context = std::make_shared<Context>();
    context->global.reset(new GlobalContext(1, 1, 1, env));
    context->global->InitDistributedConfig();

    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context->device = std::make_shared<DeviceContext>(1);
    context->device->ext->InitNcclParam(nccl_uid);

    env->ParseConfig(config_path, model_path_);
    env->UpdateModelConfig();
    env->GetModelConfig(model_config);

    BatchSchedulerConfig batch_scheduler_config;
    env->GetBatchSchedulerConfig(batch_scheduler_config);
    batch_scheduler_config.mtp_step_num = model_config.num_nextn_predict_layers;
    // UpdateModelConfig wipes dspark_config when yaml's mtp_step_num == 0. Restore via subclass hook
    // and align mtp_step_num to dspark_block_size before re-running UpdateModelConfig.
    ConfigureDspark(&model_config.dspark_config);
    if (model_config.dspark_config.dspark_block_size > 0) {
      batch_scheduler_config.mtp_step_num = model_config.dspark_config.dspark_block_size;
      env->SetModelConfig(model_config);
    }
    env->SetBatchSchedulerConfig(batch_scheduler_config);
    env->UpdateModelConfig();
    env->GetModelConfig(model_config);

    AttnBackendConfig attn_backend_config;
    env->GetAttnBackendConfig(attn_backend_config);
    attn_backend_config.enable_blocked_multi_token_forwarding_kv = true;
    env->SetAttnBackendConfig(attn_backend_config);

    InitPipelineConfigForTest(env, model_config);

    block_cache_layout_ = GetTestModelCacheLayout(env);
    env->InitializeBlockManagerConfig(block_cache_layout_);

    BlockManagerConfig block_manager_config;
    env->GetBlockManagerConfig(block_manager_config);
    block_manager_config.device_allocator_config.blocks_num = 32;
    block_manager_config.host_allocator_config.blocks_num = 32;
    if (block_manager_config.swa_device_allocator_config.block_size > 0) {
      block_manager_config.swa_device_allocator_config.blocks_num = 8;
    }
    env->SetBlockManagerConfig(block_manager_config);

    env->GetRuntimeConfig(runtime_config);
    runtime_config.mtp_step_num = batch_scheduler_config.mtp_step_num;
    env->GetPipelineConfig(pipeline_config);

    BlockAllocatorGroupConfig group_0;
    group_0.devices = {0};
    group_0.device_block_num = block_manager_config.device_allocator_config.blocks_num;
    group_0.host_block_num = block_manager_config.host_allocator_config.blocks_num;
    group_0.block_size = block_manager_config.device_allocator_config.block_size;
    BlockAllocatorManagerConfig alloc_cfg;
    alloc_cfg[0] = group_0;

    auto memory_allocator = std::make_shared<MemoryAllocator>();
    engine_block_manager = std::make_shared<BlockAllocatorManager>(true, alloc_cfg, memory_allocator, context);
    engine_block_group = engine_block_manager->GetBlockAllocatorGroup(0);
    executor_block_manager = std::make_shared<BlockAllocatorManager>(false, alloc_cfg, memory_allocator, context);
    executor_block_group = executor_block_manager->GetBlockAllocatorGroup(0);

    if (block_manager_config.swa_device_allocator_config.block_size > 0) {
      BlockAllocatorGroupConfig swa_group;
      swa_group.devices = {0};
      swa_group.device_block_num = block_manager_config.swa_device_allocator_config.blocks_num;
      swa_group.host_block_num = 0;
      swa_group.block_size = block_manager_config.swa_device_allocator_config.block_size;
      BlockAllocatorManagerConfig swa_alloc_cfg;
      swa_alloc_cfg[0] = swa_group;
      swa_block_manager_ = std::make_shared<BlockAllocatorManager>(false, swa_alloc_cfg, memory_allocator, context);
      swa_block_group_ = swa_block_manager_->GetBlockAllocatorGroup(0);
    }

    hidden_unit_buffer_helper = std::make_shared<HiddenUnitBufferHelper>(env);
    hidden_unit = hidden_unit_buffer_helper->CreateDeviceBuffer();

    inflight_resource_mgr_ = std::make_shared<InflightResourceManager>();
  }

  // Subclasses populate to enable DSpark; default leaves it disabled.
  virtual void ConfigureDspark(DsparkConfig* /*dspark*/) {}

  // Skip / fail-fast when the model checkpoint isn't on disk. Caller must return after invoking.
  void CheckModelPathOrSkip() {
    if (std::filesystem::exists(model_config.path)) return;
    const char* skip_env = std::getenv("SKIP_TEST_IF_MODEL_NOT_FOUND");
    if (skip_env != nullptr && std::string(skip_env) == "1") {
      GTEST_SKIP() << "Model path " << model_config.path << " not found, skipping test.";
    }
    EXPECT_TRUE(std::filesystem::exists(model_config.path));
  }

  std::shared_ptr<DeepSeekV4Model> CreateModel(std::shared_ptr<BaseWeight>* out_weight,
                                               std::unique_ptr<WeightInstance>* out_weight_inst) {
    SetDevice(kDeviceId);
    auto weight_inst = std::make_unique<WeightInstance>(model_config, runtime_config, context);
    weight_inst->Load();
    std::shared_ptr<BaseWeight> weight = weight_inst->GetWeight();
    auto model = std::make_shared<DeepSeekV4Model>(model_config, runtime_config, kDeviceId, context, weight,
                                                   block_cache_layout_);
    model->RegisterInflightResources(*inflight_resource_mgr_);
    model->SetInflightResourceManager(inflight_resource_mgr_.get());
    auto* slot_resource = inflight_resource_mgr_->GetTypedResource<CompressorSlotResource>("compressor_slot");
    EXPECT_NE(slot_resource, nullptr);
    *out_weight = weight;
    *out_weight_inst = std::move(weight_inst);
    return model;
  }

  // {context, decode} req pair sharing input_ids and kv-cache blocks; allocs compressor slots.
  std::vector<ForwardRequest> BuildContextDecodeReqs(const std::vector<int>& input_ids,
                                                     std::vector<FlexibleCachedCopyTask>* flex_tasks,
                                                     EmbeddingSlice* embedding_slice) {
    auto* slot_resource = inflight_resource_mgr_->GetTypedResource<CompressorSlotResource>("compressor_slot");

    auto ctx_req = std::make_unique<ForwardRequest>();
    ctx_req->req_id = 1;
    ctx_req->attn_dp_group_id = 0;
    ctx_req->block_allocator_manager = executor_block_manager;
    ctx_req->forwarding_tokens = std::make_shared<std::vector<int>>(input_ids);
    ctx_req->flexible_cached_copy_tasks = flex_tasks;
    ctx_req->input_refit_embedding = embedding_slice;

    std::vector<int> block_ids;
    const int block_token_num = runtime_config.attn_backend_config.block_token_num;
    const int use_block_num = (input_ids.size() + block_token_num - 1) / block_token_num;
    engine_block_group->GetDeviceBlockAllocator(0)->AllocateBlocks(use_block_num, block_ids);
    executor_block_group->GetDeviceBlockAllocator(0)->GetBlockPtrs(block_ids, ctx_req->kv_cache_ptrs);
    ctx_req->atb_kv_cache_base_blk_ids.assign(1, {});
    BuildFlatKVCacheBlockIds(GetTotalLayerNum(model_config, pipeline_config), block_ids,
                             ctx_req->atb_kv_cache_base_blk_ids, executor_block_group->GetDeviceBlockAllocator(0));
    for (int i = 0; i < use_block_num; i++) {
      Memset(ctx_req->kv_cache_ptrs[i], 0, runtime_config.attn_backend_config.block_size);
    }

    if (swa_block_group_) {
      std::vector<void*> swa_base_ptrs(1);
      swa_block_group_->GetDeviceBlockAllocator(0)->GetBlockPtrs({0}, swa_base_ptrs);
      ctx_req->swa_block_base_addr = swa_base_ptrs[0];
      ctx_req->swa_pool_block_num =
          static_cast<int64_t>(swa_block_group_->GetBlockAllocatorGroupConfig().device_block_num);
    }

    auto decode_req = std::make_unique<ForwardRequest>(*ctx_req);
    decode_req->req_id = 2;
    decode_req->infer_stage = InferStage::kDecode;
    decode_req->forwarding_tokens = std::make_shared<std::vector<int>>(*ctx_req->forwarding_tokens);
    decode_req->kv_cached_token_num = decode_req->forwarding_tokens->size() - 1;

    slot_resource->Alloc(ctx_req->req_id);
    slot_resource->Alloc(decode_req->req_id);

    return {*ctx_req, *decode_req};
  }

  // Greedy sampling config.
  static void FillGreedySamplingRequest(SamplingRequest& s, std::vector<int>* out_tokens, NgramDict* ngram_dict,
                                        SamplingConfig* cfg,
                                        std::shared_ptr<std::unordered_map<std::string, TargetDescribe>> target,
                                        std::shared_ptr<std::vector<std::vector<std::pair<int, float>>>> logprobs,
                                        std::shared_ptr<std::vector<int>> input_tokens) {
    cfg->num_beams = 1;
    cfg->topk = 1;
    cfg->topp = 0;
    cfg->temperature = 0;
    cfg->repetition_penalty = 1;

    s.input_tokens = std::move(input_tokens);
    s.sampling_token_num = 1;
    s.sampling_result_tokens = out_tokens;
    s.logprobs = logprobs;
    s.ngram_dict = ngram_dict;
    s.sampling_config = cfg;
    s.request_target = target;
  }

  // kMain forward + sampling. Returns the sampler for reuse in later phases.
  std::shared_ptr<Sampler> RunMainForwardAndSample(std::shared_ptr<DeepSeekV4Model> model,
                                                   std::shared_ptr<BaseWeight> weight,
                                                   std::vector<ForwardRequest>& forward_reqs,
                                                   const std::vector<int>& input_ids,
                                                   std::vector<std::vector<int>>* out_tokens_per_req) {
    EXPECT_TRUE(model->Forward(weight, forward_reqs, hidden_unit, RunMode::kMain).OK());
    StreamSynchronize(context->device->GetPrimaryComputeStream());

    out_tokens_per_req->assign(forward_reqs.size(), {});
    std::vector<SamplingRequest> sample_reqs(forward_reqs.size());
    NgramDict ngram_dict;
    SamplingConfig sample_config;
    auto request_target = std::make_shared<std::unordered_map<std::string, TargetDescribe>>();
    auto logprobs = std::make_shared<std::vector<std::vector<std::pair<int, float>>>>();
    auto input_tokens_shared = std::make_shared<std::vector<int>>(input_ids);
    for (size_t r = 0; r < forward_reqs.size(); ++r) {
      FillGreedySamplingRequest(sample_reqs[r], &(*out_tokens_per_req)[r], &ngram_dict, &sample_config, request_target,
                                logprobs, input_tokens_shared);
      sample_reqs[r].logits_offset = forward_reqs[r].logits_offset;
      sample_reqs[r].logits_buf = model->GetLogitsPtr();
    }

    BatchSchedulerConfig batch_scheduler_config;
    Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);
    auto sampler = std::make_shared<Sampler>(batch_scheduler_config, kDeviceId, context);
    sampler->Sampling(sample_reqs, context->device->GetPrimaryComputeStream());
    StreamSynchronize(context->device->GetPrimaryComputeStream());
    for (size_t r = 0; r < forward_reqs.size(); ++r) {
      EXPECT_EQ((*out_tokens_per_req)[r].size(), 1u);
    }
    return sampler;
  }

  // Pad forwarding_tokens with dspark_block_size noise tokens, run kDSpark, per-step ForwardHead + Sampling.
  void RunDSparkPhase(std::shared_ptr<DeepSeekV4Model> model, std::shared_ptr<BaseWeight> weight,
                      std::vector<ForwardRequest>& forward_reqs, std::shared_ptr<Sampler> sampler,
                      std::vector<std::vector<int>>* out_draft_tokens_per_req) {
    const size_t block_size = model_config.dspark_config.dspark_block_size;
    ASSERT_GT(block_size, 0u) << "RunDSparkPhase requires a DSpark-enabled model";
    const int noise_id = static_cast<int>(model_config.dspark_config.dspark_noise_token_id);
    const int block_token_num = runtime_config.attn_backend_config.block_token_num;

    // Prefix stays in kv-cache; append noise tokens for the scored block.
    for (auto& req : forward_reqs) {
      req.kv_cached_token_num = req.forwarding_tokens->size();
      req.infer_stage = InferStage::kContext;
      req.forwarding_tokens->insert(req.forwarding_tokens->end(), block_size, noise_id);
      req.sampling_token_num = block_size;

      const int need_blocks = (static_cast<int>(req.forwarding_tokens->size()) + block_token_num - 1) / block_token_num;
      if (need_blocks > static_cast<int>(req.kv_cache_ptrs.size())) {
        std::vector<int> extra_ids;
        engine_block_group->GetDeviceBlockAllocator(0)->AllocateBlocks(need_blocks - req.kv_cache_ptrs.size(),
                                                                       extra_ids);
        std::vector<void*> extra_ptrs;
        executor_block_group->GetDeviceBlockAllocator(0)->GetBlockPtrs(extra_ids, extra_ptrs);
        req.kv_cache_ptrs.insert(req.kv_cache_ptrs.end(), extra_ptrs.begin(), extra_ptrs.end());
      }
    }

    EXPECT_TRUE(model->Forward(weight, forward_reqs, hidden_unit, RunMode::kDSpark).OK());
    StreamSynchronize(context->device->GetPrimaryComputeStream());

    out_draft_tokens_per_req->assign(forward_reqs.size(), {});
    NgramDict ngram_dict;
    SamplingConfig sample_config;
    auto request_target = std::make_shared<std::unordered_map<std::string, TargetDescribe>>();
    auto logprobs = std::make_shared<std::vector<std::vector<std::pair<int, float>>>>();
    auto input_tokens_shared = std::make_shared<std::vector<int>>();

    for (size_t step_index = 0; step_index < block_size; ++step_index) {
      EXPECT_TRUE(model->ForwardHead(forward_reqs, step_index).OK());

      std::vector<SamplingRequest> step_reqs(forward_reqs.size());
      std::vector<std::vector<int>> step_out(forward_reqs.size());
      for (size_t r = 0; r < forward_reqs.size(); ++r) {
        FillGreedySamplingRequest(step_reqs[r], &step_out[r], &ngram_dict, &sample_config, request_target, logprobs,
                                  input_tokens_shared);
        step_reqs[r].logits_offset = r;
        step_reqs[r].logits_buf = model->GetLogitsPtr();
      }
      sampler->Sampling(step_reqs, context->device->GetPrimaryComputeStream());
      StreamSynchronize(context->device->GetPrimaryComputeStream());

      for (size_t r = 0; r < forward_reqs.size(); ++r) {
        ASSERT_EQ(step_out[r].size(), 1u);
        (*out_draft_tokens_per_req)[r].emplace_back(step_out[r].back());
        // Feed sampled token back into the block for the next step (matches ExecutorRuntime).
        if (step_index < block_size - 1) {
          auto& tokens = *forward_reqs[r].forwarding_tokens;
          tokens[tokens.size() - block_size + step_index + 1] = step_out[r].back();
        }
      }
    }
  }

  void Finalize(std::shared_ptr<DeepSeekV4Model>& model, std::shared_ptr<BaseWeight>& weight) {
    model.reset();
    weight.reset();
    StreamSynchronize(context->device->GetMemoryManageStream());
    DeviceSynchronize();
  }
};

}  // namespace ksana_llm
