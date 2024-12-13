/* Copyright 2024 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/

#include "ksana_llm/samplers/sampler.h"
#include "test.h"

using namespace ksana_llm;

class SamplerTest : public testing::Test {
 protected:
  class DerivedSampler : public Sampler {
   public:
    DerivedSampler(const BatchSchedulerConfig &batch_scheduler_config, int rank, std::shared_ptr<Context> context)
        : Sampler(batch_scheduler_config, rank, context) {}

    std::vector<float> GetHostTemperatures() const { return host_temperatures_; }
  };

 protected:
  void SetUp() override {
    context_ = std::make_shared<Context>(1, 1);

    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    const auto &env = Singleton<Environment>::GetInstance();
    env->ParseConfig(config_path);
    env->GetModelConfig("", model_config_);
    BlockManagerConfig block_manager_config;
    env->GetBlockManagerConfig(block_manager_config);
    block_manager_ = new BlockManager(block_manager_config, context_);
    SetBlockManager(block_manager_);

    vocab_size_ = model_config_.vocab_size;
    // logits_buf.shape = [max_batch_size, vocab_size]
    // logits_buf.dtype = float32
    Malloc(&logits_buf_, max_batch_size_ * vocab_size_ * sizeof(float));

    BatchSchedulerConfig batch_scheduler_config;
    Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);
    sampler_ = std::make_shared<DerivedSampler>(batch_scheduler_config, device_id_, context_);

    // The default sampling mode is greedy.
    sampling_config_.num_beams = 1;
    sampling_config_.topk = 1;
    sampling_config_.topp = 0;
    sampling_config_.temperature = 0;
    sampling_config_.repetition_penalty = 1;
    sampling_config_.no_repeat_ngram_size = 0;
    sampling_config_.encoder_no_repeat_ngram_size = 0;
    sampling_config_.stop_token_ids = {};
    sampling_config_.max_new_tokens = 1024;
    sampling_config_.logprobs_num = 0;
  }

  void TearDown() override {
    Free(logits_buf_);
    sampler_.reset();
    delete block_manager_;
  }

  SamplingRequest GetSamlingRequest() {
    SamplingRequest sample_req;
    sample_req.input_tokens = &token_ids_;
    sample_req.output_tokens = &token_ids_;
    sample_req.logits_offset = 0;
    sample_req.logprobs = &logprobs_;
    sample_req.ngram_dict = &ngram_dict_;
    sample_req.output_mutex = &output_mutex_;
    sample_req.logits_buf = {reinterpret_cast<float *>(logits_buf_)};
    sample_req.model_config = &model_config_;
    sample_req.sampling_config = &sampling_config_;
    return sample_req;
  }

  void SetLogitsBuf(const std::vector<float> &logits_buf_cpu) {
    if (logits_buf_cpu.size() > vocab_size_ * max_batch_size_) {
      KLLM_LOG_ERROR << fmt::format("logits_buf_cpu is out of space in logits_buf: {} > {}", logits_buf_cpu.size(),
                                    vocab_size_ * max_batch_size_);
      return;
    }
    MemcpyAsync(logits_buf_, logits_buf_cpu.data(), logits_buf_cpu.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
                context_->GetH2DStreams()[device_id_]);
    StreamSynchronize(context_->GetH2DStreams()[device_id_]);
  }

 protected:
  // Parameters used for create the sampler_
  int device_id_ = 0;
  std::shared_ptr<Context> context_{nullptr};
  std::shared_ptr<DerivedSampler> sampler_{nullptr};
  BlockManager *block_manager_ = nullptr;
  int vocab_size_;
  int max_batch_size_ = 4;

  // Parameters used for default initialization of sample_req.
  void *logits_buf_ = nullptr;
  std::vector<int> token_ids_ = {1, 2, 3, 4, 5};
  NgramDict ngram_dict_;
  std::vector<std::vector<std::pair<int, float>>> logprobs_;
  std::mutex output_mutex_;
  ModelConfig model_config_;
  SamplingConfig sampling_config_;
};

TEST_F(SamplerTest, BaseSamplerTest) {
  SamplingRequest sample_req = GetSamlingRequest();

  // Assign a value of 1 to the logits for token_id=6 to make the sampler result as 6.
  std::vector<float> logits_buf_cpu(vocab_size_);
  logits_buf_cpu[6] = 1;
  SetLogitsBuf(logits_buf_cpu);
  std::vector<SamplingRequest> sample_reqs = {sample_req};

  sampler_->Sampling(sample_reqs, context_->GetComputeStreams()[device_id_]);
  EXPECT_EQ(6, (*sample_req.output_tokens).size());
  EXPECT_EQ(6, (*sample_req.output_tokens).back());
}

TEST_F(SamplerTest, TemperatureAutoVerifyTest) {
  // When the temperature is 0, it should be automatically corrected to 1 to avoid division by zero exception.
  SamplingRequest sample_req = GetSamlingRequest();
  std::vector<SamplingRequest> sample_reqs = {sample_req};

  float *device_logits = nullptr;
  SamplingDevideParameter sampling_devide_parameter;
  sampler_->PrepareDevideLogitsAndParameter(sample_reqs, sampling_devide_parameter, device_logits,
                                            context_->GetComputeStreams()[device_id_]);

  std::vector<float> temperature_cpu = sampler_->GetHostTemperatures();
  EXPECT_NEAR(1.0f, temperature_cpu[0], 1e-6);
}

TEST_F(SamplerTest, LogprobsSamplerTest) {
  SamplingRequest sample_req = GetSamlingRequest();

  std::vector<float> logits_buf_cpu = {0.0, 1.0, 2.0, 1.5, 0.7, 1.8};
  SetLogitsBuf(logits_buf_cpu);

  // set logprobs num to enable logprobs
  sampling_config_.logprobs_num = 2;
  std::vector<SamplingRequest> sample_reqs = {sample_req};

  sampler_->Sampling(sample_reqs, context_->GetComputeStreams()[device_id_]);
  EXPECT_EQ(1, logprobs_.size());

  // logprobs is not supported in ACL.
#ifdef ENABLE_CUDA
  EXPECT_EQ(2, logprobs_[0].size());
  EXPECT_EQ(2, logprobs_[0][0].first);
  EXPECT_NEAR(-0.598139f, logprobs_[0][0].second, 1e-6);
  EXPECT_EQ(5, logprobs_[0][1].first);
  EXPECT_NEAR(-0.798139f, logprobs_[0][1].second, 1e-6);
#endif

  sampling_config_.logprobs_num = 0;
}
