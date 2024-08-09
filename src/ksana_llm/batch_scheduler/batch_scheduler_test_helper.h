/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <sstream>

#include "ksana_llm/block_manager/block_manager_interface.h"
#include "ksana_llm/profiler/collector.h"
#include "ksana_llm/runtime/infer_request.h"
#include "test.h"

namespace ksana_llm {

thread_local int g_cur_device_id;
// The memory pool management.
// This simulator sumulates block actions
// Note: Functions for contiguous memory are not supported.
class BlockManagerSimulator : public BlockManagerInterface {
 public:
  BlockManagerSimulator(const BlockManagerConfig& block_manager_config, int tp_num)
      : block_manager_config_(block_manager_config) {
    block_token_num_ = block_manager_config.device_allocator_config.block_token_num;
    host_block_num_ = block_manager_config.host_allocator_config.blocks_num;
    device_block_num_ = block_manager_config.device_allocator_config.blocks_num;

    device_num_ = tp_num;
    g_cur_device_id = 0;
    KLLM_LOG_INFO << "device_num=" << device_num_ << ", block_token_num=" << block_token_num_
                  << ", host_block_num=" << host_block_num_ << ", device_block_num=" << device_block_num_;
    KLLM_CHECK_WITH_INFO((host_block_num_ <= DEVICE_ID_OFFSET) && (device_block_num_ < DEVICE_ID_OFFSET),
                         FormatStr("block_num should be less than DEVICE_ID_OFFSET=%d", DEVICE_ID_OFFSET));
    KLLM_CHECK_WITH_INFO((device_num_ <= HOST_DEVICE_ID),
                         FormatStr("device_num should be less than HOST_DEVICE_ID=%d", HOST_DEVICE_ID));
    // Init free block ids and kv cache contents
    for (int i = 0; i < host_block_num_; i++) {
      int d_blk_id = HOST_DEVICE_ID * DEVICE_ID_OFFSET + i;
      std::vector<int> temp_kv;
      for (int j = 0; j < block_token_num_; j++) {
        temp_kv.push_back(DEFAULT_KV_CONTENT);
      }
      host_kv_cache_contents_[d_blk_id] = temp_kv;
      host_free_block_.insert(d_blk_id);
    }

    for (int d_i = 0; d_i < device_num_; d_i++) {
      std::map<int, std::vector<int>> temp_block_kv;
      device_kv_cache_contents_.push_back(temp_block_kv);
      std::unordered_set<int> temp_blocks;
      device_free_block_.push_back(temp_blocks);
      device_alloc_block_.push_back(temp_blocks);
      for (int b_i = 0; b_i < device_block_num_; b_i++) {
        int d_blk_id = DEVICE_ID_OFFSET * (d_i + 1) + b_i;
        std::vector<int> temp_kv;
        for (int j = 0; j < block_token_num_; j++) {
          temp_kv.push_back(DEFAULT_KV_CONTENT);
        }
        device_kv_cache_contents_[d_i][d_blk_id] = temp_kv;
        device_free_block_[d_i].insert(d_blk_id);
      }
    }

    KLLM_LOG_DEBUG << "BlockManagerSimulator started";
  }

  ~BlockManagerSimulator() {}

  // Preallocate blocks.
  Status PreAllocateBlocks() { return Status(); }

  // Reset the preallocated blocks for device & host.
  Status ResetPreAllocatedBlocks() { return Status(); }

  // This function maybe called concurrently from different threads.
  // DO NOT store the device id in variable.
  void SetDeviceId(int device_id) {
    KLLM_LOG_DEBUG << "SetDeviceId from " << g_cur_device_id << " to " << device_id;
    g_cur_device_id = device_id;
  }

  // This function maybe called concurrently from different threads.
  int GetDeviceId() { return g_cur_device_id; }

  // The data type of the memory block allocated.
  DataType GetDtype() {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return TYPE_INVALID;
  }

  // Allocate blocked memory on devicen
  Status AllocateBlocks(int64_t block_num, std::vector<int>& blocks) {
    return AllocBlocks(g_cur_device_id, block_num, blocks, device_free_block_[g_cur_device_id],
                       device_alloc_block_[g_cur_device_id]);
  }

  // Allocate contiguous memory on device.
  Status AllocateContiguous(int64_t size, int& block_id) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Free blocked memory on device.
  Status FreeBlocks(const std::vector<int>& blocks) {
    return FreeBlocks(g_cur_device_id, blocks, device_free_block_[g_cur_device_id],
                      device_alloc_block_[g_cur_device_id]);
  }

  // Free contiguous memory on device.
  Status FreeContiguous(int block_id) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Check contiguous memory is in used.
  bool IsContiguousUsed(const int block_id) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return false;
  }

  // Get memory addresses of blocked memory on device.
  Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Get memory address of contiguous memory on device.
  Status GetContiguousPtr(int block_id, void*& addr) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Get number of free blocked memory on device.
  size_t GetDeviceFreeBlockNumber() { return device_free_block_[g_cur_device_id].size(); }

  // Get number of used blocked memory on device.
  size_t GetDeviceUsedBlockNumber() { return device_alloc_block_[g_cur_device_id].size(); }

  // Allocate blocked memory on host.
  Status AllocateHostBlocks(int64_t block_num, std::vector<int>& blocks) {
    std::lock_guard<std::recursive_mutex> guard(mux_);
    return AllocBlocks(HOST_DEVICE_ID, block_num, blocks, host_free_block_, host_alloc_block_);
  }

  // Allocate contiguous memory on host.
  Status AllocateHostContiguous(int64_t size, int& block_id) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Free blocked memory on host.
  Status FreeHostBlocks(const std::vector<int>& blocks) {
    return FreeBlocks(HOST_DEVICE_ID, blocks, host_free_block_, host_alloc_block_);
  }

  // Free contiguous memory on host.
  Status FreeHostContiguous(int block_id) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Get memory addresses of blocked memory on host.
  Status GetHostBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Get memory address of contiguous memory on host.
  Status GetHostContiguousPtr(int block_id, void*& addr) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Get number of free blocked memory on host.
  size_t GetHostFreeBlockNumber() { return host_free_block_.size(); }

  // Get number of used blocked memory on host.
  size_t GetHostUsedBlockNumber() { return host_alloc_block_.size(); }

  // The swap out/in for single block, the device block has been allocated on current device.
  // Do not free memory after swapness, the caller will do that.
  Status SwapOut(int host_block_id, int device_block_id) {
    int device_id = GetDeviceId();
    KLLM_LOG_DEBUG << "SwapOut on device " << device_id << " device_blockid=" << device_block_id
                   << ", host_block_id=" << host_block_id;

    std::vector<int> device_blocks{device_block_id};
    std::vector<int> host_blocks{host_block_id};

    // Copy kv-cache contents
    CopyKvCacheContents(device_blocks, host_blocks, device_kv_cache_contents_[device_id], host_kv_cache_contents_);

    // Reset contents in block
    ResetKvCacheContents(device_blocks, device_kv_cache_contents_[device_id]);
    // TODO(robertyuan): Simulate communication delay
    std::this_thread::sleep_for(std::chrono::microseconds(1));

    stat_.swapout_succ_num += device_blocks.size();
    return Status();
  }

  Status SwapIn(int device_block_id, int host_block_id) {
    int device_id = GetDeviceId();
    KLLM_LOG_DEBUG << "SwapIn on device " << device_id << ", host_block_id=" << host_block_id
                   << ", device_block_id=" << device_block_id;

    std::vector<int> device_blocks{device_block_id};
    std::vector<int> host_blocks{host_block_id};

    // Copy kv-cache contents
    CopyKvCacheContents(host_blocks, device_blocks, host_kv_cache_contents_, device_kv_cache_contents_[device_id]);

    // Reset contents in block
    ResetKvCacheContents(host_blocks, host_kv_cache_contents_);
    // TODO(robertyuan): Simulate communication delay
    std::this_thread::sleep_for(std::chrono::microseconds(1));

    KLLM_LOG_DEBUG << "SwapIn finished on device " << device_id << ". host_block_id=" << host_block_id
                   << ", device_block_id=" << device_block_id;
    stat_.swapin_succ_num += host_blocks.size();
    return Status();
  }

  // Drop the swapped blocks on host, and the block ids could be resued.
  Status SwapDrop(const std::vector<int>& host_blocks) {
    FreeHostBlocks(host_blocks);
    return Status();
  }

  // Get the size in bytes for one block.
  size_t GetBlockSize() const {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return 0;
  }

  // Get the token number for one block.
  size_t GetBlockTokenNum() const { return block_token_num_; }

  // Get block manager config
  const BlockManagerConfig& GetBlockManagerConfig() const { return block_manager_config_; }

 public:  // Functions not in BlockManagerInferface
  void CollectKvCacheContent(std::shared_ptr<InferRequest>& req, std::vector<int>& kv_cache_contents) {
    std::lock_guard<std::recursive_mutex> guard(mux_);
    int kv_cache_token_num = req->output_tokens.size();
    // Collect kv cache content from device 0
    CollectKvCacheContentFromDevice(0, req->kv_cache_blocks[0], kv_cache_token_num, kv_cache_contents);

    // Check all devices have some contents
    if (device_num_ > 1) {
      for (int d_i = 1; d_i < device_num_; d_i++) {
        std::vector<int> temp_contents;
        CollectKvCacheContentFromDevice(d_i, req->kv_cache_blocks[d_i], kv_cache_token_num, temp_contents);
        // Check results
        for (int i = 0; i < kv_cache_token_num; i++) {
          KLLM_CHECK_WITH_INFO(
              kv_cache_contents[i] == temp_contents[i] - d_i,
              FormatStr("Kv cache content diff between device 0 and device %d, token_idx=%d.", d_i, i));
        }
      }
    }
  }

  void RecordGeneratedToken(std::shared_ptr<InferRequest>& req, int offset, int output_token) {
    std::lock_guard<std::recursive_mutex> guard(mux_);
    // Compute block offset
    // If block num is not enough for recording, there must be some bug in scheduler
    int block_offset = offset / block_token_num_;
    int offset_in_block = offset % block_token_num_;
    for (int d_i = 0; d_i < device_num_; d_i++) {
      std::vector<int>& kv_cache_blocks = req->kv_cache_blocks[d_i];
      KLLM_CHECK_WITH_INFO(
          kv_cache_blocks.size() > (size_t)block_offset,
          FormatStr("Block not exist. Req id %d, block offset=%d, device_idx=%d, kv_cache_blocks.size()=%d.",
                    req->req_id, block_offset, d_i, kv_cache_blocks.size()));
      int block_idx = kv_cache_blocks[block_offset];
      KLLM_CHECK_WITH_INFO(device_kv_cache_contents_[d_i].find(block_idx) != device_kv_cache_contents_[d_i].end(),
                           FormatStr("Block kv cache content not exist on device %d. Req id %d, block idx=%d.", d_i,
                                     req->req_id, block_idx));
      device_kv_cache_contents_[d_i][block_idx][offset_in_block] = output_token + d_i;
    }
  }

  struct Statistics {
    int swapout_succ_num = 0;
    int swapout_fail_num = 0;
    int swapin_succ_num = 0;
    int swapin_fail_num = 0;
  };

  const Statistics& GetStatistics() { return stat_; }

  const BlockManagerConfig& GetConfig() { return block_manager_config_; }

 private:
  void CollectKvCacheContentFromDevice(int device_idx, std::vector<int>& block_list, int token_num,
                                       std::vector<int>& kv_cache_contents) {
    std::lock_guard<std::recursive_mutex> guard(mux_);
    kv_cache_contents.resize(token_num);
    for (int i = 0; i < token_num; i++) {
      int block_offset = i / block_token_num_;
      int offset_in_block = i % block_token_num_;
      KLLM_CHECK_WITH_INFO((size_t)block_offset < block_list.size(),
                           FormatStr("block list on device %d is broken. size=%d, visiting block idx=%d.", device_idx,
                                     block_list.size(), block_offset));
      int block_idx = block_list[block_offset];
      kv_cache_contents[i] = device_kv_cache_contents_[device_idx][block_idx][offset_in_block];
    }
  }

  Status AllocBlocks(int device_id, size_t block_num, std::vector<int>& blocks, std::unordered_set<int>& free_blocks,
                     std::unordered_set<int>& used_blocks) {
    std::lock_guard<std::recursive_mutex> guard(mux_);
    if (block_num > free_blocks.size()) {
      KLLM_LOG_DEBUG << "Failed to alloc on device " << device_id << ", block_num=" << block_num
                     << ", free_blocks.size()=" << free_blocks.size();
      return Status(RET_ALLOCATE_FAIL,
                    FormatStr("No more free blocks, expect %d, free %d", block_num, free_blocks.size()));
    }

    blocks.clear();
    blocks.reserve(block_num);
    auto it = free_blocks.begin();
    while (block_num--) {
      used_blocks.insert(*it);
      blocks.push_back(*it);
      it = free_blocks.erase(it);
    }
    std::stringstream ss_blocks;
    for (auto block_id : blocks) {
      ss_blocks << block_id << ", ";
    }
    KLLM_LOG_DEBUG << "Alloc OK on device " << device_id << ", alloc block num=" << blocks.size()
                   << ", free_blocks.size()=" << free_blocks.size() << ", used_blocks.size()=" << used_blocks.size()
                   << ", blocks=[" << ss_blocks.str() << "].";
    return Status();
  }

  Status FreeBlocks(int device_id, const std::vector<int>& blocks, std::unordered_set<int>& free_blocks,
                    std::unordered_set<int>& used_blocks) {
    std::lock_guard<std::recursive_mutex> guard(mux_);
    std::stringstream ss_blocks;
    for (auto block_id : blocks) {
      ss_blocks << block_id << ", ";
    }

    KLLM_LOG_DEBUG << "Free start on device " << device_id << ", block num=" << blocks.size()
                   << ", free_blocks.size()=" << free_blocks.size() << ", used_blocks.size()=" << used_blocks.size()
                   << ", blocks=[" << ss_blocks.str() << "].";

    for (auto block_id : blocks) {
      auto it = used_blocks.find(block_id);
      if (it != used_blocks.end()) {
        free_blocks.insert(*it);
        used_blocks.erase(it);
      } else {
        assert(false);
        KLLM_CHECK_WITH_INFO(false, "Double free");
        return Status(RET_FREE_FAIL, fmt::format("Double free error, block id {}", block_id));
      }
    }
    KLLM_LOG_DEBUG << "Free OK on device " << device_id;
    return Status();
  }

  void CopyKvCacheContents(const std::vector<int>& src_blks, const std::vector<int>& dst_blks,
                           std::map<int, std::vector<int>>& src_kv_contents,
                           std::map<int, std::vector<int>>& dst_kv_contents) {
    std::lock_guard<std::recursive_mutex> guard(mux_);
    KLLM_CHECK_WITH_INFO(src_blks.size() <= dst_blks.size(),
                         FormatStr("src_blks.size > dst_blks.size(), %d, %d", src_blks.size(), dst_blks.size()));
    for (size_t i = 0; i < src_blks.size(); i++) {
      int src_blk = src_blks[i];
      int dst_blk = dst_blks[i];
      auto src_content_it = src_kv_contents.find(src_blk);
      auto dst_content_it = dst_kv_contents.find(dst_blk);
      KLLM_CHECK_WITH_INFO(src_content_it != src_kv_contents.end(),
                           FormatStr("Kv cache content of src block %d does not exist", src_blk));
      KLLM_CHECK_WITH_INFO(dst_content_it != dst_kv_contents.end(),
                           FormatStr("Kv cache content of dst block %d does not exist", dst_blk));
      for (int j = 0; j < block_token_num_; j++) {
        dst_content_it->second[j] = src_content_it->second[j];
      }
    }
  }

  void ResetKvCacheContents(const std::vector<int>& blks, std::map<int, std::vector<int>>& kv_contents) {
    std::lock_guard<std::recursive_mutex> guard(mux_);
    for (auto blk : blks) {
      auto content_it = kv_contents.find(blk);
      KLLM_CHECK_WITH_INFO(content_it != kv_contents.end(),
                           FormatStr("Kv cache content of block %d does not exist", blk));
      for (int i = 0; i < block_token_num_; i++) {
        content_it->second[i] = DEFAULT_KV_CONTENT;
      }
    }
  }

 private:
  BlockManagerConfig block_manager_config_;
  std::recursive_mutex mux_;

  int block_token_num_;
  int host_block_num_;
  int device_block_num_;
  int device_num_;
  int tp_num_;

  std::unordered_set<int> host_free_block_;
  std::vector<std::unordered_set<int>> device_free_block_;

  std::unordered_set<int> host_alloc_block_;
  std::vector<std::unordered_set<int>> device_alloc_block_;

  // TODO(robertyuan): remove when dynamic prefix cache is implemented.
  std::vector<std::vector<int>> prefix_cache_blocks_;
  std::vector<int> prefix_cache_tokens_;

  // kv_cache_contents is used to simulate kv-cache contents generated by LLM,
  // next token will be generated by these contents,
  // if copy operations have any bugs, generated sequence will give hints
  // kv_cache_contents[device_id][block_idx][token_offset]
  std::vector<std::map<int, std::vector<int>>> device_kv_cache_contents_;

  std::map<int, std::vector<int>> host_kv_cache_contents_;

  Statistics stat_;

  int DEFAULT_KV_CONTENT = -678;
  int DEVICE_ID_OFFSET = 10000;
  int HOST_DEVICE_ID = 9;
};

//
class BatchSchedulerEnvironmentSimulator {
 public:
  BatchSchedulerEnvironmentSimulator(const BlockManagerConfig& block_manager_config, int tp_num) : tp_num_(tp_num) {
    blk_mgr_ = new BlockManagerSimulator(block_manager_config, tp_num);
    SetBlockManager(blk_mgr_);
    ProfilerConfig profiler_config;
    profiler_config.report_threadpool_size = 1;
    profiler_config.stat_interval_second = 1;
    profile_collector_ = new ProfileCollector(profiler_config);
    SetProfileCollector(profile_collector_);
    profile_collector_->Start();
  }
  ~BatchSchedulerEnvironmentSimulator() {
    delete blk_mgr_;
    profile_collector_->Stop();
    delete profile_collector_;
  }

  void RunAStep(std::vector<std::shared_ptr<InferRequest>>& scheduled_reqs) {
    for (auto req : scheduled_reqs) {
      // Generate kv cache content
      // This operation should be done before generation because kv cache is writen during generating next token
      if (req->output_tokens.size() == req->input_tokens.size()) {
        // This is the first time to generate, do context kv cache generation
        for (size_t i = 0; i < req->input_tokens.size(); i++) {
          int cur_input = req->input_tokens[i];
          blk_mgr_->RecordGeneratedToken(req, i, cur_input);
        }
      } else {
        // Generate kv cache for last output
        blk_mgr_->RecordGeneratedToken(req, req->output_tokens.size() - 1, req->output_tokens.back());
      }

      // Generate a token
      int output_token = GetEndId();
      KLLM_CHECK_WITH_INFO(req_output_num_map_.find(req->req_id) != req_output_num_map_.end(),
                           FormatStr("Req id %d is not exist in req_output_num_map.", req->req_id));
      if ((req->output_tokens.size() - req->input_tokens.size()) < (size_t)(req_output_num_map_[req->req_id] - 1)) {
        std::vector<int> kv_contents;
        // Generate next token based on recorded kv cache content
        // If memory operations break kv cache content, generation results will be wrong
        blk_mgr_->CollectKvCacheContent(req, kv_contents);
        GenerateAToken(kv_contents, output_token,
                       GetSeed(req->output_tokens.size(), req_generation_seeds_[req->req_id]));
      }
      req->output_tokens.push_back(output_token);
    }
    // Assumption: A step is slower than swapout
    std::this_thread::sleep_for(std::chrono::microseconds(2));
  }

  void GenerateTokens(std::vector<int>& input_tokens, int output_token_num, std::vector<int>& output_tokens,
                      bool with_eos, const std::vector<std::pair<int, int>>& seed_list) {
    output_tokens.clear();
    if (input_tokens.size() > 0) {
      output_tokens.resize(input_tokens.size());
      std::copy(input_tokens.begin(), input_tokens.end(), output_tokens.begin());
    }

    // Generate tokens
    if (with_eos) {
      output_token_num--;
    }
    for (int i = 0; i < output_token_num; i++) {
      int output_token;
      GenerateAToken(output_tokens, output_token, GetSeed(input_tokens.size() + i, seed_list));
      output_tokens.push_back(output_token);
    }
    if (with_eos) {
      output_tokens.push_back(GetEndId());
    }
  }

  bool IsRequestFinished(std::shared_ptr<InferRequest>& req) { return req->output_tokens.back() == GetEndId(); }

  int GetEndId() { return -1; }

  std::vector<std::shared_ptr<InferRequest>> InitRequest(int req_id, int input_token_num, int expected_output_token_num,
                                                         std::shared_ptr<Request>& req,
                                                         const std::vector<std::pair<int, int>>& seeds) {
    KLLM_LOG_DEBUG << "Init req " << req_id << ", input_token_num=" << input_token_num
                   << ", expect_output_token_num=" << expected_output_token_num;
    std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
    ksana_python_input->sampling_config.num_beams = 0;
    ksana_python_input->sampling_config.num_return_sequences = 1;
    req = std::make_shared<Request>(ksana_python_input);
    req->req_id = req_id;
    req->model_name = "llama";
    req->waiter = std::make_shared<Waiter>(1);
    std::vector<int> dummy_tokens;
    GenerateTokens(dummy_tokens, input_token_num, req->input_tokens, false, seeds);
    req->output_tokens = req->input_tokens;

    std::vector<std::shared_ptr<InferRequest>> infer_req_list;
    for (size_t i = 0; i < req->output_group.size(); i++) {
      std::shared_ptr<InferRequest> infer_req = std::make_shared<InferRequest>(req, i);
      infer_req->sampling_config.stop_token_ids.push_back(GetEndId());
      infer_req->kv_cache_blocks.resize(tp_num_);

      SetRequestOutputTokenNum(infer_req, expected_output_token_num);
      SetRequestGenerationSeeds(infer_req->req_id, seeds);

      infer_req_list.push_back(infer_req);
    }
    return infer_req_list;
  }

  void CheckRequestOutput(const std::shared_ptr<InferRequest>& req) {
    // Check request results
    int expected_output_token_num = req_output_num_map_[req->req_id];
    KLLM_LOG_DEBUG << "Checking req_id=" << req->req_id << ", input_tokens.size=" << req->input_tokens.size()
                   << ", output_tokens.size=" << req->output_tokens.size()
                   << ", expected_output_token_num=" << expected_output_token_num;
    std::vector<int> expect_output_tokens;
    bool with_eos = true;

    GenerateTokens(req->input_tokens, expected_output_token_num, expect_output_tokens, with_eos,
                   req_generation_seeds_[req->req_id]);

    EXPECT_EQ(expect_output_tokens.size(), req->output_tokens.size());
    EXPECT_EQ(expect_output_tokens.size(), req->input_tokens.size() + expected_output_token_num);
    for (size_t i = 0; i < req->output_tokens.size(); i++) {
      EXPECT_EQ(expect_output_tokens[i], req->output_tokens[i]);
    }
  }

  const BlockManagerSimulator::Statistics& GetBlockManagerStat() { return blk_mgr_->GetStatistics(); }
  const BlockManagerConfig& GetBlockManagerConfig() { return blk_mgr_->GetConfig(); }

 private:
  void SetRequestOutputTokenNum(std::shared_ptr<InferRequest>& req, int output_token_num) {
    KLLM_CHECK_WITH_INFO(req_output_num_map_.find(req->req_id) == req_output_num_map_.end(),
                         FormatStr("SetRequestOutputTokenNum: Req id %d is already set.", req->req_id));
    req_output_num_map_[req->req_id] = output_token_num;
  }

  void SetRequestGenerationSeeds(int req_id, const std::vector<std::pair<int, int>>& seeds) {
    KLLM_CHECK_WITH_INFO(req_generation_seeds_.find(req_id) == req_generation_seeds_.end(),
                         FormatStr("SetRequestGenerationSeed: Req id %d is already set.", req_id));
    KLLM_CHECK_WITH_INFO(seeds.size() > 0,
                         FormatStr("SetRequestGenerationSeed: Trying to set empty seed for req %d.", req_id));
    KLLM_CHECK_WITH_INFO(seeds[0].first == 0, "SetRequestGenerationSeed: First offset must be 0.");

    if (seeds.size() > 0) {
      int last_offset = 0;
      for (size_t i = 1; i < seeds.size(); i++) {
        KLLM_CHECK_WITH_INFO(seeds[i].first > last_offset, "SetRequestGenerationSeed: Wrong offset order.");
        last_offset = seeds[i].first;
      }
    }

    req_generation_seeds_[req_id] = seeds;
  }

  void GenerateAToken(std::vector<int>& input_tokens, int& output_token, int seed) {
    int sum_of_elems = std::accumulate(input_tokens.begin(), input_tokens.end(), 0);
    output_token = ((sum_of_elems * 0x3e1f9d7) % 0x19de1f3 + seed) % 200000;
  }

  int GetSeed(int offset, const std::vector<std::pair<int, int>>& seeds) {
    KLLM_CHECK_WITH_INFO(offset >= 0, "");
    int seed = -1;
    for (auto& it : seeds) {
      if (offset >= it.first) {
        seed = it.second;
      }
      if (offset < it.first) {
        break;
      }
    }
    return seed;
  }

 private:
  BlockManagerSimulator* blk_mgr_;
  ProfileCollector* profile_collector_;
  std::unordered_map<int, int> req_output_num_map_;

  // map for seeds used to generate input and output
  // <req_1, {<0, seed1>, <8, seed2>}> means req_1 will use seed1 to generate token from 0 to 8-1, seed2 for 8 to end.
  std::unordered_map<int, std::vector<std::pair<int, int>>> req_generation_seeds_;
  int tp_num_;
};

}  // namespace ksana_llm
