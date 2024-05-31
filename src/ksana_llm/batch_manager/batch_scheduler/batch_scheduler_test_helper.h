/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/block_manager/block_manager_interface.h"
#include "ksana_llm/profiler/collector.h"
#include "ksana_llm/runtime/infer_request.h"
#include "test.h"

namespace ksana_llm {

// The memory pool management.
// This simulator sumulates block actions
// Note: Functions for contiguous memory are not supported.
class BlockManagerSimulator : public BlockManagerInterface {
 public:
  BlockManagerSimulator(const BlockManagerConfig& block_manager_config, int tp_num) {
    block_token_num_ = block_manager_config.device_allocator_config.block_token_num;
    host_block_num_ = block_manager_config.host_allocator_config.blocks_num;
    device_block_num_ = block_manager_config.device_allocator_config.blocks_num;

    device_num_ = tp_num;
    cur_device_id_ = 0;
    NLLM_LOG_INFO << "device_num=" << device_num_ << ", block_token_num=" << block_token_num_
                  << ", host_block_num=" << host_block_num_ << ", device_block_num=" << device_block_num_;

    // Init free block ids and kv cache contents
    int blk_id = 0;
    for (int i = 0; i < host_block_num_; i++) {
      std::vector<int> temp_kv;
      for (int j = 0; j < block_token_num_; j++) {
        temp_kv.push_back(default_content_);
      }
      host_kv_cache_contents_[blk_id] = temp_kv;
      host_free_block_.insert(blk_id);
      blk_id++;
    }

    for (int d_i = 0; d_i < device_num_; d_i++) {
      std::map<int, std::vector<int>> temp_block_kv;
      device_kv_cache_contents_.push_back(temp_block_kv);
      std::unordered_set<int> temp_blocks;
      device_free_block_.push_back(temp_blocks);
      device_alloc_block_.push_back(temp_blocks);
      for (int b_i = 0; b_i < device_block_num_; b_i++) {
        std::vector<int> temp_kv;
        for (int j = 0; j < block_token_num_; j++) {
          temp_kv.push_back(default_content_);
        }
        device_kv_cache_contents_[d_i][blk_id] = temp_kv;
        device_free_block_[d_i].insert(blk_id);
        blk_id++;
      }
    }
    NLLM_LOG_INFO << "BlockManagerSimulator started";
  }

  ~BlockManagerSimulator() {}

  // Preallocate blocks.
  Status PreAllocateBlocks() { return Status(); }

  // Reset the preallocated blocks for device & host.
  Status ResetPreAllocatedBlocks() { return Status(); }

  // This function maybe called concurrently from different threads.
  // DO NOT store the device id in variable.
  void SetDeviceId(int device_id) { cur_device_id_ = device_id; }

  // This function maybe called concurrently from different threads.
  int GetDeviceId() { return cur_device_id_; }

  // Allocate blocked memory on devicen
  Status AllocateBlocks(int64_t block_num, std::vector<int>& blocks) {
    return AllocBlocks(block_num, blocks, device_free_block_[cur_device_id_], device_alloc_block_[cur_device_id_]);
  }

  // Allocate contiguous memory on device.
  Status AllocateContiguous(int64_t size, int& block_id) {
    NLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Free blocked memory on device.
  Status FreeBlocks(const std::vector<int>& blocks) { return Status(); }

  // Free contiguous memory on device.
  Status FreeContiguous(int block_id) {
    NLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Check contiguous memory is in used.
  bool IsContiguousUsed(const int block_id) {
    NLLM_CHECK_WITH_INFO(false, "Not implemented");
    return false;
  }

  // Get memory addresses of blocked memory on device.
  Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) {
    NLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Get memory address of contiguous memory on device.
  Status GetContiguousPtr(int block_id, void*& addr) {
    NLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Get number of free blocked memory on device.
  size_t GetDeviceFreeBlockNumber() { return device_free_block_[cur_device_id_].size(); }

  // Get number of used blocked memory on device.
  size_t GetDeviceUsedBlockNumber() { return device_alloc_block_[cur_device_id_].size(); }

  // Allocate blocked memory on host.
  Status AllocateHostBlocks(int64_t block_num, std::vector<int>& blocks) {
    return AllocBlocks(block_num, blocks, host_free_block_, host_alloc_block_);
  }

  // Allocate contiguous memory on host.
  Status AllocateHostContiguous(int64_t size, int& block_id) {
    NLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Free blocked memory on host.
  Status FreeHostBlocks(const std::vector<int>& blocks) { return Status(); }

  // Free contiguous memory on host.
  Status FreeHostContiguous(int block_id) {
    NLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Get memory addresses of blocked memory on host.
  Status GetHostBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) {
    NLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Get memory address of contiguous memory on host.
  Status GetHostContiguousPtr(int block_id, void*& addr) {
    NLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Get number of free blocked memory on host.
  size_t GetHostFreeBlockNumber() { return host_free_block_.size(); }

  // Get number of used blocked memory on host.
  size_t GetHostUsedBlockNumber() { return host_alloc_block_.size(); }

  // Swap out blocks from device to host,
  // it could be swapped in later and keep block id not changed.
  Status SwapOut(const std::vector<int>& device_blocks, std::vector<int>& host_blocks,
                 const int host_block_num_to_add) {
    NLLM_LOG_INFO << "SwapOut device_blocks.size()=" << device_blocks.size()
                  << ", host_blocks.size()=" << host_blocks.size()
                  << ", host_block_num_to_add=" << host_block_num_to_add;
    AllocateHostBlocks(device_blocks.size() + host_block_num_to_add, host_blocks);
    int device_id = GetDeviceId();

    // Copy kv-cache contents
    CopyKvCacheContens(device_blocks, host_blocks, device_kv_cache_contents_[device_id], host_kv_cache_contents_);

    // Reset contents in block
    ResetKvCacheContents(device_blocks, device_kv_cache_contents_[device_id]);
    // TODO(robertyuan): Simulate communication delay

    FreeBlocks(device_blocks);
    return Status();
  }

  // Swap in blocks from host to device.
  Status SwapIn(const std::vector<int>& host_blocks, std::vector<int>& device_blocks) {
    NLLM_LOG_INFO << "SwapIn host_blocks.size()=" << host_blocks.size()
                  << ", device_blocks.size()=" << device_blocks.size();
    int device_id = GetDeviceId();
    AllocateBlocks(host_blocks.size(), device_blocks);

    // Copy kv-cache contents
    CopyKvCacheContens(host_blocks, device_blocks, host_kv_cache_contents_, device_kv_cache_contents_[device_id]);

    // Reset contents in block
    ResetKvCacheContents(host_blocks, host_kv_cache_contents_);
    // TODO(robertyuan): Simulate communication delay

    FreeHostBlocks(host_blocks);
    return Status();
  }

  // Drop the swapped blocks on host, and the block ids could be resued.
  Status SwapDrop(const std::vector<int>& host_blocks) {
    FreeHostBlocks(host_blocks);
    return Status();
  }

  // Get the size in bytes for one block.
  size_t GetBlockSize() const {
    NLLM_CHECK_WITH_INFO(false, "Not implemented");
    return 0;
  }

  // Get the token number for one block.
  size_t GetBlockTokenNum() const { return block_token_num_; }

  // Prepare blocks for prefix cache
  Status PreparePrefixCacheBlocks() {
    NLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Get the prefix cache tokens numbers
  int GetPrefixCacheTokensNumber() const {
    NLLM_CHECK_WITH_INFO(false, "Not implemented");
    return 0;
  }

  // Get the prefix cache blocks numbers
  size_t GetPrefixCacheBlocksNumber() const {
    NLLM_CHECK_WITH_INFO(false, "Not implemented");
    return 0;
  }

  // Check the input token is valid for prefix cache
  bool CheckReqIsValidForPrefixCache(const std::vector<int>& input_tokens) {
    NLLM_CHECK_WITH_INFO(false, "Not implemented");
    return false;
  }

  // Fill prefix kv cache to input blocks vector
  Status FillPrefixCacheBlocks(std::vector<std::vector<int>>& kv_cache_blocks) {
    NLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Get block manager config
  const BlockManagerConfig& GetBlockManagerConfig() const { return block_manager_config_; }

 public:
  void CollectKvCacheContent(std::shared_ptr<InferRequest>& req, std::vector<int>& kv_cache_contents) {
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
          NLLM_CHECK_WITH_INFO(
              kv_cache_contents[i] == temp_contents[i] - d_i,
              FormatStr("Kv cache content diff between device 0 and device %d, token_idx=%d.", d_i, i));
        }
      }
    }
  }
  void RecordGeneratedToken(std::shared_ptr<InferRequest>& req, int offset, int output_token) {
    // Compute block offset
    // If block num is not enough for recording, there must be some bug in scheduler
    int block_offset = offset / block_token_num_;
    int offset_in_block = offset % block_token_num_;
    for (int d_i = 0; d_i < device_num_; d_i++) {
      std::vector<int>& kv_cache_blocks = req->kv_cache_blocks[d_i];
      NLLM_CHECK_WITH_INFO(
          kv_cache_blocks.size() > (size_t)block_offset,
          FormatStr("Block not exist. Req id %d, block offset=%d, device_idx=%d, kv_cache_blocks.size()=%d.",
                    req->req_id, block_offset, d_i, kv_cache_blocks.size()));
      int block_idx = kv_cache_blocks[block_offset];
      NLLM_CHECK_WITH_INFO(device_kv_cache_contents_[d_i].find(block_idx) != device_kv_cache_contents_[d_i].end(),
                           FormatStr("Block kv cache content not exist on device %d. Req id %d, block idx=%d.", d_i,
                                     req->req_id, block_idx));
      device_kv_cache_contents_[d_i][block_idx][offset_in_block] = output_token + d_i;
    }
  }

 private:
  void CollectKvCacheContentFromDevice(int device_idx, std::vector<int>& block_list, int token_num,
                                       std::vector<int>& kv_cache_contents) {
    kv_cache_contents.resize(token_num);
    for (int i = 0; i < token_num; i++) {
      int block_offset = i / block_token_num_;
      int offset_in_block = i % block_token_num_;
      NLLM_CHECK_WITH_INFO((size_t)block_offset < block_list.size(),
                           FormatStr("block list on device %d is broken. size=%d, visiting block idx=%d.", device_idx,
                                     block_list.size(), block_offset));
      int block_idx = block_list[block_offset];
      kv_cache_contents[i] = device_kv_cache_contents_[device_idx][block_idx][offset_in_block];
    }
  }
  Status AllocBlocks(size_t block_num, std::vector<int>& blocks, std::unordered_set<int>& free_blocks,
                     std::unordered_set<int>& used_blocks) {
    NLLM_LOG_INFO << "Allocing, block_num=" << block_num << ", free_blocks.size()=" << free_blocks.size()
                  << ", used_blocks.size()=" << used_blocks.size();
    if (block_num > free_blocks.size()) {
      NLLM_LOG_INFO << "Failed to alloc, block_num=" << block_num << ", free_blocks.size()=" << free_blocks.size();
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
    NLLM_LOG_INFO << "Alloc OK, blocks.size()=" << blocks.size() << ", free_blocks.size()=" << free_blocks.size()
                  << ", used_blocks.size()=" << used_blocks.size();
    return Status();
  }
  Status FreeBlocks(std::vector<int>& blocks, std::unordered_set<int>& free_blocks,
                    std::unordered_set<int>& used_blocks) {
    for (auto block_id : blocks) {
      auto it = used_blocks.find(block_id);
      if (it != used_blocks.end()) {
        free_blocks.insert(*it);
        used_blocks.erase(it);
      } else {
        return Status(RET_FREE_FAIL, fmt::format("Double free error, block id {}", block_id));
      }
    }
    return Status();
  }

  void CopyKvCacheContens(const std::vector<int>& src_blks, const std::vector<int>& dst_blks,
                          std::map<int, std::vector<int>>& src_kv_contents,
                          std::map<int, std::vector<int>>& dst_kv_contents) {
    NLLM_CHECK_WITH_INFO(src_blks.size() <= dst_blks.size(),
                         FormatStr("src_blks.size > dst_blks.size(), %d, %d", src_blks.size(), dst_blks.size()));
    for (size_t i = 0; i < src_blks.size(); i++) {
      int src_blk = src_blks[i];
      int dst_blk = dst_blks[i];
      auto src_content_it = src_kv_contents.find(src_blk);
      auto dst_content_it = dst_kv_contents.find(dst_blk);
      NLLM_CHECK_WITH_INFO(src_content_it != src_kv_contents.end(),
                           FormatStr("Kv cache content of src block %d does not exist", src_blk));
      NLLM_CHECK_WITH_INFO(dst_content_it != dst_kv_contents.end(),
                           FormatStr("Kv cache content of dst block %d does not exist", dst_blk));
      for (int j = 0; j < block_token_num_; j++) {
        dst_content_it->second[j] = src_content_it->second[j];
      }
    }
  }

  void ResetKvCacheContents(const std::vector<int>& blks, std::map<int, std::vector<int>>& kv_contents) {
    for (auto blk : blks) {
      auto content_it = kv_contents.find(blk);
      NLLM_CHECK_WITH_INFO(content_it != kv_contents.end(),
                           FormatStr("Kv cache content of block %d does not exist", blk));
      for (int i = 0; i < block_token_num_; i++) {
        content_it->second[i] = default_content_;
      }
    }
  }

 private:
  BlockManagerConfig block_manager_config_;

  int block_token_num_;
  int host_block_num_;
  int device_block_num_;
  int device_num_;
  int tp_num_;

  int cur_device_id_;

  std::unordered_set<int> host_free_block_;
  std::vector<std::unordered_set<int>> device_free_block_;

  std::unordered_set<int> host_alloc_block_;
  std::vector<std::unordered_set<int>> device_alloc_block_;

  // kv_cache_contents is used to simulate kv-cache contents generated by LLM,
  // next token will be generated by these contents,
  // if copy operations have any bugs, generated sequence will give hints
  // kv_cache_contents[device_id][block_idx][token_offset]
  std::vector<std::map<int, std::vector<int>>> device_kv_cache_contents_;

  std::map<int, std::vector<int>> host_kv_cache_contents_;
  int default_content_ = -678;
};

//
class BatchSchedulerEvironmentSimulator {
 public:
  BatchSchedulerEvironmentSimulator(const BlockManagerConfig& block_manager_config, int tp_num) : tp_num_(tp_num) {
    blk_mgr_ = new BlockManagerSimulator(block_manager_config, tp_num);
    SetBlockManager(blk_mgr_);
    ProfilerConfig profiler_config;
    profiler_config.report_threadpool_size = 1;
    profiler_config.stat_interval_second = 1;
    profile_collector_ = new ProfileCollector(profiler_config);
    SetProfileCollector(profile_collector_);
    profile_collector_->Start();
  }
  ~BatchSchedulerEvironmentSimulator() {
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
      if ((req->output_tokens.size() - req->input_tokens.size()) < (size_t)(req_output_num_map_[req->req_id] - 1)) {
        std::vector<int> kv_contents;
        // Generate next token based on recorded kv cache content
        // If memory operations break kv cache content, generation results will be wrong
        blk_mgr_->CollectKvCacheContent(req, kv_contents);
        GenerateAToken(kv_contents, output_token);
      }
      req->output_tokens.push_back(output_token);
    }
  }

  void GenerateTokens(std::vector<int>& input_tokens, int output_token_num, std::vector<int>& output_tokens,
                      bool with_eos) {
    output_tokens.clear();
    output_tokens.resize(input_tokens.size());
    std::copy(input_tokens.begin(), input_tokens.end(), output_tokens.begin());
    NLLM_LOG_INFO << "output_tokens.size=" << output_tokens.size() << ", output_token_num=" << output_token_num
                  << ", with_eos=" << with_eos;
    if (with_eos) {
      output_token_num--;
    }
    for (int i = 0; i < output_token_num; i++) {
      int output_token;
      GenerateAToken(output_tokens, output_token);
      output_tokens.push_back(output_token);
    }
    if (with_eos) {
      output_tokens.push_back(GetEndId());
    }
  }

  void StartHeartBeat() {}
  void SetRequestOutputTokenNum(std::shared_ptr<InferRequest>& req, int output_token_num) {
    NLLM_CHECK_WITH_INFO(req_output_num_map_.find(req->req_id) == req_output_num_map_.end(),
                         FormatStr("Req id %d is already set.", req->req_id));
    req_output_num_map_[req->req_id] = output_token_num;
  }

  bool IsRequestFinished(std::shared_ptr<InferRequest>& req) { return req->output_tokens.back() == GetEndId(); }

  int GetEndId() { return -1; };

  std::vector<std::shared_ptr<InferRequest>> InitRequest(int req_id, int seed, int input_token_num,
                                                         int expected_output_token_num, std::shared_ptr<Request>& req,
                                                         int num_ret_seq = 1) {
    NLLM_LOG_INFO << "Init req " << req_id << ", seed=" << seed << ", input_token_num=" << input_token_num;
    ksana_llm::KsanaPythonInput ksana_python_input;
    ksana_python_input.sampling_config.num_beams = 0;
    ksana_python_input.sampling_config.num_return_sequences = num_ret_seq;
    req = std::make_shared<Request>(ksana_python_input);
    req->req_id = req_id;
    req->model_name = "llama";
    std::vector<int> seed_tokens;
    seed_tokens.push_back(seed);
    GenerateTokens(seed_tokens, input_token_num - 1, req->input_tokens, false);
    req->output_tokens = req->input_tokens;
    std::vector<std::shared_ptr<InferRequest>> infer_req_list;
    for (size_t i = 0; i < req->output_group.size(); i++) {
      std::shared_ptr<InferRequest> infer_req = std::make_shared<InferRequest>(req, i);
      infer_req->end_id = GetEndId();
      infer_req->kv_cache_blocks.resize(tp_num_);
      SetRequestOutputTokenNum(infer_req, expected_output_token_num);
      infer_req_list.push_back(infer_req);
    }
    return infer_req_list;
  }

  void CheckRequestOutput(std::shared_ptr<InferRequest>& req) {
    // Check request results
    int expected_output_token_num = req_output_num_map_[req->req_id];
    NLLM_LOG_INFO << "Checking req_id=" << req->req_id << ", input_tokens.size=" << req->input_tokens.size()
                  << ", output_tokens.size=" << req->output_tokens.size()
                  << ", expected_output_token_num=" << expected_output_token_num;
    std::vector<int> expect_output_tokens;
    bool with_eos = true;
    GenerateTokens(req->input_tokens, expected_output_token_num, expect_output_tokens, with_eos);
    EXPECT_EQ(expect_output_tokens.size(), req->output_tokens.size());
    EXPECT_EQ(expect_output_tokens.size(), req->input_tokens.size() + expected_output_token_num);
    for (size_t i = 0; i < req->output_tokens.size(); i++) {
      EXPECT_EQ(expect_output_tokens[i], req->output_tokens[i]);
    }
  }

 private:
  void GenerateAToken(std::vector<int>& input_tokens, int& output_token) {
    int sum_of_elems = std::accumulate(input_tokens.begin(), input_tokens.end(), 0);
    output_token = (sum_of_elems * 0x3e1f9d7) % 0x19de1f3;
  }

 private:
  BlockManagerSimulator* blk_mgr_;
  ProfileCollector* profile_collector_;
  std::unordered_map<int, int> req_output_num_map_;
  int tp_num_;
};

}  // namespace ksana_llm
