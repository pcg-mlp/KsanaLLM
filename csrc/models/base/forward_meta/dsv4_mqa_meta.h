/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "configure/context.h"
#include "configure/environment.h"
#include "configure/model_config.h"
#include "runtime/forward_request.h"
#include "utils/tensor.h"

namespace ksana_llm {

class Dsv4MqaMeta {
 public:
  Dsv4MqaMeta(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int layer_num_on_node, int rank,
              std::shared_ptr<Context> context, std::shared_ptr<BaseModelCacheLayout> block_cache_layout);

  void Prepare(const std::vector<ForwardRequest>& forward_reqs);

  struct input_info {
    std::vector<const ForwardRequest*> dp_reqs;

    // Tensors in input_info are views of the above shared tensors
    // Their shape is used to locate the offset of each input_info in the shared tensors
    Tensor raw_out_loc_tensor;
    Tensor swa_block_indices_tensor;
    Tensor c128_block_indices_tensor;
    Tensor c128_topk_lengths_tensor;

    // tile_scheduler_metadata for c4/c128 compressed paths (shape[2] as init flag)
    Tensor c4_tile_scheduler_metadata;
    Tensor c128_tile_scheduler_metadata;
    Tensor c4_num_splits;
    Tensor c128_num_splits;

    // Persistent host vectors to prevent destruction during MemcpyAsync
    std::vector<int> raw_out_loc_host;
    std::vector<int> swa_block_indices_host;
    std::vector<int> c128_block_indices_host;
    std::vector<int> c128_topk_lengths_host;

    size_t q_seq_len = 0;
    size_t total_dp_input_ids_len = 0;

    void Reset() {
      dp_reqs.clear();
      raw_out_loc_host.clear();
      swa_block_indices_host.clear();
      c128_block_indices_host.clear();
      c128_topk_lengths_host.clear();
      q_seq_len = 0;
      total_dp_input_ids_len = 0;
    }
  };

  // Category is tightly coupled to underlying kernel
  input_info flash_input;               // extend_len is non-specialized, use flash attention
  std::vector<input_info> page_inputs;  // extend_len is fixed (`[1, decode_token_num_threshold]`), use page attention

 private:
  // FlashMLA kernel supports q_seq_len >= 1 in decode phase
  ForwardRequestType GetForwardRequestType(const ForwardRequest& forward_request) const {
    if (forward_request.GetInputIdsLength() <= GetDecodeTokenNumThreshold() &&
        forward_request.kv_cached_token_num > 0) {
      return ForwardRequestType::kPage;
    } else {
      return ForwardRequestType::kFlash;
    }
  }

  void PrepareOutLoc(input_info& input);
  void PrepareBlockIndices(input_info& input);
  void PrepareTopkLengths(input_info& input);

  // Convert a logical token position to a physical cache position given block IDs and block size.
  inline int GetPhysicalTokenPos(const std::vector<int32_t>& block_ids, int token_pos, int compress_ratio = 1) const {
    return token_pos < 0 ? -1
                         : block_ids[token_pos / block_token_num_] * (block_token_num_ / compress_ratio) +
                               (token_pos / compress_ratio) % (block_token_num_ / compress_ratio);
  }

  // 将 token 位置映射到 SWA pool 内的物理位置 (SWA 与主 KV 独立编址)。
  //   global_block_idx = token_pos / block_token_num
  //   local_swa_idx    = global_block_idx - swa_block_idx_offsets[0]
  //                      (swa_block_idx_offsets[0] 是 SWA 窗口起始 FA block 的全局索引)
  // 若 token_pos < 0 或 local_swa_idx 不在 swa_block_ids 范围: 返回 -1
  // (kernel 在 store/load 时遇到 -1 会安全跳过, 见 store.cu fused_store_flashmla_cache_kernel)。
  // SWA pool 不共享主 KV 的 atb_kv_cache_base_blk_ids; 所有写入/读取 SWA pool 的位置参数
  // 都必须通过本函数 (而不是 GetPhysicalTokenPos) 计算。
  // NOTE: 当前底层算子尚未支持不连续 SWA block, 此处仍假定 swa_block_idx_offsets 内的索引连续,
  //       使用 swa_block_idx_offsets[0] 作为起始偏移做算术映射。
  inline int GetSwaPhysicalTokenPos(const std::vector<int>& swa_block_ids,
                                    const std::vector<size_t>& swa_block_idx_offsets, int token_pos) const {
    if (token_pos < 0) return -1;
    if (swa_block_idx_offsets.empty()) return -1;
    const int global_block_idx = token_pos / block_token_num_;
    const int local_swa_idx = global_block_idx - static_cast<int>(swa_block_idx_offsets[0]);
    if (local_swa_idx < 0 || local_swa_idx >= static_cast<int>(swa_block_ids.size())) {
      return -1;
    }
    return swa_block_ids[local_swa_idx] * block_token_num_ + (token_pos % block_token_num_);
  }

  const int layer_num_on_node_ = 0;
  std::shared_ptr<Context> context_;

  uint32_t attn_dp_group_id_ = 0;
  uint32_t attn_dp_rank_id_ = 0;

  int block_size_ = 0;      // 主 KV pool 单 block 字节数 (kv-cache layout 的 BlockCacheSize)
  int swa_block_size_ = 0;  // SWA pool 单 block 字节数 (SWA-cache layout 的 BlockCacheSize)
  int block_token_num_ = 0;
  int index_topk_ = 0;
  bool layer_kv_cache_ptr_initialized_ = false;

  static constexpr size_t kSwaSize = 128;
  static constexpr size_t kC4Ratio = 4;
  static constexpr size_t kC128Ratio = 128;
  static constexpr size_t kTopkBlockSize = 64;  // FlashMLA kernel performs block-style index reads
  static_assert(kSwaSize % kTopkBlockSize == 0, "kSwaSize must be a multiple of kTopkBlockSize");

  // Tensors shared by all input_infos
  Tensor raw_out_loc_tensor;
  Tensor swa_block_indices_tensor;
  Tensor c128_block_indices_tensor;
  Tensor c128_topk_lengths_tensor;

  // Pre-allocated tile_scheduler_metadata for c4/c128 compressed paths (one per q_seq_len)
  std::vector<Tensor> c4_tile_scheduler_metadatas;
  std::vector<Tensor> c128_tile_scheduler_metadatas;
  Tensor c4_num_splits_tensor;
  Tensor c128_num_splits_tensor;

 public:
  // Number of dp tokens in context/decode phase
  size_t dp_context_tokens = 0;
  size_t dp_decode_tokens = 0;

  // Tensors with public access
  // layer_swa_kv_cache_ptr_tensor layout:
  //   slot 0       : device_block_num (int64), SWA pool 的 block 总数。
  //   slot 1       : swa_block_size   (int64), SWA pool 单 block 字节数 (= primary stride for SWA store/sparse mla).
  //   slot 2..2+N-1: 每层 SWA base ptr (void*), = forward_req.swa_block_base_addr + per-layer swa_offset。
  // attention layer 通过 slot 1 拿 SWA stride, 与主 KV 的 block_size_ 解耦。
  Tensor layer_swa_kv_cache_ptr_tensor;
  Tensor c4_block_indices_tensor;

  // Per-layer base pointers for the c4 overlap tail regions; layout 与 SWA 类似但不带 swa_block_size:
  //   slot 0       : device_block_num (int64)
  //   slot 1..1+N-1: 每层 tail base ptr (= forward_req.swa_block_base_addr + per-layer tail offset)。
  // All layers are allocated; c128 / MTP layer pointers are valid but never read/written by kernels.
  Tensor layer_tail_attn_cache_ptr_tensor;
  Tensor layer_tail_indexer_cache_ptr_tensor;

  // 主 KV pool (kv-cache layout) 单 block 字节数。
  int GetBlockSize() const { return block_size_; }

  // SWA pool (SWA-cache layout) 单 block 字节数。compressor 的 tail copy 与 attention layer 的 SWA store
  // 都需要 SWA pool 自身的 stride，不能用主 KV pool 的 block_size_。
  int GetSwaBlockSize() const { return swa_block_size_; }
};

}  // namespace ksana_llm
