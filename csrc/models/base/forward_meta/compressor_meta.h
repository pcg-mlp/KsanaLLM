/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "configure/context.h"
#include "configure/model_config.h"
#include "runtime/forward_request.h"
#include "utils/tensor.h"

namespace ksana_llm {

class CompressorMeta {
 public:
  CompressorMeta(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int layer_num_on_node,
                 int layer_start_on_node, int rank, std::shared_ptr<Context> context,
                 std::shared_ptr<BaseModelCacheLayout> block_cache_layout);

  void Prepare(const std::vector<ForwardRequest>& forward_reqs);

  struct input_info {
    std::vector<const ForwardRequest*> dp_reqs;

    // Tensors in input_info are views of the above shared tensors
    // Their shape is used to locate the offset of each input_info in the shared tensors
    Tensor c4_indices_tensor;
    Tensor c128_indices_tensor;
    Tensor c4_indices_bundle_tensor;
    Tensor c128_load_indices_tensor;
    Tensor c4_indices_prev_tensor;
    Tensor seq_lens_tensor;
    Tensor extend_lens_tensor;
    Tensor c4_out_loc_tensor;
    Tensor c128_out_loc_tensor;

    // Persistent host vectors to prevent destruction during MemcpyAsync
    std::vector<int> c4_indices_host;
    std::vector<int> c128_indices_host;
    std::vector<int> c4_indices_bundle_host;
    std::vector<int> c128_load_indices_host;
    std::vector<int> c4_indices_prev_host;
    std::vector<int> seq_lens_host;
    std::vector<int> extend_lens_host;
    std::vector<int> c4_out_loc_host;
    std::vector<int> c128_out_loc_host;

    void Reset() {
      dp_reqs.clear();
      c4_indices_host.clear();
      c128_indices_host.clear();
      c4_indices_bundle_host.clear();
      c128_load_indices_host.clear();
      c4_indices_prev_host.clear();
      seq_lens_host.clear();
      extend_lens_host.clear();
      c4_out_loc_host.clear();
      c128_out_loc_host.clear();
    }
  };

  // Category is tightly coupled to underlying kernel
  input_info flash_input;  // extend_len is non-specialized, use flash attention
  input_info page_input;   // extend_len is 1, use page attention

 private:
  // Compressor kernel only supports q_seq_len = 1 in decode phase
  ForwardRequestType GetForwardRequestType(const ForwardRequest& forward_request) const {
    if (forward_request.GetInputIdsLength() == 1 && forward_request.kv_cached_token_num > 0) {
      return ForwardRequestType::kPage;
    } else {
      return ForwardRequestType::kFlash;
    }
  }

  void PreparePrefillPlan(const input_info& input);
  void PrepareTailIndices(const std::vector<ForwardRequest>& reqs, const std::unordered_set<int64_t>& new_req_ids);
  void PrepareIndices(input_info& input, bool is_paged = false);
  void PrepareOutLoc(input_info& input);
  void PrepareSeqLens(std::vector<int>& host, const ForwardRequest& req);
  void PrepareExtendLens(std::vector<int>& host, const ForwardRequest& req);

  // Translate a global block index to the physical SWA block ID.
  // swa_block_ids starts at swa_offset in global numbering; asserts bounds.
  static int MapGlobalBlockToSwa(const std::vector<int>& swa_block_ids,
                                 const std::vector<size_t>& swa_block_idx_offsets, int global_block_idx,
                                 const char* purpose);

  // Convert a logical token position to a physical cache position given block IDs and block size.
  inline int GetPhysicalTokenPos(const std::vector<int32_t>& block_ids, int token_pos, int compress_ratio) const {
    return token_pos < 0 ? -1
                         : block_ids[token_pos / block_token_num_] * (block_token_num_ / compress_ratio) +
                               (token_pos / compress_ratio) % (block_token_num_ / compress_ratio);
  }

  // Compute the sub-slot index for the group containing the last token of a sequence.
  // token_num is the total number of tokens; returns ((token_num - 1) / ratio) % slot_num.
  // Note: token_num==0 yields (-1)/ratio=0, returns 0, which is safe.
  int GetSubSlotId(int token_num, int compress_ratio) const {
    if (compress_ratio == kC4Ratio) {
      return (token_num - 1) / kC4Ratio % c4_slot_num;
    } else {  // compress_ratio == kC128Ratio
      return (token_num - 1) / kC128Ratio % c128_slot_num;
    }
  }

  // Compute the previous C4 sub-slot index by wrapping around the slot ring.
  // Returns (slot_id + c4_slot_num - 1) % c4_slot_num.
  int GetC4PrevSlotId(int slot_id) const { return (slot_id + c4_slot_num - 1) % c4_slot_num; }

  // Compute the page index in the state buffer for a (slot_id, sub_slot) pair.
  // compress_ratio: kC4Ratio  → c4_slot_num  / c4_layer_num_on_node_,
  //                 kC128Ratio → c128_slot_num / c128_layer_num_on_node_.
  int GetPageIndex(int slot_id, int sub_slot, int compress_ratio) const {
    if (compress_ratio == kC4Ratio) {
      return (slot_id * c4_slot_num + sub_slot) * c4_layer_num_on_node_;
    } else {  // compress_ratio == kC128Ratio
      return (slot_id * c128_slot_num + sub_slot) * c128_layer_num_on_node_;
    }
  }

  const int layer_num_on_node_ = 0;
  const int layer_start_on_node_ = 0;
  std::shared_ptr<Context> context_;

  uint32_t attn_dp_rank_id_ = 0;
  uint32_t attn_dp_group_id_ = 0;

  int c4_layer_num_on_node_ = 0;
  int c128_layer_num_on_node_ = 0;
  bool layer_state_ptr_initialized_ = false;

  static constexpr int kC4Ratio = 4;
  static constexpr int kC128Ratio = 128;

  // Tensors shared by all input_infos
  Tensor c4_indices_tensor;
  Tensor c128_indices_tensor;
  Tensor c4_indices_bundle_tensor;
  Tensor c128_load_indices_tensor;
  Tensor c4_indices_prev_tensor;
  Tensor seq_lens_tensor;
  Tensor extend_lens_tensor;
  Tensor c4_out_loc_tensor;
  Tensor c128_out_loc_tensor;

  int block_token_num_ = 0;  // cached from runtime config; used by tail index math

  // TODO(yfnjin): Temporarily manage the state buffer here
  Tensor c4_state_buffer;
  Tensor c4_indexer_state_buffer;
  Tensor c128_state_buffer;
  std::unordered_map<int64_t, int> req_to_slot_;  // map req_id to block slot
  std::queue<int> free_slot_;

  std::vector<int> tail_read_block_host;
  std::vector<int> tail_target_indices_host;
  std::vector<int> tail_write_indices_host;

 public:
  // Number of dp tokens in context/decode phase
  size_t dp_context_tokens = 0;
  size_t dp_decode_tokens = 0;

  int c4_slot_num = 2;    // C4 state slots occupied per request
  int c128_slot_num = 1;  // C128 state slots occupied per request

  // Tensors with public access
  Tensor layer_c4_state_ptr_tensor;
  Tensor layer_c4_indexer_state_ptr_tensor;
  Tensor layer_c128_state_ptr_tensor;
  Tensor c4_compress_plan_tensor;
  Tensor c4_write_plan_tensor;
  Tensor c128_compress_plan_tensor;
  Tensor c128_write_plan_tensor;

  // c4 overlap tail 持久化的索引 (prefix cache 使用)
  Tensor tail_read_block_tensor;
  Tensor tail_target_indices_tensor;
  Tensor tail_write_indices_tensor;
};

}  // namespace ksana_llm
