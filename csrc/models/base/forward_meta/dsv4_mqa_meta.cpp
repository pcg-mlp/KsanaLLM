/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "models/base/forward_meta/dsv4_mqa_meta.h"

#include <fmt/core.h>

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/flash_sparse_mla/flash_sparse_mla.h"
#endif
#include "device/device_utils.h"
#include "utils/memory_utils.h"

namespace ksana_llm {

Dsv4MqaMeta::Dsv4MqaMeta(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int layer_num_on_node,
                         int rank, std::shared_ptr<Context> context,
                         std::shared_ptr<BaseModelCacheLayout> block_cache_layout)
    : layer_num_on_node_(layer_num_on_node), context_(context) {
  attn_dp_group_id_ = rank / runtime_config.parallel_basic_config.attn_tensor_parallel_size;
  attn_dp_rank_id_ = rank % runtime_config.parallel_basic_config.attn_tensor_parallel_size;
  block_size_ = runtime_config.attn_backend_config.block_size;
  block_token_num_ = runtime_config.attn_backend_config.block_token_num;
  KLLM_CHECK_WITH_INFO(block_token_num_ % kC4Ratio == 0, "block_token_num must be a multiple of 4");
  KLLM_CHECK_WITH_INFO(block_token_num_ % kC128Ratio == 0, "block_token_num must be a multiple of 128");
  KLLM_CHECK_WITH_INFO(model_config.dsv4_attn_config.sliding_window == kSwaSize, "sliding_window must be 128");
  index_topk_ = model_config.dsa_config.index_topk;
  // SWA-cache layout 由 DeepSeekV4ModelCacheLayout::RegisterLayouts 无条件注册。
  KLLM_CHECK_WITH_INFO(block_cache_layout->IsBlockCacheTypeExist("SWA-cache"),
                       "Dsv4MqaMeta requires 'SWA-cache' layout to be registered.");
  swa_block_size_ = static_cast<int>(block_cache_layout->GetBlockCacheLayout("SWA-cache")->GetBlockCacheSize());
  KLLM_CHECK_WITH_INFO(swa_block_size_ > 0, "SWA-cache block size must be > 0");
  KLLM_LOG_INFO << fmt::format("Dsv4MqaMeta init: rank={} layer_num_on_node={} block_size={} swa_block_size={}", rank,
                               layer_num_on_node_, block_size_, swa_block_size_);

  const size_t max_token_num = runtime_config.max_step_token_num;
  raw_out_loc_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
  c128_topk_lengths_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
  swa_block_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num, kSwaSize});
  KLLM_CHECK_WITH_INFO(model_config.dsa_config.index_topk % kTopkBlockSize == 0,
                       "index_topk must be a multiple of kTopkBlockSize");
  c4_block_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
                                   {max_token_num, model_config.dsa_config.index_topk});  // computed by c4_indexer
  const size_t max_seq_len = runtime_config.max_seq_len;
  c128_block_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
                                     {max_token_num, RoundUp(max_seq_len / kC128Ratio, kTopkBlockSize)});

  // layer_swa_kv_cache_ptr_tensor 槽位 = 2 (device_block_num + swa_block_size) + layer_num_on_node。
  layer_swa_kv_cache_ptr_tensor =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {2ul + static_cast<size_t>(layer_num_on_node_)});
  // tail tensor 不携带 swa_block_size，slot 0 = device_block_num, slot 1.. 为 per-layer base ptr。
  layer_tail_attn_cache_ptr_tensor =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {1ul + static_cast<size_t>(layer_num_on_node_)});
  layer_tail_indexer_cache_ptr_tensor =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {1ul + static_cast<size_t>(layer_num_on_node_)});

  // Slot 1 of layer_swa_kv_cache_ptr_tensor 在构造期就把 swa_block_size 写入；
  // forward 阶段不会再变化 (由 SWA-cache layout 在启动时一次性确定)。
  *(layer_swa_kv_cache_ptr_tensor.GetPtr<int64_t>() + 1) = static_cast<int64_t>(swa_block_size_);

  // Pre-compute kv cache types and per-layer offsets once. The layout is fixed at startup,
  // so subsequent forward steps can reuse the cached results instead of recomputing them.
  // swa / tail_attn / tail_indexer 全部由 SWA-cache layout 提供。
  size_t* layer_swa_kv_cache_ptr = layer_swa_kv_cache_ptr_tensor.GetPtr<size_t>() + 2;
  size_t* layer_tail_attn_cache_ptr = layer_tail_attn_cache_ptr_tensor.GetPtr<size_t>() + 1;
  size_t* layer_tail_indexer_cache_ptr = layer_tail_indexer_cache_ptr_tensor.GetPtr<size_t>() + 1;
  const auto swa_block_layout = block_cache_layout->GetBlockCacheLayout("SWA-cache");
  for (int layer_idx = 0; layer_idx < layer_num_on_node_; ++layer_idx) {
    const auto layer_offsets = swa_block_layout->GetKvCacheLayerOffsets(layer_idx, /*kv_cache_type*/ "swa");
    KLLM_CHECK_WITH_INFO(layer_offsets.k_offset == layer_offsets.v_offset, "Dsv4MqaMeta requires k_offset == v_offset");
    *layer_swa_kv_cache_ptr++ = layer_offsets.k_offset;

    const auto tail_attn_offsets = swa_block_layout->GetKvCacheLayerOffsets(layer_idx, /*kv_cache_type*/ "tail_attn");
    *layer_tail_attn_cache_ptr++ = tail_attn_offsets.k_offset;

    const auto tail_indexer_offsets =
        swa_block_layout->GetKvCacheLayerOffsets(layer_idx, /*kv_cache_type*/ "tail_indexer");
    *layer_tail_indexer_cache_ptr++ = tail_indexer_offsets.k_offset;

    KLLM_LOG_DEBUG << fmt::format("layer {} swa_offset={} tail_attn_offset={} tail_indexer_offset={}", layer_idx,
                                  layer_offsets.k_offset, tail_attn_offsets.k_offset, tail_indexer_offsets.k_offset);
  }

#ifdef ENABLE_CUDA
  // Pre-allocate tile_scheduler_metadatas for c4/c128 compressed paths.
  // One entry per q_seq_len (1..decode_token_num_threshold), plus one extra for flash (prefill).
  // Hack shape[2] as init flag.
  const size_t head_num_per_tp = model_config.head_num / runtime_config.parallel_basic_config.attn_tensor_parallel_size;
  for (size_t q_len = 1; q_len <= GetDecodeTokenNumThreshold(); q_len++) {
    c4_tile_scheduler_metadatas.emplace_back(
        MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
        std::vector<size_t>{static_cast<size_t>(llm_kernels::nvidia::GetNumSmParts(head_num_per_tp, q_len)),
                            static_cast<size_t>(llm_kernels::nvidia::GetDecodingSchedMetaSize() / sizeof(int)), 1});
    c128_tile_scheduler_metadatas.emplace_back(
        MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
        std::vector<size_t>{static_cast<size_t>(llm_kernels::nvidia::GetNumSmParts(head_num_per_tp, q_len)),
                            static_cast<size_t>(llm_kernels::nvidia::GetDecodingSchedMetaSize() / sizeof(int)), 1});
  }
  // Extra entry for flash (prefill): q_len=1 gives max num_sm_parts
  c4_tile_scheduler_metadatas.emplace_back(
      MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
      std::vector<size_t>{static_cast<size_t>(llm_kernels::nvidia::GetNumSmParts(head_num_per_tp, 1)),
                          static_cast<size_t>(llm_kernels::nvidia::GetDecodingSchedMetaSize() / sizeof(int)), 1});
  c128_tile_scheduler_metadatas.emplace_back(
      MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
      std::vector<size_t>{static_cast<size_t>(llm_kernels::nvidia::GetNumSmParts(head_num_per_tp, 1)),
                          static_cast<size_t>(llm_kernels::nvidia::GetDecodingSchedMetaSize() / sizeof(int)), 1});
  // Pre-allocate num_splits for c4/c128
  const size_t num_splits_size = runtime_config.max_batch_size + 1 + GetDecodeTokenNumThreshold();
  c4_num_splits_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {num_splits_size});
  c128_num_splits_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {num_splits_size});
#endif
}

// Upload input's persistent host vector to a device tensor view, updating shared tensor offset.
#define HOST_TO_DEVICE(name, input)                                                                           \
  do {                                                                                                        \
    KLLM_LOG_DEBUG << #name "_host: " << (input).name##_host;                                                 \
    (input).name##_tensor = name##_tensor.GetView({(input).name##_host.size()}, name##_tensor.shape[0]);      \
    name##_tensor.shape[0] += (input).name##_tensor.GetElementNumber();                                       \
    MemcpyAsync((input).name##_tensor.GetPtr<void>(), (input).name##_host.data(),                             \
                (input).name##_host.size() * sizeof(std::decay_t<decltype((input).name##_host)>::value_type), \
                MEMCPY_HOST_TO_DEVICE, context_->device->GetH2DStream());                                     \
  } while (0)

void Dsv4MqaMeta::Prepare(const std::vector<ForwardRequest>& forward_reqs) {
  // Reset the offsets of shared tensors
  raw_out_loc_tensor.shape = {0};
  swa_block_indices_tensor.shape = {0};
  c128_block_indices_tensor.shape = {0};
  c128_topk_lengths_tensor.shape = {0};
  c4_num_splits_tensor.shape = {0};
  c128_num_splits_tensor.shape = {0};

  // SWA / tail base ptr 在第一次 Prepare 时从 forward_req 拿到, 整个进程生命周期内不变。
  if (__builtin_expect(!layer_kv_cache_ptr_initialized_, false)) {  // unlikely
    layer_kv_cache_ptr_initialized_ = true;

    // 选第一个携带 SWA 资源的 req 取 base ptr 与 pool 总 block 数;
    // swa_block_base_addr 由 model_builder 在 SWA pool 启用时填充。
    void* swa_pool_base = nullptr;
    int64_t swa_pool_block_num = 0;
    for (const auto& req : forward_reqs) {
      if (req.swa_block_base_addr != nullptr) {
        swa_pool_base = req.swa_block_base_addr;
        swa_pool_block_num = req.swa_pool_block_num;
        break;
      }
    }
    KLLM_CHECK_WITH_INFO(swa_pool_base != nullptr,
                         "Dsv4MqaMeta::Prepare: forward_req.swa_block_base_addr is nullptr "
                         "(SWA pool not enabled or model_builder did not populate it).");
    KLLM_CHECK_WITH_INFO(swa_pool_block_num > 0, "Dsv4MqaMeta::Prepare: swa_pool_block_num must be > 0.");

    int64_t* swa_meta = layer_swa_kv_cache_ptr_tensor.GetPtr<int64_t>();
    int64_t* tail_attn_meta = layer_tail_attn_cache_ptr_tensor.GetPtr<int64_t>();
    int64_t* tail_indexer_meta = layer_tail_indexer_cache_ptr_tensor.GetPtr<int64_t>();
    swa_meta[0] = swa_pool_block_num;
    // swa_meta[1] (swa_block_size) 已在构造期写入，此处保持不变。
    tail_attn_meta[0] = swa_pool_block_num;
    tail_indexer_meta[0] = swa_pool_block_num;

    // SWA / tail per-layer base ptr = swa_pool_base + per-layer offset (offset 已在构造期预存)。
    void** layer_swa_kv_cache_ptr = layer_swa_kv_cache_ptr_tensor.GetPtr<void*>() + 2;
    void** layer_tail_attn_cache_ptr = layer_tail_attn_cache_ptr_tensor.GetPtr<void*>() + 1;
    void** layer_tail_indexer_cache_ptr = layer_tail_indexer_cache_ptr_tensor.GetPtr<void*>() + 1;
    uint8_t* const cache_base_ptr = static_cast<uint8_t*>(swa_pool_base);
    KLLM_LOG_INFO << fmt::format(
        "init dsv4 SWA pool base_ptr={} swa_pool_block_num={} swa_block_size={} block_size(main)={}",
        static_cast<const void*>(cache_base_ptr), swa_pool_block_num, swa_block_size_, block_size_);
    for (int layer_idx = 0; layer_idx < layer_num_on_node_; ++layer_idx) {
      *layer_swa_kv_cache_ptr = cache_base_ptr + reinterpret_cast<size_t>(*layer_swa_kv_cache_ptr);
      *layer_tail_attn_cache_ptr = cache_base_ptr + reinterpret_cast<size_t>(*layer_tail_attn_cache_ptr);
      *layer_tail_indexer_cache_ptr = cache_base_ptr + reinterpret_cast<size_t>(*layer_tail_indexer_cache_ptr);
      KLLM_LOG_DEBUG << fmt::format("layer {} swa_base={} tail_attn_base={} tail_indexer_base={}", layer_idx,
                                    *layer_swa_kv_cache_ptr, *layer_tail_attn_cache_ptr, *layer_tail_indexer_cache_ptr);
      ++layer_swa_kv_cache_ptr;
      ++layer_tail_attn_cache_ptr;
      ++layer_tail_indexer_cache_ptr;
    }
  }

  // Reset meta
  flash_input.Reset();
  page_inputs.clear();
  dp_context_tokens = 0;
  dp_decode_tokens = 0;

  // Categorize requests
  for (const auto& req : forward_reqs) {
    if (req.attn_dp_group_id == attn_dp_group_id_) {
      const size_t extend_len = req.GetInputIdsLength();
      if (GetForwardRequestType(req) == ForwardRequestType::kFlash) {
        flash_input.dp_reqs.emplace_back(&req);
        flash_input.total_dp_input_ids_len += extend_len;
        dp_context_tokens += extend_len;
      } else {
        if (page_inputs.empty() || page_inputs.back().q_seq_len != extend_len) {
          // `page_inputs.back().q_seq_len` should be greater than `extend_len`,
          // since requests in the same dp group are sorted in descending order by their token numbers
          page_inputs.emplace_back().q_seq_len = extend_len;
        }
        auto& page_input = page_inputs.back();
        page_input.dp_reqs.emplace_back(&req);
        page_input.total_dp_input_ids_len += extend_len;
        dp_decode_tokens += extend_len;
      }
    }
  }

  if (!flash_input.dp_reqs.empty()) {
    PrepareOutLoc(flash_input);
    PrepareBlockIndices(flash_input);
    PrepareTopkLengths(flash_input);
    // Assign c4/c128 tile_scheduler_metadata for flash (prefill) path.
    // Flash uses the extra entry at index GetDecodeTokenNumThreshold().
    flash_input.c4_tile_scheduler_metadata = c4_tile_scheduler_metadatas.back();
    flash_input.c4_tile_scheduler_metadata.shape[2] = 0;  // mark as uninitialized
    flash_input.c128_tile_scheduler_metadata = c128_tile_scheduler_metadatas.back();
    flash_input.c128_tile_scheduler_metadata.shape[2] = 0;  // mark as uninitialized
    // Assign c4/c128 num_splits views
    const size_t flash_batch = flash_input.dp_reqs.size();
    flash_input.c4_num_splits = c4_num_splits_tensor.GetView({flash_batch + 1}, c4_num_splits_tensor.shape[0]);
    c4_num_splits_tensor.shape[0] += flash_input.c4_num_splits.GetElementNumber();
    flash_input.c128_num_splits = c128_num_splits_tensor.GetView({flash_batch + 1}, c128_num_splits_tensor.shape[0]);
    c128_num_splits_tensor.shape[0] += flash_input.c128_num_splits.GetElementNumber();
  }
  for (auto& page_input : page_inputs) {
    PrepareOutLoc(page_input);
    PrepareBlockIndices(page_input);
    PrepareTopkLengths(page_input);
    // Assign c4/c128 tile_scheduler_metadata for page (decode) path.
    const size_t idx = page_input.q_seq_len - 1;
    page_input.c4_tile_scheduler_metadata = c4_tile_scheduler_metadatas[idx];
    page_input.c4_tile_scheduler_metadata.shape[2] = 0;  // mark as uninitialized
    page_input.c128_tile_scheduler_metadata = c128_tile_scheduler_metadatas[idx];
    page_input.c128_tile_scheduler_metadata.shape[2] = 0;  // mark as uninitialized
    // Assign c4/c128 num_splits views
    const size_t page_batch = page_input.dp_reqs.size();
    page_input.c4_num_splits = c4_num_splits_tensor.GetView({page_batch + 1}, c4_num_splits_tensor.shape[0]);
    c4_num_splits_tensor.shape[0] += page_input.c4_num_splits.GetElementNumber();
    page_input.c128_num_splits = c128_num_splits_tensor.GetView({page_batch + 1}, c128_num_splits_tensor.shape[0]);
    c128_num_splits_tensor.shape[0] += page_input.c128_num_splits.GetElementNumber();
  }
}

void Dsv4MqaMeta::PrepareOutLoc(input_info& input) {
  for (const auto* req : input.dp_reqs) {
    const int extend_len = req->GetInputIdsLength();
    const int seq_len = req->forwarding_tokens->size();
    const int prefix_len = seq_len - extend_len;
    // swa_block_ids = SWA pool 内分配给本 req 的 block 列表 (与主 KV pool 独立编址);
    // raw_out_loc 写入 SWA pool, 必须使用 SWA pool 自身的物理位置。
    const auto& swa_block_ids = req->swa_kv_cache_blocks;
    const auto& swa_block_idx_offsets = req->swa_kv_cache_block_idx_offsets;
    for (int i = prefix_len; i < seq_len; ++i) {
      // raw_out_loc: 写入 SWA pool (attention layer 中 kcache_ptr 指向 SWA pool base)。
      // 窗口外的 token (i 不在 swa_block_ids 范围内) 返回 -1, kernel 安全跳过。
      input.raw_out_loc_host.push_back(GetSwaPhysicalTokenPos(swa_block_ids, swa_block_idx_offsets, i));
    }
  }

  HOST_TO_DEVICE(raw_out_loc, input);
}

void Dsv4MqaMeta::PrepareBlockIndices(input_info& input) {
  int c128_max_seq_len_clamp1 = 1;  // Clamp to 1 to avoid issues in FlashMLA kernel
  for (const auto* req : input.dp_reqs) {
    const int c128_seq_len = req->forwarding_tokens->size() / kC128Ratio;
    if (c128_seq_len > c128_max_seq_len_clamp1) {
      c128_max_seq_len_clamp1 = c128_seq_len;
    }
  }
  const int aligned_c128_max_seq_len_clamp1 =
      RoundUp(c128_max_seq_len_clamp1, kTopkBlockSize);  // FlashMLA kernel requires 64-element aligned indices

  // SWA 物理 token 位置映射由成员函数 GetSwaPhysicalTokenPos 实现,
  // 与 PrepareOutLoc.raw_out_loc 共用同一映射逻辑, 保证编址一致。

  for (const auto* req : input.dp_reqs) {
    const int extend_len = req->GetInputIdsLength();
    const int seq_len = req->forwarding_tokens->size();
    const int prefix_len = seq_len - extend_len;
    const auto& block_ids = req->atb_kv_cache_base_blk_ids[attn_dp_rank_id_];  // 仅用于 c128 (主 KV 路径)
    const auto& swa_block_ids = req->swa_kv_cache_blocks;
    const auto& swa_block_idx_offsets = req->swa_kv_cache_block_idx_offsets;
    for (int i = prefix_len; i < seq_len; ++i) {
      for (int j = i - kSwaSize + 1; j <= i; ++j) {
        input.swa_block_indices_host.push_back(GetSwaPhysicalTokenPos(swa_block_ids, swa_block_idx_offsets, j));
      }
      int c128_seq_len = 0;
      for (int j = kC128Ratio - 1; j <= i; j += kC128Ratio) {
        input.c128_block_indices_host.push_back(GetPhysicalTokenPos(block_ids, j, kC128Ratio));
        ++c128_seq_len;
      }
      while (c128_seq_len < aligned_c128_max_seq_len_clamp1) {  // Pad to the max length
        input.c128_block_indices_host.push_back(-1);
        ++c128_seq_len;
      }
    }
  }

  HOST_TO_DEVICE(swa_block_indices, input);
  HOST_TO_DEVICE(c128_block_indices, input);
  // Correctly set the 2D shape
  input.swa_block_indices_tensor.shape = {input.total_dp_input_ids_len, kSwaSize};
  input.c128_block_indices_tensor.shape = {input.total_dp_input_ids_len,
                                           static_cast<size_t>(aligned_c128_max_seq_len_clamp1)};
}

void Dsv4MqaMeta::PrepareTopkLengths(input_info& input) {
  for (const auto* req : input.dp_reqs) {
    const int extend_len = req->GetInputIdsLength();
    const int seq_len = req->forwarding_tokens->size();
    const int c128_topk_len = static_cast<int>(seq_len / kC128Ratio);
    for (int i = 0; i < extend_len; ++i) {
      input.c128_topk_lengths_host.push_back(c128_topk_len);
    }
  }

  HOST_TO_DEVICE(c128_topk_lengths, input);
}

#undef HOST_TO_DEVICE

}  // namespace ksana_llm
