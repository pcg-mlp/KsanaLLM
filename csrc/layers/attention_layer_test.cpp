/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/layer_test_base.h"

#include "layers/attention_layer.h"
#include "layers/flash_attention_layer.h"
#include "layers/flashinfer_resource_manager.h"
#include "layers/paged_attention_layer.h"

namespace ksana_llm {

TEST_F(LayerTest, AttentionLayerTest) {
#ifndef ENABLE_CUDA
  GTEST_SKIP();
#endif

#ifdef ENABLE_CUDA
  FlashAttentionLayer flash_attention_layer;
  QuantMode quant_mode = QUANT_NONE;
  int head_num = 32;
  int kv_head_num = 32;
  int size_per_head = 128;
  int rotary_embedding = 128;
  int max_position_embeddings = 2048;
  int max_batch_size = 1;
  size_t attn_temperature_tuning = 0;
  float attn_scale = 0;
  size_t floor_scale = 0;
  Tensor mrope_section_ptr;
  CreateHalfDataTypeTensor(mrope_section_ptr, {(size_t)rotary_embedding, (size_t)max_position_embeddings},
                           GetDataType<half>());
  bool enable_qk_pre_norm_before_rotary_pos = false;
  int stride_size = head_num * size_per_head;
  float k_scale = 1.0f;
  float v_scale = 1.0f;
  float rope_theta = 10000.0f;
  bool is_neox = true;
  uint32_t qk_nope_head_dim = 0;
  uint32_t v_head_dim = 0;
  uint32_t qk_rope_head_dim = 0;
  uint32_t q_lora_rank = 0;
  uint32_t kv_lora_rank = 0;
  Tensor cos_sin_cache_tensor;
  RoPEScalingFactor rope_scaling_factor;
  CreateHalfDataTypeTensor(cos_sin_cache_tensor, {(size_t)rotary_embedding, (size_t)max_position_embeddings},
                           GetDataType<half>());
  EXPECT_TRUE(flash_attention_layer
                  .Init({quant_mode,
                         static_cast<float>(0),
                         static_cast<bool>(false),
                         static_cast<int>(0),
                         static_cast<int>(1),
                         static_cast<int>(2048),
                         head_num,
                         kv_head_num,
                         size_per_head,
                         stride_size,
                         static_cast<size_t>(1),
                         TYPE_FP16,
                         k_scale,
                         v_scale,
                         rotary_embedding,
                         rope_theta,
                         v_head_dim,
                         qk_rope_head_dim,
                         qk_nope_head_dim,
                         q_lora_rank,
                         kv_lora_rank,
                         is_neox,
                         PositionEncoding::ROPE,
                         std::any(cos_sin_cache_tensor.GetPtr<void>()),
                         rope_scaling_factor,
                         max_batch_size,
                         attn_temperature_tuning,
                         attn_scale,
                         floor_scale,
                         true,
                         mrope_section_ptr,
                         enable_qk_pre_norm_before_rotary_pos},
                        runtime_config, context_, 0)
                  .OK());

  Tensor qkv, input_len, prefix_offsets, pos, mask, flag_tensor, flexible_rotary_embedding_pos,
      flexible_rotary_embedding_mask, dst_flexible_kv_cache_tensor, src_flexible_kv_cache_tensor,
      dst_flexible_token_idx_tensor, src_flexible_token_idx_tensor;
  std::vector<size_t> input_shape = {2, 12288};
  CreateHalfDataTypeTensor(qkv, input_shape, GetDataType<half>());
  CreateHalfDataTypeTensor(input_len, {2}, GetDataType<uint64_t>(), sizeof(uint64_t));
  CreateHalfDataTypeTensor(prefix_offsets, {2}, GetDataType<int>(), sizeof(int));
  CreateHalfDataTypeTensor(pos, {2}, GetDataType<uint64_t>(), /*dtype_size*/ sizeof(uint64_t));
  CreateHalfDataTypeTensor(mask, {2}, GetDataType<uint64_t>(), /*dtype_size*/ sizeof(uint64_t));

  flag_tensor = Tensor(MemoryLocation::LOCATION_HOST, TYPE_BOOL, {1});

  CreateHalfDataTypeTensor(flexible_rotary_embedding_pos, {0}, GetDataType<int>(), sizeof(int));
  CreateHalfDataTypeTensor(flexible_rotary_embedding_mask, {0}, GetDataType<int>(), sizeof(int));
  CreateHalfDataTypeTensor(dst_flexible_kv_cache_tensor, {0}, GetDataType<int>(), sizeof(int));
  CreateHalfDataTypeTensor(src_flexible_kv_cache_tensor, {0}, GetDataType<int>(), sizeof(int));
  CreateHalfDataTypeTensor(dst_flexible_token_idx_tensor, {0}, GetDataType<int>(), sizeof(int));
  CreateHalfDataTypeTensor(src_flexible_token_idx_tensor, {0}, GetDataType<int>(), sizeof(int));
  void* pos_ptr = pos.GetPtr<void>();
  std::vector<uint64_t> pos_cpu({0, 1});
  Memcpy(pos_ptr, pos_cpu.data(), pos_cpu.size() * sizeof(uint64_t), MEMCPY_HOST_TO_DEVICE);
  void* mask_ptr = mask.GetPtr<void>();
  std::vector<uint64_t> mask_cpu({1, 1});
  Memcpy(mask_ptr, mask_cpu.data(), mask_cpu.size() * sizeof(uint64_t), MEMCPY_HOST_TO_DEVICE);
  void* input_len_ptr = input_len.GetPtr<void>();
  std::vector<uint64_t> input_len_cpu({0, 2});
  Memcpy(input_len_ptr, input_len_cpu.data(), input_len_cpu.size() * sizeof(uint64_t), MEMCPY_HOST_TO_DEVICE);
  Memset(prefix_offsets.GetPtr<void>(), 0, 2 * sizeof(int));
  flag_tensor.GetPtr<bool>()[0] = true;  // use_cache
  Tensor output_tensor;
  CreateHalfDataTypeTensor(output_tensor, input_shape, GetDataType<half>());
  std::vector<Tensor> output_tensors = {output_tensor};

  int block_size = runtime_config.attn_backend_config.block_size;
  std::vector<int> h_block_offsets = {0, 1};
  Tensor block_offsets;
  CreateHalfDataTypeTensor(block_offsets, {h_block_offsets.size()}, GetDataType<int>(), sizeof(int));
  Memcpy(block_offsets.GetPtr<void>(), h_block_offsets.data(), h_block_offsets.size() * sizeof(int),
         MEMCPY_HOST_TO_DEVICE);
  // 为 kv_list 分配内存并初始化
  Tensor kv_list;
  CreateHalfDataTypeTensor(kv_list, {static_cast<uint64_t>(h_block_offsets.back() * 20)}, GetDataType<uint64_t>());
  std::vector<void*> h_kv_list_ptrs(h_block_offsets.back() * 2);
  for (size_t i = 0; i < h_kv_list_ptrs.size(); i++) {
    Malloc(&h_kv_list_ptrs[i], block_size);
  }
  Memcpy(kv_list.GetPtr<void>(), h_kv_list_ptrs.data(), h_kv_list_ptrs.size() * sizeof(void*), MEMCPY_HOST_TO_DEVICE);

  // For blocke_prefill.
  Tensor layer_kv_cache_ptr_tensor;
  layer_kv_cache_ptr_tensor = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {static_cast<uint64_t>(1 + 2)});
  int64_t* kv_cache_block_num = layer_kv_cache_ptr_tensor.GetPtr<int64_t>();
  *kv_cache_block_num = static_cast<uint64_t>(1);
  void** layer_kv_cache_ptr = layer_kv_cache_ptr_tensor.GetPtr<void*>() + 1;
  layer_kv_cache_ptr[0] = h_kv_list_ptrs[0];
  layer_kv_cache_ptr[1] = h_kv_list_ptrs[1];

  std::vector<int32_t> multi_token_request_block_table_host = {0};
  Tensor multi_token_request_block_table;
  multi_token_request_block_table =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {static_cast<uint64_t>(1), static_cast<uint64_t>(1)});
  Memcpy(multi_token_request_block_table.GetPtr<void>(), multi_token_request_block_table_host.data(),
         multi_token_request_block_table_host.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE);

  Tensor empty_q_norm_weight;
  Tensor empty_k_norm_weight;
  FlashAttentionLayer::CudaForwardParams flash_params;
  flash_params.qkv = &qkv;
  flash_params.token_offset = &input_len;
  flash_params.kv_list = &kv_list;
  flash_params.prefix_offset = &prefix_offsets;
  flash_params.kv_cache_offset = &block_offsets;
  flash_params.rotary_embedding_pos = &pos;
  flash_params.rotary_embedding_mask = &mask;
  flash_params.src_flexible_rotary_embedding_pos = &flexible_rotary_embedding_pos;
  flash_params.flexible_rotary_embedding_mask = &flexible_rotary_embedding_mask;
  flash_params.dst_flexible_kv_cache = &dst_flexible_kv_cache_tensor;
  flash_params.src_flexible_kv_cache = &src_flexible_kv_cache_tensor;
  flash_params.dst_flexible_token_idx = &dst_flexible_token_idx_tensor;
  flash_params.src_flexible_token_idx = &src_flexible_token_idx_tensor;
  flash_params.flexible_offset_uint64 = &prefix_offsets;
  // Scalars formerly encoded in forward_shape.shape[2/6/7/8/12]:
  flash_params.layer_block_num = 1;        // shape[2]
  flash_params.decode_token_num = 0;       // shape[12] (pure-prefill batch)
  flash_params.max_forwarding_tokens = 2;  // shape[6]
  flash_params.batch_size = 1;             // shape[7]
  flash_params.max_tokens = 2;             // shape[8]
  flash_params.query_layernorm_weight = &empty_q_norm_weight;
  flash_params.key_layernorm_weight = &empty_k_norm_weight;
  flash_params.use_cache = flag_tensor.GetPtr<bool>()[0];
  flash_params.kv_cache_base_ptr = &layer_kv_cache_ptr_tensor;
  flash_params.block_table = &multi_token_request_block_table;
  flash_params.input_without_prefix_offset = &input_len;
  EXPECT_TRUE(flash_attention_layer.Forward(flash_params, &output_tensors[0]).OK());
  PagedAttentionLayer attention_layer;
  EXPECT_TRUE(attention_layer
                  .Init({quant_mode,
                         static_cast<float>(0),
                         static_cast<bool>(false),
                         static_cast<int>(1),
                         static_cast<int>(2),
                         static_cast<int>(2048),
                         static_cast<int>(head_num),
                         kv_head_num,
                         static_cast<int>(size_per_head),
                         stride_size,
                         static_cast<size_t>(1),
                         TYPE_FP16,
                         k_scale,
                         v_scale,
                         rotary_embedding,
                         rope_theta,
                         static_cast<uint32_t>(0),
                         static_cast<uint32_t>(0),
                         static_cast<uint32_t>(0),
                         static_cast<uint32_t>(0),
                         static_cast<uint32_t>(0),
                         is_neox,
                         PositionEncoding::ROPE,
                         std::any(cos_sin_cache_tensor.GetPtr<void>()),
                         rope_scaling_factor,
                         max_batch_size,
                         attn_temperature_tuning,
                         attn_scale,
                         floor_scale,
                         false,
                         nullptr,
                         enable_qk_pre_norm_before_rotary_pos},
                        runtime_config, context_, 0)
                  .OK());
#endif
}

// Exercises the hpc-ops attention paths (flash_attn_impl=hpc) so the HPC branches in
// AttenVarlen (paged prefill) and InvokePagedAttention (paged decode) get covered.
// hpc-ops kernels are SM90+ (Hopper) only, so the test skips on older GPUs / no-CUDA.
// This is a coverage + smoke test: it only asserts Forward().OK() and output shape,
// it does NOT check numerical correctness.
TEST_F(LayerTest, HpcPagedAttentionLayerTest) {
#ifndef ENABLE_CUDA
  GTEST_SKIP();
#else
  if (GetCurrentDeviceSmVersion() < 90) {
    GTEST_SKIP() << "hpc-ops attention requires SM90+ (Hopper).";
  }

  // ---- HPC-supported shape: bf16, head_size=128, GQA group=4, block_size=32 ----
  const int head_num = 32;
  const int kv_head_num = 8;  // GQA group = head_num / kv_head_num = 4
  const int size_per_head = 128;
  const int rotary_embedding = 128;
  const int stride_size = (head_num + kv_head_num * 2) * size_per_head;  // fused QKV row stride = 6144
  const size_t block_token_num = 32;                                    // block_size in {32, 64}
  const int max_position_embeddings = 2048;
  const int max_batch_size = 1;
  const size_t attn_temperature_tuning = 0;
  const float attn_scale = 0;
  const size_t floor_scale = 0;
  const float k_scale = 1.0f;
  const float v_scale = 1.0f;
  const float rope_theta = 10000.0f;
  const bool is_neox = true;
  const bool enable_qk_pre_norm_before_rotary_pos = false;
  const int qkv_cols = stride_size;                        // (num_heads + 2*num_kv_heads) * head_size
  const int out_cols = head_num * size_per_head;           // attention output width
  const size_t kv_block_bytes = 1 << 20;                   // 1 MiB per K/V block, >> what the kernels touch
  QuantMode quant_mode = QUANT_NONE;
  RoPEScalingFactor rope_scaling_factor;

  Tensor cos_sin_cache_tensor;
  CreateHalfDataTypeTensor(cos_sin_cache_tensor, {(size_t)rotary_embedding, (size_t)max_position_embeddings},
                           GetDataType<half>());
  Tensor empty_mrope_section;
  Tensor empty_q_norm_weight;
  Tensor empty_k_norm_weight;

  // ---- Enable hpc-ops attention in the global Environment (save + restore) ----
  // IsUsingHpcAttention() reads flash_attn_impl_choice from the global Environment singleton,
  // while the layers read the RuntimeConfig passed to Init(); keep both consistent.
  const auto& env = Singleton<Environment>::GetInstance();
  AttnBackendConfig original_attn_cfg;
  env->GetAttnBackendConfig(original_attn_cfg);

  AttnBackendConfig hpc_cfg = original_attn_cfg;
  hpc_cfg.flash_attn_impl_choice = AttnBackendConfig::FlashAttnImplChoice::HPC;
  hpc_cfg.enable_blocked_multi_token_forwarding_kv = true;  // precondition for reaching both HPC branches
  hpc_cfg.use_flashinfer_for_decode = false;
  hpc_cfg.kv_cache_dtype = TYPE_BF16;  // non-fp8 KV cache
  hpc_cfg.block_token_num = block_token_num;
  hpc_cfg.block_size = kv_block_bytes;
  env->SetAttnBackendConfig(hpc_cfg);

  runtime_config.inter_data_type = TYPE_BF16;
  runtime_config.attn_backend_config = hpc_cfg;

  // Common Init parameter list (mirrors PagedAttention creation ordering in
  // csrc/modules/basic/paged_attention.cpp). All layers use layer_index=0 / layer_num=1
  // so k_list / layer_kv_cache_ptr indexing stays at offset 0.
  auto make_init_params = [&](bool is_multi_token_forward) {
    return std::vector<std::any>{quant_mode,
                                 static_cast<float>(0),
                                 static_cast<bool>(false),
                                 static_cast<int>(0),  // layer_index
                                 static_cast<int>(1),  // layer_num
                                 static_cast<int>(max_position_embeddings),
                                 head_num,
                                 kv_head_num,
                                 size_per_head,
                                 stride_size,
                                 static_cast<size_t>(1),
                                 TYPE_BF16,  // kv_cache_dtype (non-fp8)
                                 k_scale,
                                 v_scale,
                                 rotary_embedding,
                                 rope_theta,
                                 static_cast<uint32_t>(0),
                                 static_cast<uint32_t>(0),
                                 static_cast<uint32_t>(0),
                                 static_cast<uint32_t>(0),
                                 static_cast<uint32_t>(0),
                                 is_neox,
                                 PositionEncoding::ROPE,
                                 std::any(cos_sin_cache_tensor.GetPtr<void>()),
                                 rope_scaling_factor,
                                 max_batch_size,
                                 attn_temperature_tuning,
                                 attn_scale,
                                 floor_scale,
                                 is_multi_token_forward,
                                 empty_mrope_section,
                                 enable_qk_pre_norm_before_rotary_pos};
  };

  // ---- Shared paged KV cache: one big K block + one big V block, aliased by both the
  //      cache-copy write path (via kv_list) and the hpc-ops read path (via layer_kv_cache_ptr). ----
  const int kv_cache_block_num = 1;
  void* k_block_ptr = nullptr;
  void* v_block_ptr = nullptr;
  Malloc(&k_block_ptr, kv_block_bytes);
  Malloc(&v_block_ptr, kv_block_bytes);
  Memset(k_block_ptr, 0, kv_block_bytes);
  Memset(v_block_ptr, 0, kv_block_bytes);

  // kv_list: [k_block, v_block] (layer_index=0, layer_block_num=1 -> k_list=[0], v_list=[1]).
  Tensor kv_list;
  CreateHalfDataTypeTensor(kv_list, {2}, GetDataType<uint64_t>());
  std::vector<void*> h_kv_list_ptrs = {k_block_ptr, v_block_ptr};
  Memcpy(kv_list.GetPtr<void>(), h_kv_list_ptrs.data(), h_kv_list_ptrs.size() * sizeof(void*), MEMCPY_HOST_TO_DEVICE);

  // layer_kv_cache_ptr: [0]=block_num, [1]=k_cache(layer0), [2]=v_cache(layer0).
  Tensor layer_kv_cache_ptr_tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {static_cast<uint64_t>(1 + 2)});
  *layer_kv_cache_ptr_tensor.GetPtr<int64_t>() = static_cast<int64_t>(kv_cache_block_num);
  void** layer_kv_cache_ptr = layer_kv_cache_ptr_tensor.GetPtr<void*>() + 1;
  layer_kv_cache_ptr[0] = k_block_ptr;
  layer_kv_cache_ptr[1] = v_block_ptr;

  // block_table: {batch, max_blocks_per_seq} = {1, 1}, all -> block 0.
  std::vector<int32_t> block_table_host = {0};
  Tensor block_table(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {1, 1});
  Memcpy(block_table.GetPtr<void>(), block_table_host.data(), block_table_host.size() * sizeof(int32_t),
         MEMCPY_HOST_TO_DEVICE);

  // ============================ Prefill (covers AttenVarlen HPC branch) ============================
  {
    const int batch = 1;
    const int total_tokens = 2;
    const int max_tokens = total_tokens;

    FlashAttentionLayer flash_attention_layer;
    EXPECT_TRUE(flash_attention_layer.Init(make_init_params(/*is_multi_token_forward=*/true), runtime_config,
                                           context_, 0).OK());

    Tensor qkv, output_tensor;
    CreateHalfDataTypeTensor(qkv, {static_cast<size_t>(total_tokens), static_cast<size_t>(qkv_cols)}, TYPE_BF16);
    CreateHalfDataTypeTensor(output_tensor, {static_cast<size_t>(total_tokens), static_cast<size_t>(out_cols)},
                             TYPE_BF16);
    Memset(qkv.GetPtr<void>(), 0, qkv.GetTotalBytes());

    // seqlen / cumulative offsets (uint64, read as int64): {0, total_tokens}.
    Tensor input_len, prefix_offsets, pos, mask;
    CreateHalfDataTypeTensor(input_len, {static_cast<size_t>(batch + 1)}, GetDataType<uint64_t>());
    CreateHalfDataTypeTensor(prefix_offsets, {static_cast<size_t>(batch + 1)}, GetDataType<int>(), sizeof(int));
    CreateHalfDataTypeTensor(pos, {static_cast<size_t>(total_tokens)}, GetDataType<uint64_t>());
    CreateHalfDataTypeTensor(mask, {static_cast<size_t>(total_tokens)}, GetDataType<uint64_t>());
    std::vector<uint64_t> input_len_cpu = {0, static_cast<uint64_t>(total_tokens)};
    Memcpy(input_len.GetPtr<void>(), input_len_cpu.data(), input_len_cpu.size() * sizeof(uint64_t),
           MEMCPY_HOST_TO_DEVICE);
    Memset(prefix_offsets.GetPtr<void>(), 0, prefix_offsets.GetTotalBytes());
    std::vector<uint64_t> pos_cpu = {0, 1};
    Memcpy(pos.GetPtr<void>(), pos_cpu.data(), pos_cpu.size() * sizeof(uint64_t), MEMCPY_HOST_TO_DEVICE);
    std::vector<uint64_t> mask_cpu = {1, 1};
    Memcpy(mask.GetPtr<void>(), mask_cpu.data(), mask_cpu.size() * sizeof(uint64_t), MEMCPY_HOST_TO_DEVICE);

    // kv_cache_offset (int) per seq: {0, 1}.
    Tensor kv_cache_offset;
    std::vector<int> kv_cache_offset_host = {0, 1};
    CreateHalfDataTypeTensor(kv_cache_offset, {static_cast<size_t>(batch + 1)}, GetDataType<int>(), sizeof(int));
    Memcpy(kv_cache_offset.GetPtr<void>(), kv_cache_offset_host.data(), kv_cache_offset_host.size() * sizeof(int),
           MEMCPY_HOST_TO_DEVICE);

    Tensor empty_flex;
    CreateHalfDataTypeTensor(empty_flex, {0}, GetDataType<int>(), sizeof(int));

    FlashAttentionLayer::CudaForwardParams flash_params;
    flash_params.qkv = &qkv;
    flash_params.token_offset = &input_len;
    flash_params.kv_list = &kv_list;
    flash_params.prefix_offset = &prefix_offsets;
    flash_params.kv_cache_offset = &kv_cache_offset;
    flash_params.rotary_embedding_pos = &pos;
    flash_params.rotary_embedding_mask = &mask;
    flash_params.src_flexible_rotary_embedding_pos = &empty_flex;
    flash_params.flexible_rotary_embedding_mask = &empty_flex;
    flash_params.dst_flexible_kv_cache = &empty_flex;
    flash_params.src_flexible_kv_cache = &empty_flex;
    flash_params.dst_flexible_token_idx = &empty_flex;
    flash_params.src_flexible_token_idx = &empty_flex;
    flash_params.flexible_offset_uint64 = &prefix_offsets;
    flash_params.layer_block_num = 1;
    flash_params.decode_token_num = 0;
    flash_params.max_forwarding_tokens = total_tokens;
    flash_params.batch_size = batch;
    flash_params.max_tokens = max_tokens;
    flash_params.query_layernorm_weight = &empty_q_norm_weight;
    flash_params.key_layernorm_weight = &empty_k_norm_weight;
    flash_params.use_cache = true;
    flash_params.kv_cache_base_ptr = &layer_kv_cache_ptr_tensor;
    flash_params.block_table = &block_table;
    flash_params.input_without_prefix_offset = &input_len;

    std::vector<Tensor> flash_outputs = {output_tensor};
    EXPECT_TRUE(flash_attention_layer.Forward(flash_params, &flash_outputs[0]).OK());
    EXPECT_EQ(flash_outputs[0].shape[0], static_cast<size_t>(total_tokens));
    EXPECT_EQ(flash_outputs[0].shape[1], static_cast<size_t>(out_cols));
  }

  // ============================ Decode (covers InvokePagedAttention HPC branch) ====================
  {
    const int batch = 1;
    const int q_seq_len = 1;  // hpc-ops decode requires q_seq_len <= 3
    const int total_tokens = batch * q_seq_len;
    const int context_len = 2;
    const int max_tokens = context_len;

    PagedAttentionLayer paged_attention_layer;
    EXPECT_TRUE(
        paged_attention_layer.Init(make_init_params(/*is_multi_token_forward=*/false), runtime_config, context_, 0)
            .OK());

    Tensor query, paged_output;
    CreateHalfDataTypeTensor(query, {static_cast<size_t>(total_tokens), static_cast<size_t>(qkv_cols)}, TYPE_BF16);
    CreateHalfDataTypeTensor(paged_output, {static_cast<size_t>(total_tokens), static_cast<size_t>(out_cols)},
                             TYPE_BF16);
    Memset(query.GetPtr<void>(), 0, query.GetTotalBytes());

    // context_lens (int32) per seq.
    Tensor context_lens;
    std::vector<int32_t> context_lens_host = {context_len};
    CreateHalfDataTypeTensor(context_lens, {static_cast<size_t>(batch)}, GetDataType<int>(), sizeof(int));
    Memcpy(context_lens.GetPtr<void>(), context_lens_host.data(), context_lens_host.size() * sizeof(int32_t),
           MEMCPY_HOST_TO_DEVICE);

    // cache_offset (int) per seq (+1 slack): {0, 1}.
    Tensor cache_offset;
    std::vector<int> cache_offset_host = {0, 1};
    CreateHalfDataTypeTensor(cache_offset, {static_cast<size_t>(batch + 1)}, GetDataType<int>(), sizeof(int));
    Memcpy(cache_offset.GetPtr<void>(), cache_offset_host.data(), cache_offset_host.size() * sizeof(int),
           MEMCPY_HOST_TO_DEVICE);

    // rotary pos / mask (uint64, read as int64) for the single decode token.
    Tensor pos, mask;
    CreateHalfDataTypeTensor(pos, {static_cast<size_t>(total_tokens)}, GetDataType<uint64_t>());
    CreateHalfDataTypeTensor(mask, {static_cast<size_t>(total_tokens)}, GetDataType<uint64_t>());
    std::vector<uint64_t> pos_cpu = {static_cast<uint64_t>(context_len)};
    Memcpy(pos.GetPtr<void>(), pos_cpu.data(), pos_cpu.size() * sizeof(uint64_t), MEMCPY_HOST_TO_DEVICE);
    std::vector<uint64_t> mask_cpu = {1};
    Memcpy(mask.GetPtr<void>(), mask_cpu.data(), mask_cpu.size() * sizeof(uint64_t), MEMCPY_HOST_TO_DEVICE);

    // workspace / qkv_workspace: unused by the HPC decode branch, but the wrapper still receives them.
    Tensor workspace, qkv_workspace;
    CreateHalfDataTypeTensor(workspace, {static_cast<size_t>(1 << 20)}, TYPE_UINT8, sizeof(uint8_t));
    CreateHalfDataTypeTensor(qkv_workspace, {static_cast<size_t>(1 << 20)}, TYPE_UINT8, sizeof(uint8_t));

    // forward_shape carries scalars in its .shape vector (see forwarding_context.cpp):
    // [3]=batch, [4]=max_tokens, [5]=layer_block_num, [12]=total_decode_tokens.
    Tensor forward_shape(MemoryLocation::LOCATION_HOST, TYPE_INT32, {1});
    forward_shape.shape = {0,
                           0,
                           0,
                           static_cast<size_t>(batch),
                           static_cast<size_t>(max_tokens),
                           static_cast<size_t>(1),
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           static_cast<size_t>(total_tokens)};

    PagedAttentionInputs inputs;
    inputs.query = &query;
    inputs.context_lens = &context_lens;
    inputs.kv_list = &kv_list;
    inputs.cache_offset = &cache_offset;
    inputs.rotary_embedding_pos = &pos;
    inputs.rotary_embedding_mask = &mask;
    inputs.workspace = &workspace;
    inputs.forward_shape = &forward_shape;
    inputs.qkv_workspace = &qkv_workspace;
    inputs.query_layernorm_weight = &empty_q_norm_weight;
    inputs.key_layernorm_weight = &empty_k_norm_weight;
    inputs.layer_kv_cache_ptr = &layer_kv_cache_ptr_tensor;
    inputs.block_table = &block_table;

    std::vector<Tensor> paged_outputs = {paged_output};
    EXPECT_TRUE(paged_attention_layer.Forward(inputs, paged_outputs).OK());
    EXPECT_EQ(paged_outputs[0].shape[0], static_cast<size_t>(total_tokens));
    EXPECT_EQ(paged_outputs[0].shape[1], static_cast<size_t>(out_cols));
  }

  // ---- restore Environment config so sibling tests are not affected ----
  env->SetAttnBackendConfig(original_attn_cfg);

  Free(k_block_ptr);
  Free(v_block_ptr);
#endif
}

TEST_F(LayerTest, FlashInferResourceManagerTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank0 = 0;
  size_t num_heads = 32;
  size_t num_kv_heads = 4;
  size_t head_dim = 128;

  // Test 1: Get pinned host workspace for rank 0 (first time initialization)
  void* pinned_workspace_rank0 =
      FlashInferResourceManager::GetPinnedHostWorkspace(num_heads, num_kv_heads, head_dim, kDeviceRank0);
  EXPECT_NE(pinned_workspace_rank0, nullptr);
  KLLM_LOG_INFO << "Test 1 passed: Pinned host workspace initialized for rank 0";

  // Test 2: Get pinned host workspace again for rank 0 (should return same pointer)
  void* pinned_workspace_rank0_again =
      FlashInferResourceManager::GetPinnedHostWorkspace(num_heads, num_kv_heads, head_dim, kDeviceRank0);
  EXPECT_EQ(pinned_workspace_rank0, pinned_workspace_rank0_again);
  KLLM_LOG_INFO << "Test 2 passed: Pinned host workspace reused for rank 0";

  // Test 3: Get device workspace for rank 0
  std::shared_ptr<Tensor>& device_workspace_rank0 =
      FlashInferResourceManager::GetDeviceWorkspace(num_heads, num_kv_heads, head_dim, kDeviceRank0);
  EXPECT_NE(device_workspace_rank0, nullptr);
  EXPECT_GT(device_workspace_rank0->GetTotalBytes(), 0);
  EXPECT_EQ(device_workspace_rank0->location, MemoryLocation::LOCATION_DEVICE);
  EXPECT_EQ(device_workspace_rank0->dtype, DataType::TYPE_INT8);
  KLLM_LOG_INFO << "Test 3 passed: Device workspace initialized for rank 0, size: "
                << device_workspace_rank0->GetTotalBytes() << " bytes";

  // Test 4: Get device workspace again for rank 0 (should return same reference)
  std::shared_ptr<Tensor>& device_workspace_rank0_again =
      FlashInferResourceManager::GetDeviceWorkspace(num_heads, num_kv_heads, head_dim, kDeviceRank0);
  EXPECT_EQ(device_workspace_rank0.get(), device_workspace_rank0_again.get());
  KLLM_LOG_INFO << "Test 4 passed: Device workspace reused for rank 0";

  // Test 5: Get prefill helper before initialization (should return nullptr)
  std::shared_ptr<void> helper_before_init = FlashInferResourceManager::GetPrefillHelper(kDeviceRank0);
  EXPECT_EQ(helper_before_init, nullptr);
  KLLM_LOG_INFO << "Test 5 passed: GetPrefillHelper returns nullptr before initialization";

  // Test 6: Set prefill helper for rank 0
  auto mock_helper_rank0 = std::make_shared<int>(42);  // Mock helper object
  FlashInferResourceManager::SetPrefillHelper(kDeviceRank0, mock_helper_rank0);
  KLLM_LOG_INFO << "Test 6 passed: Prefill helper set for rank 0";

  // Test 7: Get prefill helper after initialization
  std::shared_ptr<void> helper_rank0_after = FlashInferResourceManager::GetPrefillHelper(kDeviceRank0);
  EXPECT_NE(helper_rank0_after, nullptr);
  EXPECT_EQ(helper_rank0_after, mock_helper_rank0);
  KLLM_LOG_INFO << "Test 7 passed: GetPrefillHelper returns correct helper after initialization";
  KLLM_LOG_INFO << "All FlashInferResourceManager tests passed successfully! ";
  FlashInferResourceManager::FreeAllResources();
#else
  GTEST_SKIP() << "FlashInferResourceManager requires CUDA support";
#endif
}

}  // namespace ksana_llm
