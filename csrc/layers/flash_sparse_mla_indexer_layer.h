/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "configure/model_config_parser.h"  // For RoPEScalingFactor
#include "layers/base_layer.h"

#ifdef ENABLE_CUDA
#  include <optional>
#  include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#endif

namespace ksana_llm {

class FlashSparseMlaIndexerLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;
  struct FlashSparseMlaIndexerParams {
    const Tensor* q;
    const Tensor* k;
    const Tensor* weights;
    const Tensor* quant_workspace;
    const Tensor* rotary_embedding_pos;
    const Tensor* store_out_loc;
    const Tensor* seq_len_offset;
    const Tensor* block_table;
    const Tensor* cu_seq_len_start;
    const Tensor* cu_seq_len_end;
    const Tensor* batch_indices;
    const Tensor* layer_kv_cache_ptr;
    const Tensor* chunk_token_starts_host;
    const Tensor* chunk_kv_token_ranges_host;
    // Scalars formerly encoded in forward_shape (input[14]):
    int batch_size;           // shape[7]
    int total_prefix_tokens;  // shape[11]
  };
  Status Forward(const FlashSparseMlaIndexerParams& params, Tensor* output, const LayerExecContext& exec_ctx = {});

  // Get workspace size needed for this layer
  virtual size_t GetWorkspaceSize() override;

 private:
  template <typename SCALAR_T>
  Status ForwardT(const FlashSparseMlaIndexerParams& p, Tensor* output);

 private:
  // Model parameters
  int n_heads_;                   // number of index heads
  int head_dim_;                  // index head dimension
  int rope_head_dim_;             // rope head dimension
  int index_topk_;                // top-k value
  int block_size_;                // kv block size: number of tokens in one block
  int page_bits_;                 // log2(block_size)
  float n_heads_inv_sqrt_;        // n_heads ** -0.5
  float softmax_scale_;           // head_dim ** -0.5
  int max_seq_len_;               // max sequence length
  int layer_index_;               // layer index
  int layer_num_;                 // number of layers on this node
  size_t max_step_token_num_;     // max step token num from runtime_config
  size_t max_chunked_q_tokens_;   // max Q tokens per logits chunk
  size_t max_chunked_kv_tokens_;  // max KV tokens per chunk
  bool scale_ue8m0_;              // use ue8m0 format for fp8 scales
  bool is_neox_;                  // true: NeoX style rotary, false: GPT-J interleaved style
  size_t compress_ratio_;         // for indexer using compressor
  int block_stride_;              // stride of kv block

  std::unique_ptr<BaseLayer> topk_layer_;  // shared topk dispatch logic

#ifdef ENABLE_CUDA
  // RoPE embedding
  std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda> rotary_embedding_cuda_;
#endif
};

}  // namespace ksana_llm
