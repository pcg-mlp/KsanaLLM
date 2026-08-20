/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// Stub implementations for layers not yet supported on Kunlun backend.
// These prevent undefined vtable symbols when libtorch_serving.so is loaded.

#include "layers/add_mul_layer.h"
#include "layers/all_reduce_residual_add_norm_layer.h"
#include "layers/attn_res_layer.h"
#include "layers/compressor_layer.h"
#include "layers/concat_layer.h"
#include "layers/dspark_layer.h"
#include "layers/flash_dsv4_attention_layer.h"
#include "layers/flash_sparse_mla_attention_layer.h"
#include "layers/flash_sparse_mla_indexer_layer.h"
#include "layers/mem_adjuster_layer.h"
#include "layers/mhc_layer.h"
#include "layers/mla_output_gate_layer.h"
#include "layers/paged_dsv4_attention_layer.h"
#include "layers/paged_sparse_mla_attention_layer.h"
#include "layers/paged_sparse_mla_indexer_layer.h"
#include "layers/situ_mul_layer.h"

namespace ksana_llm {

// FlashSparseMlaAttentionLayer
Status FlashSparseMlaAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                          std::shared_ptr<Context> context, int rank) {
  return AttentionLayer::Init(parameters, runtime_config, context, rank);
}
Status FlashSparseMlaAttentionLayer::Forward(const FlashSparseMlaAttentionInputs&, Tensor*) {
  KLLM_THROW("FlashSparseMlaAttentionLayer not supported on Kunlun.");
}
size_t FlashSparseMlaAttentionLayer::GetWorkspaceSize() { return 0; }

// FlashSparseMlaIndexerLayer
Status FlashSparseMlaIndexerLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                        std::shared_ptr<Context> context, int rank) {
  return BaseLayer::Init(parameters, runtime_config, context, rank);
}
Status FlashSparseMlaIndexerLayer::Forward(const FlashSparseMlaIndexerParams&, Tensor*, const LayerExecContext&) {
  KLLM_THROW("FlashSparseMlaIndexerLayer not supported on Kunlun.");
}
size_t FlashSparseMlaIndexerLayer::GetWorkspaceSize() { return 0; }

// PagedSparseMlaAttentionLayer
Status PagedSparseMlaAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                          std::shared_ptr<Context> context, int rank) {
  return AttentionLayer::Init(parameters, runtime_config, context, rank);
}
Status PagedSparseMlaAttentionLayer::Forward(const PagedSparseMlaAttentionInputs&, Tensor*) {
  KLLM_THROW("PagedSparseMlaAttentionLayer not supported on Kunlun.");
}

// PagedSparseMlaIndexerLayer
Status PagedSparseMlaIndexerLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                        std::shared_ptr<Context> context, int rank) {
  return BaseLayer::Init(parameters, runtime_config, context, rank);
}
Status PagedSparseMlaIndexerLayer::Forward(const PagedSparseMlaIndexerParams&, Tensor*, const LayerExecContext&) {
  KLLM_THROW("PagedSparseMlaIndexerLayer not supported on Kunlun.");
}
size_t PagedSparseMlaIndexerLayer::GetWorkspaceSize() { return 0; }

// AttnResLayer
Status AttnResLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                          std::shared_ptr<Context> context, int rank) {
  return BaseLayer::Init(parameters, runtime_config, context, rank);
}
Status AttnResLayer::Forward(const Tensor*, const Tensor*, const Tensor*, const Tensor*, const Tensor*, const Tensor*,
                             Tensor*, bool, const LayerExecContext&) {
  KLLM_THROW("AttnResLayer not supported on Kunlun.");
}

// AddMulLayer
Status AddMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                         std::shared_ptr<Context> context, int rank) {
  return BaseLayer::Init(parameters, runtime_config, context, rank);
}
Status AddMulLayer::Forward(const Tensor*, const Tensor*, Tensor*, const LayerExecContext&) {
  KLLM_THROW("AddMulLayer not supported on Kunlun.");
}

// AllReduceResidualAddNormLayer
Status AllReduceResidualAddNormLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                           std::shared_ptr<Context> context, int rank) {
  return BaseLayer::Init(parameters, runtime_config, context, rank);
}
Status AllReduceResidualAddNormLayer::Forward(const Tensor*, Tensor*, const LayerExecContext&) {
  KLLM_THROW("AllReduceResidualAddNormLayer not supported on Kunlun.");
}
Status AllReduceResidualAddNormLayer::Forward(const Tensor*, const Tensor*, Tensor*, const LayerExecContext&) {
  KLLM_THROW("AllReduceResidualAddNormLayer not supported on Kunlun.");
}

// ConcatLayer
Status ConcatLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                         std::shared_ptr<Context> context, int rank) {
  return BaseLayer::Init(parameters, runtime_config, context, rank);
}
Status ConcatLayer::Forward(const Tensor*, const Tensor*, Tensor*, const LayerExecContext&) {
  KLLM_THROW("ConcatLayer not supported on Kunlun.");
}

// MemAdjusterLayer
Status MemAdjusterLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                              std::shared_ptr<Context> context, int rank) {
  return BaseLayer::Init(parameters, runtime_config, context, rank);
}
Status MemAdjusterLayer::ExtractSubMatrix(const Tensor& input_tensor, Tensor& output_tensor, size_t input_offset,
                                          size_t output_n) {
  KLLM_THROW("MemAdjusterLayer::ExtractSubMatrix not supported on Kunlun.");
}

// FlashDsv4AttentionLayer
Status FlashDsv4AttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                     std::shared_ptr<Context> context, int rank) {
  return AttentionLayer::Init(parameters, runtime_config, context, rank);
}
Status FlashDsv4AttentionLayer::Forward(const Dsv4AttentionInputs&, Tensor*) {
  KLLM_THROW("FlashDsv4AttentionLayer not supported on Kunlun.");
}
size_t FlashDsv4AttentionLayer::GetWorkspaceSize() { return 0; }

// PagedDsv4AttentionLayer
Status PagedDsv4AttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                     std::shared_ptr<Context> context, int rank) {
  return AttentionLayer::Init(parameters, runtime_config, context, rank);
}
Status PagedDsv4AttentionLayer::Forward(const Dsv4AttentionInputs&, Tensor*) {
  KLLM_THROW("PagedDsv4AttentionLayer not supported on Kunlun.");
}

// CompressorLayer
Status CompressorLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                             std::shared_ptr<Context> context, int rank) {
  return BaseLayer::Init(parameters, runtime_config, context, rank);
}
Status CompressorLayer::ForwardPrefillC4(const Tensor*, const Tensor*, const Tensor*, const Tensor*, const Tensor*,
                                         Tensor*, Tensor*, const Tensor*, const Tensor*, const Tensor*, const Tensor*,
                                         const Tensor*, Tensor*, bool) {
  KLLM_THROW("CompressorLayer not supported on Kunlun.");
}
Status CompressorLayer::ForwardPrefillC128(const Tensor*, const Tensor*, const Tensor*, const Tensor*, const Tensor*,
                                           Tensor*, Tensor*, const Tensor*, const Tensor*, const Tensor*, const Tensor*,
                                           const Tensor*, Tensor*, bool) {
  KLLM_THROW("CompressorLayer not supported on Kunlun.");
}
Status CompressorLayer::ForwardDecodeC4(const Tensor*, const Tensor*, const Tensor*, const Tensor*, const Tensor*,
                                        const Tensor*, const Tensor*, const Tensor*, const Tensor*, Tensor*) {
  KLLM_THROW("CompressorLayer not supported on Kunlun.");
}
Status CompressorLayer::ForwardDecodeC128(const Tensor*, const Tensor*, const Tensor*, const Tensor*, const Tensor*,
                                          const Tensor*, const Tensor*, const Tensor*, Tensor*) {
  KLLM_THROW("CompressorLayer not supported on Kunlun.");
}

// DSparkLayer
Status DSparkLayer::ForwardCaptureMeanReduce(void*, const void*, int, int, int, int64_t, DataType) {
  KLLM_THROW("DSparkLayer not supported on Kunlun.");
}
Status DSparkLayer::ForwardInjectKV(const void*, const void*, void*, const int64_t*, const void*, const int32_t*, float,
                                    int, int64_t, int, DataType) {
  KLLM_THROW("DSparkLayer not supported on Kunlun.");
}
Status DSparkLayer::ForwardMarkovEmbedLookup(void*, const uint32_t*, const void*, int, int, DataType) {
  KLLM_THROW("DSparkLayer not supported on Kunlun.");
}
Status DSparkLayer::ForwardLogitsAdd(float*, const void*, int, int, DataType) {
  KLLM_THROW("DSparkLayer not supported on Kunlun.");
}

// MhcLayer
Status MhcLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) {
  return BaseLayer::Init(parameters, runtime_config, context, rank);
}
Status MhcLayer::ForwardPre(const Tensor*, const Tensor*, const Tensor*, const Tensor*, Tensor*, Tensor*, Tensor*,
                            Tensor*, Tensor*, const LayerExecContext&) {
  KLLM_THROW("MhcLayer not supported on Kunlun.");
}
Status MhcLayer::ForwardPreWithNorm(const Tensor*, const Tensor*, const Tensor*, const Tensor*, const Tensor*, Tensor*,
                                    Tensor*, Tensor*, Tensor*, Tensor*, const LayerExecContext&) {
  KLLM_THROW("MhcLayer not supported on Kunlun.");
}
Status MhcLayer::ForwardPost(const Tensor*, const Tensor*, const Tensor*, Tensor*, const LayerExecContext&) {
  KLLM_THROW("MhcLayer not supported on Kunlun.");
}
Status MhcLayer::ForwardInitial(const Tensor*, Tensor*, const LayerExecContext&) {
  KLLM_THROW("MhcLayer not supported on Kunlun.");
}
Status MhcLayer::ForwardFinal(const Tensor*, const Tensor*, const Tensor*, const Tensor*, Tensor*, Tensor*, Tensor*,
                              const LayerExecContext&) {
  KLLM_THROW("MhcLayer not supported on Kunlun.");
}

// MlaOutputGateLayer
Status MlaOutputGateLayer::Forward(const Tensor*, const Tensor*, Tensor*, const LayerExecContext&) {
  KLLM_THROW("MlaOutputGateLayer not supported on Kunlun.");
}

// SituMulLayer
Status SituMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                          std::shared_ptr<Context> context, int rank) {
  return BaseLayer::Init(parameters, runtime_config, context, rank);
}
Status SituMulLayer::Forward(const Tensor*, Tensor*, const LayerExecContext&) {
  KLLM_THROW("SituMulLayer not supported on Kunlun.");
}

}  // namespace ksana_llm
