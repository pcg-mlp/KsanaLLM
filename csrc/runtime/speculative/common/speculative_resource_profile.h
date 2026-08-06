/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/speculative/common/speculative_decoder_interface.h"

namespace ksana_llm {

// Forward declaration
struct BatchSchedulerConfig;
class CompositeSpeculativeDecoder;
class Context;
class DraftGeneratorInterface;
class ExecutorRuntime;
class ModelInstance;

enum class BufferShapeExpr {
  // Speculative hidden buffer: shape = max_token_num * hidden_units * mhc_mult
  kSpecHiddenBuffer,
};

struct BufferDecl {
  std::string name;
  BufferShapeExpr shape_expr;
};

struct SpeculativeResourceProfile {
  size_t extra_decode_tokens_per_req = 0;
  size_t workload_multiplier = 0;
  size_t extra_token_capacity = 0;
  // Extra KV tokens needed for lookahead (MTP/DSpark: step_num/block_size, PTP: 0).
  // Decoupled from workload_multiplier so PTP (which has multiplier=2*step_num) does
  // not over-reserve KV cache slots.
  size_t extra_kv_tokens = 0;
  std::vector<BufferDecl> buffer_decls;
};

// Estimator function signature: takes BatchSchedulerConfig, returns profile.
// Estimates are called in Parser phase, before model config is fully available.
using SpeculativeResourceEstimator = std::function<SpeculativeResourceProfile(const BatchSchedulerConfig&)>;

// Factory result: adapter + optional draft generator, both owned by caller.
// generator may be nullptr for decoders that don't produce draft tokens.
struct DecoderFactoryResult {
  std::unique_ptr<SpeculativeDecoderInterface> adapter;
  std::shared_ptr<DraftGeneratorInterface> generator;
};

// Factory function: takes BatchSchedulerConfig, returns adapter + generator.
// Called during service init (single-threaded).
using DecoderFactory = std::function<DecoderFactoryResult(const BatchSchedulerConfig&)>;

// Context passed to post-init setup callbacks.
struct SpeculativeSetupContext {
  std::shared_ptr<CompositeSpeculativeDecoder> composite_decoder;
  std::shared_ptr<ModelInstance> model_instance;
  std::shared_ptr<Context> context;
  ExecutorRuntime* executor_runtime = nullptr;
};

// Post-init setup function: registers post-initialization work (e.g. lazy decoder
// creation) via ExecutorRuntime::RegisterPostInitCallback. Executed synchronously
// during service init, before warmup.
using PostInitSetup = std::function<void(const SpeculativeSetupContext&)>;

// Reverse static registry for speculative decoding resource estimation.
// Speculative decoder adapters register their estimator functions via a
// lambda in an anonymous namespace (main-before-init). The config parser
// calls Aggregate() to collect resource requirements without knowing the
// specific speculative decoding method.
//
// Thread safety: registration occurs during static init (single-threaded).
// Aggregate() is called during service init (single-threaded).
class SpeculativeRegistry {
 public:
  static SpeculativeRegistry& Get();
  void Register(SpeculativeResourceEstimator estimator);
  SpeculativeResourceProfile Aggregate(const BatchSchedulerConfig& cfg) const;

  // Decoder factory registration via reverse static registry.
  // Each adapter registers during static init. inference_executor calls
  // CreateDecoder to instantiate the correct adapter without knowing the
  // specific decoder type.
  void RegisterDecoder(const std::string& method, DecoderFactory factory);
  DecoderFactoryResult CreateDecoder(const std::string& method, const BatchSchedulerConfig& cfg) const;
  bool HasDecoderMethod(const std::string& method) const;

  // Register a post-init setup callback. Decoders register during static init
  // (main-before-init). InferenceExecutor calls RunPostInitSetups to execute all
  // registered setups, which in turn register their creation logic with
  // ExecutorRuntime::RegisterPostInitCallback.
  void RegisterPostInitSetup(PostInitSetup setup);

  // Execute all registered post-init setup callbacks. Each setup receives the
  // SpeculativeSetupContext and registers lazy-creation work with ExecutorRuntime.
  void RunPostInitSetups(const SpeculativeSetupContext& ctx) const;

 private:
  std::vector<SpeculativeResourceEstimator> estimators_;
  std::unordered_map<std::string, DecoderFactory> decoder_factories_;
  std::vector<PostInitSetup> post_init_setups_;
};

}  // namespace ksana_llm
