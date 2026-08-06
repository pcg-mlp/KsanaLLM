/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "runtime/speculative/common/spec_scheduling_info.h"
#include "utils/status.h"

namespace ksana_llm {

// Forward declarations
class ExecutorInferRequest;
class ForwardingContext;
struct DraftTokens;
class HiddenUnitDeviceBuffer;
class ScheduleOutput;

// Typed context for speculative decoder lifecycle methods.
// Each decoder type uses a dedicated factory method that constructs a context
// with the fields it actually needs, enforcing compile-time correctness.
struct SpecContext {
  // Common fields — used by ALL decoders.
  std::vector<std::shared_ptr<ExecutorInferRequest>>* reqs = nullptr;

  // --- Factory methods ---

  // For draft-generation-only decoders (PTP, Trie, ASR-custom).
  // These decoders only need reqs and an output field for draft tokens.
  static SpecContext ForDraftGen(std::vector<std::shared_ptr<ExecutorInferRequest>>* r, DraftTokens* out) {
    SpecContext ctx;
    ctx.reqs = r;
    ctx.output = out;
    return ctx;
  }

  // For MTP — requires hidden_unit, schedule_output, and forwarding_context
  // for gather index injection and hidden buffer management.
  static SpecContext ForMtp(std::vector<std::shared_ptr<ExecutorInferRequest>>* r, HiddenUnitDeviceBuffer* hu,
                            std::shared_ptr<ScheduleOutput> so, ForwardingContext* fwd) {
    return ForModelDecoder(r, hu, std::move(so), fwd);
  }

  // For DSpark — requires hidden_unit and schedule_output, but NOT
  // forwarding_context (DSpark does not use the gather index builder or
  // hidden buffer override mechanism).
  static SpecContext ForDSpark(std::vector<std::shared_ptr<ExecutorInferRequest>>* r, HiddenUnitDeviceBuffer* hu,
                               std::shared_ptr<ScheduleOutput> so) {
    return ForModelDecoder(r, hu, std::move(so), nullptr);
  }

  // For Verify-only context (used in CompositeSpeculativeDecoder::VerifyDraftTokens).
  static SpecContext ForVerify(std::vector<std::shared_ptr<ExecutorInferRequest>>* r) {
    SpecContext ctx;
    ctx.reqs = r;
    return ctx;
  }

  // --- Fields ---
  // Draft tokens produced by this decoder (used by PTP/Trie/ASR).
  DraftTokens* output = nullptr;

  // Hidden-unit buffer for draft-model decoders (MTP/DSpark).
  HiddenUnitDeviceBuffer* hidden_unit = nullptr;

  // Schedule output shared with the main forward (MTP/DSpark).
  std::shared_ptr<ScheduleOutput> schedule_output;

  // ForwardingContext for PreMainForward hook and MTP gather index management.
  // Only set for MTP context; nullptr for all other decoders.
  ForwardingContext* forwarding_context = nullptr;

 private:
  // Common init for draft-model decoders (MTP/DSpark).
  static SpecContext ForModelDecoder(std::vector<std::shared_ptr<ExecutorInferRequest>>* r, HiddenUnitDeviceBuffer* hu,
                                     std::shared_ptr<ScheduleOutput> so, ForwardingContext* fwd) {
    SpecContext ctx;
    ctx.reqs = r;
    ctx.output = nullptr;
    ctx.hidden_unit = hu;
    ctx.schedule_output = std::move(so);
    ctx.forwarding_context = fwd;
    return ctx;
  }
};

// Unified interface for all speculative decoding methods.
// Each speculative decoding type (MTP, PTP, Trie, ASR-custom, draft models)
// implements this interface. The main inference flow only interacts with
// this interface — never with specific speculative method implementations.
class SpeculativeDecoderInterface {
 public:
  virtual ~SpeculativeDecoderInterface() = default;

  // Human-readable name for logging, profiling, and metrics.
  virtual const char* GetName() const = 0;

  // Return scheduling resource requirements.
  virtual SpecSchedulingInfo GetSchedulingInfo() const = 0;

  // Called before the forward pass.
  // - No-draft-model methods: generate draft tokens (e.g. fill placeholder tokens).
  // - Draft-model methods: save request state, prepare for forward.
  virtual void BeforeForward(SpecContext& ctx) = 0;

  // Execute the forward pass for draft token generation.
  // - No-draft-model methods: no-op.
  // - Draft-model methods: execute DraftModelInterface::Forward().
  virtual void ExecuteForward(SpecContext& ctx) = 0;

  // Called after the forward pass.
  // - No-draft-model methods: no-op.
  // - Draft-model methods: restore request state, notify synchronization.
  virtual void AfterForward(SpecContext& ctx) = 0;

  // Verify draft tokens against sampling results.
  // Sets accepted_tokens and generated_tokens on each request.
  virtual void Verify(SpecContext& ctx) = 0;

  // Whether this decoder can be combined with another.
  virtual bool IsCombinableWith(const SpeculativeDecoderInterface* other) const = 0;

  // --- Lifecycle hooks (default empty bodies for backward compatibility) ---

  // Called before the main model forward in ExecutorRuntime::Forward().
  // - MtpDecoderAdapter: serialization Wait->Reset, lazy creation, hidden buffer injection.
  // - All other adapters: no-op.
  virtual void PreMainForward(SpecContext& ctx) {}

  // --- Shutdown hooks (default empty bodies for backward compatibility) ---

  // Called during ExecutorRuntime::Shutdown() to wake blocked threads.
  // - MtpDecoderAdapter: wake prev_mtp_done_ Waiter.
  // - All other adapters: no-op.
  virtual void Shutdown() {}
};

}  // namespace ksana_llm
