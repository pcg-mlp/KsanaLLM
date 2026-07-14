/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/structured_generation/structured_generator_interface.h"
#include "configure/environment.h"

namespace ksana_llm {

/*!
 * \\brief Abstract base class for generator creators
 *
 * Each Creator is responsible for creating generators for a specific constraint type.
 */
class GeneratorCreator {
 public:
  virtual ~GeneratorCreator() = default;
  virtual std::shared_ptr<StructuredGeneratorInterface> CreateGenerator(const StructuredGeneratorConfig& config) = 0;
  virtual StructuredConstraintType GetConstraintType() const = 0;
};
/*!
 * \\brief Factory class for creating structured generators.
 *
 * This factory creates appropriate structured generator instances based on
 * the constraint type and configuration using registered creator objects.
 */
class StructuredGeneratorFactory {
 public:
  StructuredGeneratorFactory() {}
  explicit StructuredGeneratorFactory(const VocabInfo& vocab_info) : vocab_info_(vocab_info) {}

  std::shared_ptr<StructuredGeneratorInterface> CreateGenerator(const StructuredGeneratorConfig& config,
                                                                const bool enable_thinking = true);

  bool IsConstraintTypeSupported(StructuredConstraintType constraint_type);

  std::vector<StructuredConstraintType> GetSupportedConstraintTypes();

  void RegisterCreator(StructuredConstraintType constraint_type, std::unique_ptr<GeneratorCreator> creator);

  // Set vocab info (including think_end_token_id for reasoning mode)
  void SetVocabInfo(const VocabInfo& vocab_info) { vocab_info_ = vocab_info; }

  // Get vocab info
  const VocabInfo& GetVocabInfo() const { return vocab_info_; }

 private:
  void InitializeRegistry();

  // Check if reasoning mode is enabled
  bool IsReasoningModeEnabled() const { return vocab_info_.think_end_token_id >= 0; }

  std::unordered_map<StructuredConstraintType, std::unique_ptr<GeneratorCreator>> creator_registry_;
  std::mutex registry_mutex_;

  // Vocab info (including think_end_token_id for reasoning mode)
  VocabInfo vocab_info_;
};

}  // namespace ksana_llm
