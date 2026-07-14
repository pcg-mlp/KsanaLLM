/* Copyright 2025 Tencent Inc.  All rights reserved.
 
==============================================================================*/
#pragma once

#include "runtime/structured_generation/structured_generator_factory.h"
#include "runtime/structured_generation/xgrammar/xgrammar_structured_generator.h"
#include "utils/grammar_backend.h"

namespace ksana_llm {

/*!
 * \\brief Creator for grammar constraint generators.
 */
class GrammarGeneratorCreator : public GeneratorCreator {
 public:
  // Constructor accepting VocabInfo reference
  explicit GrammarGeneratorCreator(const VocabInfo& vocab_info) {
    if (!vocab_info.initialized) {
      throw std::runtime_error("VocabInfo not initialized");
    }
    try {
      grammar_backend_ = GrammarBackend::Create(vocab_info);
    } catch (const std::exception& e) {
      throw std::runtime_error("Grammar backend creation failed: " + std::string(e.what()));
    }
  }

  std::shared_ptr<StructuredGeneratorInterface> CreateGenerator(const StructuredGeneratorConfig& config) override {
    if (!grammar_backend_) {
      throw std::runtime_error("Grammar backend is not initialized");
    }

    // Compile the grammar from the constraint specification
    auto compiled_grammar = grammar_backend_->CompileJSONSchema(config.constraint_spec);
    if (!compiled_grammar) {
      throw std::runtime_error("Failed to compile grammar from constraint specification");
    }

    return std::make_shared<GrammarStructuredGenerator>(compiled_grammar);
  }

  StructuredConstraintType GetConstraintType() const override { return StructuredConstraintType::JSON; }

 private:
  std::shared_ptr<GrammarBackend> grammar_backend_;
};

}  // namespace ksana_llm