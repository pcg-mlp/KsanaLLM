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
  explicit GrammarGeneratorCreator(const VocabInfo& vocab_info,
                                   StructuredConstraintType constraint_type = StructuredConstraintType::JSON)
      : grammar_backend_(GrammarBackend::Create(vocab_info)), constraint_type_(constraint_type) {
    // Create() already validates/logs the failure reason.
    // Fail fast here instead of deferring to CreateGenerator().
    if (!grammar_backend_) {
      throw std::runtime_error("Failed to create grammar backend");
    }
  }

  GrammarGeneratorCreator(const std::shared_ptr<GrammarBackend>& grammar_backend,
                          StructuredConstraintType constraint_type)
      : grammar_backend_(grammar_backend), constraint_type_(constraint_type) {}

  std::shared_ptr<StructuredGeneratorInterface> CreateGenerator(const StructuredGeneratorConfig& config) override {
    if (!grammar_backend_) {
      throw std::runtime_error("Grammar backend is not initialized");
    }

    std::shared_ptr<CompiledGrammar> compiled_grammar;
    switch (constraint_type_) {
      case StructuredConstraintType::JSON:
        compiled_grammar = grammar_backend_->CompileJSONSchema(config.constraint_spec);
        break;
      case StructuredConstraintType::REGEX:
        compiled_grammar = grammar_backend_->CompileRegex(config.constraint_spec);
        break;
      case StructuredConstraintType::GRAMMAR:
        compiled_grammar = grammar_backend_->CompileGrammar(config.constraint_spec);
        break;
      case StructuredConstraintType::STRUCTURAL_TAG:
        compiled_grammar = grammar_backend_->CompileStructuralTag(config.constraint_spec);
        break;
      default:
        throw std::runtime_error("Unsupported XGrammar constraint type");
    }
    if (!compiled_grammar) {
      throw std::runtime_error("Failed to compile grammar from constraint specification");
    }

    return std::make_shared<GrammarStructuredGenerator>(compiled_grammar, constraint_type_);
  }

  StructuredConstraintType GetConstraintType() const override { return constraint_type_; }

 private:
  std::shared_ptr<GrammarBackend> grammar_backend_;
  StructuredConstraintType constraint_type_;
};

}  // namespace ksana_llm
