/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "utils/grammar_backend.h"
#include "utils/logger.h"
#include "utils/status.h"
#include "xgrammar/xgrammar.h"

namespace ksana_llm {

class GrammarBackendNvidia : public GrammarBackend {
 public:
  explicit GrammarBackendNvidia(const VocabInfo& vocab_info);
  ~GrammarBackendNvidia() override;

  // Implement pure virtual methods from base class
  std::shared_ptr<CompiledGrammar> CompileJSONSchema(const std::string& schema) override;
  std::shared_ptr<CompiledGrammar> CompileRegex(const std::string& regex) override;
  std::shared_ptr<CompiledGrammar> CompileGrammar(const std::string& grammar) override;
  std::shared_ptr<CompiledGrammar> CompileStructuralTag(const std::string& structural_tag) override;
  std::shared_ptr<GrammarMatcherWrapper> CreateMatcher(std::shared_ptr<CompiledGrammar> grammar) override;
  const xgrammar::TokenizerInfo& GetTokenizerInfo() const override;
  bool IsInitialized() const override { return initialized_; }

 private:
  std::unique_ptr<xgrammar::TokenizerInfo> tokenizer_info_;
  std::unique_ptr<xgrammar::GrammarCompiler> compiler_;
  bool initialized_ = false;
};

}  // namespace ksana_llm
