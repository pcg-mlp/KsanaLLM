/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "utils/nvidia/grammar_backend_nvidia.h"
#include <pybind11/embed.h>
#include "configure/environment.h"
#include "utils/grammar_matcher.h"
#include "utils/singleton.h"
#include "utils/tokenizer.h"

namespace ksana_llm {

// Grammar compiler configuration constants
constexpr int kDefaultMaxThreads = 8;
constexpr bool kDefaultCacheEnabled = true;
constexpr int kDefaultMaxMemoryBytes = -1;  // unlimited

GrammarBackendNvidia::GrammarBackendNvidia(const VocabInfo& vocab_info) {
  KLLM_LOG_DEBUG << "GrammarBackendNvidia constructor started, vocab_size: " << vocab_info.vocab_size;

  if (!vocab_info.initialized) {
    throw std::runtime_error("VocabInfo not initialized");
  }

  // Get tokenizer type from VocabInfo
  int vocab_type = vocab_info.vocab_type;
  bool add_prefix_space = vocab_info.add_prefix_space;

  KLLM_LOG_INFO << "Using tokenizer type from VocabInfo: vocab_type=" << vocab_type
                << ", add_prefix_space=" << add_prefix_space;

  // Convert stop_token_ids to int32_t
  std::vector<int32_t> stop_tokens_int32(vocab_info.stop_token_ids.begin(), vocab_info.stop_token_ids.end());
  xgrammar::VocabType xgrammar_vocab_type = static_cast<xgrammar::VocabType>(vocab_type);

  // Create TokenizerInfo with VocabInfo parameters
  tokenizer_info_ = std::make_unique<xgrammar::TokenizerInfo>(vocab_info.vocab,       // encoded_vocab
                                                              xgrammar_vocab_type,    // vocab_type
                                                              vocab_info.vocab_size,  // vocab_size
                                                              stop_tokens_int32,      // stop_token_ids
                                                              add_prefix_space);      // add_prefix_space

  // Create GrammarCompiler
  compiler_ = std::make_unique<xgrammar::GrammarCompiler>(*tokenizer_info_,         // tokenizer_info
                                                          kDefaultMaxThreads,       // max_threads
                                                          kDefaultCacheEnabled,     // cache_enabled
                                                          kDefaultMaxMemoryBytes);  // max_memory_bytes (unlimited)

  KLLM_LOG_DEBUG << "xgrammar::GrammarCompiler created successfully";
  initialized_ = true;
}

GrammarBackendNvidia::~GrammarBackendNvidia() {}

std::shared_ptr<CompiledGrammar> GrammarBackendNvidia::CompileJSONSchema(const std::string& schema) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto compiled_grammar = std::make_shared<CompiledGrammar>(compiler_->CompileJSONSchema(schema,  // schema
                                                                                         true,    // any_whitespace
                                                                                         std::nullopt,  // indent
                                                                                         std::nullopt,  // separators
                                                                                         true));        // strict_mode

  KLLM_LOG_DEBUG << "JSON schema compilation completed successfully";
  KLLM_LOG_DEBUG << "Compiled grammar memory usage: " << compiled_grammar->MemorySizeBytes()
                 << " bytes (strict_mode=true, any_whitespace=true)";
  return compiled_grammar;
}

std::shared_ptr<GrammarMatcherWrapper> GrammarBackendNvidia::CreateMatcher(std::shared_ptr<CompiledGrammar> grammar) {
  return GrammarMatcherWrapper::Create(grammar);
}

const xgrammar::TokenizerInfo& GrammarBackendNvidia::GetTokenizerInfo() const { return *tokenizer_info_; }

}  // namespace ksana_llm