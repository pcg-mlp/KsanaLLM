/* Copyright 2025 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/

#include "utils/tokenizer.h"

#include "gflags/gflags.h"
#include "test.h"

#include "utils/logger.h"
#include "utils/singleton.h"

namespace ksana_llm {

TEST(TokenizerTest, WrongTokenizerPath) {
  Status status = Singleton<Tokenizer>::GetInstance()->InitTokenizer("wrong_path");
  EXPECT_EQ(status.GetCode(), RET_INVALID_ARGUMENT);
}

TEST(TokenizerTest, TokenizeTest) {
  Singleton<Tokenizer>::GetInstance()->InitTokenizer("/model/llama-hf/7B");
  std::string prompt = "Hello. What's your name?";
  std::vector<int> token_list;
  std::vector<int> target_token_list = {1, 15043, 29889, 1724, 29915, 29879, 596, 1024, 29973};
  Singleton<Tokenizer>::GetInstance()->Encode(prompt, token_list, true);
  EXPECT_EQ(token_list.size(), target_token_list.size());
  for (size_t i = 0; i < token_list.size(); ++i) {
    EXPECT_EQ(token_list[i], target_token_list[i]);
  }

  std::string output_prompt = "";
  std::string target_prompt = "Hello. What's your name? My name is David.";
  token_list.emplace_back(1619);
  token_list.emplace_back(1024);
  token_list.emplace_back(338);
  token_list.emplace_back(4699);
  token_list.emplace_back(29889);
  Singleton<Tokenizer>::GetInstance()->Decode(token_list, output_prompt, true);
  EXPECT_EQ(target_prompt, output_prompt);

  Singleton<Tokenizer>::GetInstance()->DestroyTokenizer();
}

TEST(TokenizerTest, GetVocabInfoTest) {
  // Initialize tokenizer first
  Singleton<Tokenizer>::GetInstance()->InitTokenizer("/model/llama-hf/7B");

  std::vector<std::string> vocab;
  int vocab_size = 32000;
  std::vector<int> stop_token_ids;

  Status status = Singleton<Tokenizer>::GetInstance()->GetVocabInfo(vocab, vocab_size, stop_token_ids);
  EXPECT_TRUE(status.OK());

  // Verify basic functionality
  EXPECT_GE(vocab_size, 32000);                              // Should be at least the input size
  EXPECT_EQ(vocab.size(), static_cast<size_t>(vocab_size));  // vocab vector should match vocab_size
  EXPECT_GT(stop_token_ids.size(), 0);                       // Should have at least one stop token (EOS)

  // Verify some tokens exist
  EXPECT_FALSE(vocab[1].empty());  // Token ID 1 should exist (BOS token for LLaMA)

  // Clean up
  Singleton<Tokenizer>::GetInstance()->DestroyTokenizer();
}

TEST(TokenizerTest, GetVocabInfoErrorHandlingTest) {
  // Test error handling when tokenizer is not properly initialized
  // This should trigger the catch block in GetVocabInfo

  // Ensure tokenizer is destroyed/uninitialized
  Singleton<Tokenizer>::GetInstance()->DestroyTokenizer();

  std::vector<std::string> vocab;
  int vocab_size = 32000;
  std::vector<int> stop_token_ids;

  // Call GetVocabInfo without proper initialization
  // This should trigger the exception handling in GetVocabInfo
  Status status = Singleton<Tokenizer>::GetInstance()->GetVocabInfo(vocab, vocab_size, stop_token_ids);

  // Verify that the error is properly handled
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RET_INVALID_ARGUMENT);
  EXPECT_EQ(status.GetMessage(), "Failed to extract tokenizer information");
}

TEST(TokenizerTest, DetectTokenizerMetadataTest) {
  // Initialize tokenizer with LLaMA model
  Singleton<Tokenizer>::GetInstance()->InitTokenizer("/model/llama-hf/7B");

  VocabInfo vocab_info;
  vocab_info.vocab_size = 32000;
  vocab_info.initialized = false;

  // First get vocab info
  Status status = Singleton<Tokenizer>::GetInstance()->GetVocabInfo(vocab_info.vocab, vocab_info.vocab_size,
                                                                    vocab_info.stop_token_ids);
  EXPECT_TRUE(status.OK());
  vocab_info.initialized = true;

  // Then detect tokenizer metadata
  status = Singleton<Tokenizer>::GetInstance()->DetectTokenizerMetadata(vocab_info);
  EXPECT_TRUE(status.OK());

  // Llama 模型使用 ByteFallback decoder，应该被检测为 BYTE_FALLBACK (vocab_type=1)
  // 根据 XGrammar 的定义：
  // - RAW = 0
  // - BYTE_FALLBACK = 1
  // - BYTE_LEVEL = 2
  KLLM_LOG_INFO << "Detected vocab_type: " << vocab_info.vocab_type;

  // Llama 模型应该被检测为 BYTE_FALLBACK
  EXPECT_EQ(vocab_info.vocab_type, 1) << "Expected BYTE_FALLBACK (1) for Llama model with ByteFallback decoder, got "
                                      << vocab_info.vocab_type;

  // Verify vocab_size matches
  EXPECT_EQ(vocab_info.vocab_size, 32000);

  // Verify add_prefix_space is set (LLaMA models typically don't use prefix space)
  EXPECT_FALSE(vocab_info.add_prefix_space);

  // Clean up
  Singleton<Tokenizer>::GetInstance()->DestroyTokenizer();
}

TEST(TokenizerTest, DetectTokenizerMetadataErrorHandlingTest) {
  // Test error handling when tokenizer is not initialized
  Singleton<Tokenizer>::GetInstance()->DestroyTokenizer();

  VocabInfo vocab_info;
  vocab_info.vocab_size = 32000;
  vocab_info.initialized = false;

  // Call DetectTokenizerMetadata without proper initialization
  Status status = Singleton<Tokenizer>::GetInstance()->DetectTokenizerMetadata(vocab_info);

  // Should fail
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RET_RUNTIME_FAILED);
}

}  // namespace ksana_llm
