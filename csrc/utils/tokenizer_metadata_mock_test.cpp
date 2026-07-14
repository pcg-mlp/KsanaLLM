/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/

// Tests for non-fast tokenizer vocab_type detection using Python mock objects.
// These tests verify the three detection paths in DetectTokenizerMetadata's else branch:
//   1. byte_encoder attribute → BYTE_LEVEL(2)
//   2. sp_model attribute + "<0x0A>" in vocab → BYTE_FALLBACK(1)
//   3. no matching attributes → RAW(0)
//
// These tests use types.SimpleNamespace as mock tokenizer objects, so they don't require
// real model files and can run in standalone test mode.

#include "utils/tokenizer.h"

#include "gflags/gflags.h"
#include "test.h"

#include "utils/logger.h"
#include "utils/singleton.h"

namespace ksana_llm {

TEST(TokenizerMetadataMockTest, DetectMetadata_ByteLevel) {
  // Mock a non-fast tokenizer with byte_encoder attribute (like Kimi K2's TikTokenTokenizer)
  pybind11::gil_scoped_acquire acquire;

  pybind11::module types = pybind11::module::import("types");
  pybind11::object mock_tokenizer = types.attr("SimpleNamespace")();
  mock_tokenizer.attr("byte_encoder") = pybind11::dict();  // Just needs to exist

  auto tokenizer = Singleton<Tokenizer>::GetInstance();
  tokenizer->tokenizer_ = mock_tokenizer;

  VocabInfo vocab_info;
  vocab_info.initialized = true;
  vocab_info.vocab_size = 100;
  vocab_info.vocab.resize(100);

  Status status = tokenizer->DetectTokenizerMetadata(vocab_info);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(vocab_info.vocab_type, 2) << "Expected BYTE_LEVEL(2) for tokenizer with byte_encoder attribute";
  EXPECT_FALSE(vocab_info.add_prefix_space);

  tokenizer->tokenizer_ = pybind11::none();
}

TEST(TokenizerMetadataMockTest, DetectMetadata_ByteFallback) {
  // Mock a non-fast tokenizer with sp_model attribute and byte fallback vocab (like LlamaTokenizer)
  pybind11::gil_scoped_acquire acquire;

  pybind11::module types = pybind11::module::import("types");
  pybind11::object mock_tokenizer = types.attr("SimpleNamespace")();
  mock_tokenizer.attr("sp_model") = pybind11::none();  // Just needs to exist

  auto tokenizer = Singleton<Tokenizer>::GetInstance();
  tokenizer->tokenizer_ = mock_tokenizer;

  VocabInfo vocab_info;
  vocab_info.initialized = true;
  vocab_info.vocab_size = 10;
  vocab_info.vocab = {"<unk>", "<s>", "</s>", "hello", "world", "<0x0A>", "<0x0B>", "foo", "bar", "baz"};

  Status status = tokenizer->DetectTokenizerMetadata(vocab_info);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(vocab_info.vocab_type, 1) << "Expected BYTE_FALLBACK(1) for tokenizer with sp_model + <0x0A> in vocab";
  EXPECT_TRUE(vocab_info.add_prefix_space);

  tokenizer->tokenizer_ = pybind11::none();
}

TEST(TokenizerMetadataMockTest, DetectMetadata_SpModelNoByteFallback) {
  // Mock a non-fast tokenizer with sp_model but without byte fallback tokens (like ChatGLM3)
  pybind11::gil_scoped_acquire acquire;

  pybind11::module types = pybind11::module::import("types");
  pybind11::object mock_tokenizer = types.attr("SimpleNamespace")();
  mock_tokenizer.attr("sp_model") = pybind11::none();

  auto tokenizer = Singleton<Tokenizer>::GetInstance();
  tokenizer->tokenizer_ = mock_tokenizer;

  VocabInfo vocab_info;
  vocab_info.initialized = true;
  vocab_info.vocab_size = 5;
  vocab_info.vocab = {"<unk>", "<s>", "</s>", "hello", "world"};  // No <0x0A>

  Status status = tokenizer->DetectTokenizerMetadata(vocab_info);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(vocab_info.vocab_type, 0) << "Expected RAW(0) for sp_model tokenizer without byte fallback tokens";
  EXPECT_FALSE(vocab_info.add_prefix_space);

  tokenizer->tokenizer_ = pybind11::none();
}

TEST(TokenizerMetadataMockTest, DetectMetadata_Raw) {
  // Mock a non-fast tokenizer with no byte_encoder and no sp_model → RAW
  pybind11::gil_scoped_acquire acquire;

  pybind11::module types = pybind11::module::import("types");
  pybind11::object mock_tokenizer = types.attr("SimpleNamespace")();
  // No byte_encoder, no sp_model — just a bare object

  auto tokenizer = Singleton<Tokenizer>::GetInstance();
  tokenizer->tokenizer_ = mock_tokenizer;

  VocabInfo vocab_info;
  vocab_info.initialized = true;
  vocab_info.vocab_size = 5;
  vocab_info.vocab = {"<unk>", "<s>", "</s>", "hello", "world"};

  Status status = tokenizer->DetectTokenizerMetadata(vocab_info);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(vocab_info.vocab_type, 0) << "Expected RAW(0) for tokenizer without byte_encoder or sp_model";
  EXPECT_FALSE(vocab_info.add_prefix_space);

  tokenizer->tokenizer_ = pybind11::none();
}

}  // namespace ksana_llm
