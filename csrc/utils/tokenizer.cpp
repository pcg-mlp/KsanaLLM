/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/

#include "utils/tokenizer.h"

#include <xgrammar/xgrammar.h>
#include <algorithm>

#include "configure/environment.h"
#include "nlohmann/json.hpp"
#include "utils/logger.h"
#include "utils/singleton.h"

namespace py = pybind11;

namespace ksana_llm {

Status Tokenizer::InitTokenizer(const std::string& tokenizer_path) {
  try {
    pybind11::gil_scoped_acquire acquire;
    py::module transformers = py::module::import("transformers");
    py::object auto_tokenizer = transformers.attr("AutoTokenizer");
    tokenizer_ = auto_tokenizer.attr("from_pretrained")(tokenizer_path, py::arg("trust_remote_code") = true);
  } catch (const py::error_already_set& e) {
    KLLM_LOG_ERROR << fmt::format("Failed to load tokenizer from tokenizer_path {}: {}", tokenizer_path, e.what());
    PyErr_Clear();
    return Status(RET_INVALID_ARGUMENT, fmt::format("Failed to init the tokenizer from {}.", tokenizer_path));
  }
  return Status();
}

void Tokenizer::DestroyTokenizer() {
  pybind11::gil_scoped_acquire acquire;
  tokenizer_ = py::none();
}

Status Tokenizer::Decode(std::vector<int>& output_tokens, std::string& output, bool skip_special_tokens) {
  pybind11::gil_scoped_acquire acquire;
  py::object tokens = tokenizer_.attr("decode")(output_tokens, py::arg("skip_special_tokens") = skip_special_tokens);
  output = tokens.cast<std::string>();
  return Status();
}

Status Tokenizer::Encode(const std::string& prompt, std::vector<int>& input_tokens, bool add_special_tokens) {
  pybind11::gil_scoped_acquire acquire;
  py::object tokens = tokenizer_.attr("encode")(prompt, py::arg("add_special_tokens") = add_special_tokens);
  input_tokens = tokens.cast<std::vector<int>>();
  return Status();
}

Status Tokenizer::GetVocabInfo(std::vector<std::string>& vocab, int& vocab_size, std::vector<int>& stop_token_ids) {
  // Ref: https://github.com/mlc-ai/xgrammar/blob/v0.1.21/python/xgrammar/tokenizer_info.py#L146
  try {
    pybind11::gil_scoped_acquire acquire;
    py::dict vocab_dict = tokenizer_.attr("get_vocab")();

    // Some tokenizer don't have token id 0 or 1 or 2. So the max_id could be larger than the
    // number of tokens. This follows xgrammar's from_huggingface implementation.
    int max_id = -1;
    for (auto item : vocab_dict) {
      int token_id = item.second.cast<int>();
      max_id = std::max(max_id, token_id);
    }

    // Use the input vocab_size (from model_config) or tokenizer_vocab_size, whichever is larger
    // This follows xgrammar's logic where vocab_size can be larger than tokenizer's vocabulary size
    int tokenizer_vocab_size = std::max(static_cast<int>(vocab_dict.size()), max_id + 1);
    int final_vocab_size = std::max(vocab_size, tokenizer_vocab_size);

    vocab.resize(final_vocab_size);

    // maintain tokenizer's indexing
    for (auto item : vocab_dict) {
      std::string token = item.first.cast<std::string>();
      int token_id = item.second.cast<int>();
      if (token_id < final_vocab_size) {
        vocab[token_id] = token;
      }
    }

    // Get stop token ids
    stop_token_ids.clear();
    if (py::hasattr(tokenizer_, "eos_token_id")) {
      py::object eos_token_id = tokenizer_.attr("eos_token_id");
      if (!eos_token_id.is_none()) {
        stop_token_ids.push_back(eos_token_id.cast<int>());
      }
    }

    // Update vocab_size to the final calculated value
    vocab_size = final_vocab_size;

    KLLM_LOG_INFO << "Extracted tokenizer info: input_vocab_size=" << vocab_size
                  << ", tokenizer_vocab_size=" << tokenizer_vocab_size << ", final_vocab_size=" << final_vocab_size
                  << ", vocab_dict_size=" << vocab_dict.size() << ", max_token_id=" << max_id
                  << ", stop_tokens=" << stop_token_ids.size();

    return Status();
  } catch (const std::exception& e) {
    KLLM_LOG_WARNING << "Failed to extract tokenizer info: " << e.what();
    return Status(RET_INVALID_ARGUMENT, "Failed to extract tokenizer information");
  }
}

Status Tokenizer::DetectTokenizerMetadata(VocabInfo& vocab_info) {
  try {
    pybind11::gil_scoped_acquire acquire;

    // Check if it's PreTrainedTokenizerFast
    pybind11::module transformers = pybind11::module::import("transformers");
    pybind11::object fast_tokenizer_class = transformers.attr("PreTrainedTokenizerFast");

    if (pybind11::isinstance(tokenizer_, fast_tokenizer_class)) {
      // Get backend_tokenizer.to_str()
      pybind11::object backend_tokenizer = tokenizer_.attr("backend_tokenizer");
      std::string backend_str = backend_tokenizer.attr("to_str")().cast<std::string>();

      // Use XGrammar C++ API to detect metadata
      std::string metadata_str = xgrammar::TokenizerInfo::DetectMetadataFromHF(backend_str);
      KLLM_LOG_INFO << "XGrammar detected metadata: " << metadata_str;

      // Parse metadata JSON: {"vocab_type": 1, "add_prefix_space": false}
      nlohmann::json metadata_json = nlohmann::json::parse(metadata_str);

      if (metadata_json.contains("vocab_type")) {
        vocab_info.vocab_type = metadata_json["vocab_type"].get<int>();
      }

      if (metadata_json.contains("add_prefix_space")) {
        vocab_info.add_prefix_space = metadata_json["add_prefix_space"].get<bool>();
      }

      KLLM_LOG_INFO << "Detected tokenizer type: " << vocab_info.vocab_type
                    << ", add_prefix_space: " << vocab_info.add_prefix_space;
    } else {
      // Non-fast tokenizer: detect vocab type from Python object attributes.
      //
      // BYTE_LEVEL(2): has byte_encoder attr (GPT-2 style bytes_to_unicode mapping)
      //   e.g. TikTokenTokenizer (Kimi K2), GPT2Tokenizer
      //
      // BYTE_FALLBACK(1): has sp_model attr + vocab contains "<0x0A>" (byte fallback tokens)
      //   e.g. LlamaTokenizer, QwenTokenizer (old)
      //   Note: some sp_model tokenizers (e.g. ChatGLM3) don't use byte fallback → RAW
      //
      // RAW(0): everything else
      if (py::hasattr(tokenizer_, "byte_encoder")) {
        vocab_info.vocab_type = static_cast<int>(xgrammar::VocabType::BYTE_LEVEL);
        KLLM_LOG_INFO << "Detected byte_encoder attribute, vocab_type=BYTE_LEVEL";
      } else {
        // Check SentencePiece: tokenizer.sp_model / tokenizer.tokenizer.sp_model
        bool has_sp_model = py::hasattr(tokenizer_, "sp_model");
        if (!has_sp_model && py::hasattr(tokenizer_, "tokenizer")) {
          has_sp_model = py::hasattr(tokenizer_.attr("tokenizer"), "sp_model");
        }

        if (has_sp_model) {
          // Confirm byte fallback by checking if "<0x0A>" exists in the already-extracted vocab.
          // This avoids a redundant get_vocab() Python call (vocab_info.vocab was filled by GetVocabInfo).
          bool has_byte_fallback =
              std::find(vocab_info.vocab.begin(), vocab_info.vocab.end(), "<0x0A>") != vocab_info.vocab.end();
          if (has_byte_fallback) {
            vocab_info.vocab_type = static_cast<int>(xgrammar::VocabType::BYTE_FALLBACK);
            // Detect add_prefix_space from tokenizer attribute; default to true for
            // compatibility with LLaMA/Mistral (matching XGrammar Python behavior).
            vocab_info.add_prefix_space = true;
            if (py::hasattr(tokenizer_, "add_prefix_space")) {
              py::object aps = tokenizer_.attr("add_prefix_space");
              if (!aps.is_none()) {
                vocab_info.add_prefix_space = aps.cast<bool>();
              }
            }
            KLLM_LOG_INFO << "Detected sp_model + <0x0A> in vocab, vocab_type=BYTE_FALLBACK"
                          << ", add_prefix_space=" << vocab_info.add_prefix_space;
          } else {
            KLLM_LOG_INFO << "Detected sp_model but no byte fallback tokens, vocab_type=RAW(0)";
          }
        } else {
          KLLM_LOG_INFO << "No byte_encoder or sp_model found, vocab_type=RAW(0)";
        }
      }
    }

    return Status();
  } catch (const std::exception& e) {
    KLLM_LOG_WARNING << "Failed to detect tokenizer type: " << e.what() << ", using default: RAW";
    return Status(RET_RUNTIME_FAILED, "Failed to detect tokenizer metadata");
  }
}

}  // namespace ksana_llm
