/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/weight_instance.h"

#include <future>
#include <memory>
#include <vector>

#include "device/device_types.h"
#include "nlohmann/json.hpp"
#include "utils/get_custom_weight_name.h"
#include "utils/logger.h"
#include "utils/optional_file.h"
#include "utils/status.h"

#include "models/baichuan/baichuan_weight.h"
#include "models/glm4_moe/glm4_moe_weight.h"
#include "models/gpt/gpt_weight.h"
#include "models/hunyuan_large/hunyuan_large_weight.h"
#include "models/llama/llama_weight.h"
#include "models/llama4/llama4_weight.h"
#include "models/qwen2_moe/qwen2_moe_weight.h"
#include "models/qwen3_moe/qwen3_moe_weight.h"

#include "utils/gguf_file_tensor_loader.h"
#include "utils/pytorch_file_tensor_loader.h"
#include "utils/safetensors_file_tensor_loader.h"

#include "models/deepseek_v3/deepseek_v3_config.h"
#include "models/hunyuan_turbo/hunyuan_turbo_config.h"
#include "models/internlm2/internlm2_config.h"
#include "models/internlmxcomposer2/internlmxcomposer2_config.h"
#include "models/llama/llama_model_config.h"
#include "models/qwen/qwen_config.h"

namespace ksana_llm {

// Create the object and return a shared pointer.
template <template <class> class ClassT, class... Args>
std::shared_ptr<BaseWeight> CreateModelWeight(int rank, ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                              std::shared_ptr<Context>& context) {
  std::shared_ptr<BaseWeight> model_obj = nullptr;
  switch (model_config.weight_data_type) {
    case DataType::TYPE_FP16:
      model_obj = std::make_shared<ClassT<float16>>(model_config, runtime_config, rank, context);
      break;
    case DataType::TYPE_BF16:
      model_obj = std::make_shared<ClassT<bfloat16>>(model_config, runtime_config, rank, context);
      break;
    case DataType::TYPE_FP32:
      model_obj = std::make_shared<ClassT<float>>(model_config, runtime_config, rank, context);
      break;
    default:
      KLLM_THROW(fmt::format("Unsupported Tensor type: {}.", model_config.weight_data_type));
  }
  return model_obj;
}

template <template <class> class WeightType>
void CreateWeightInstance(const std::string model_name, ModelConfig& model_config, const RuntimeConfig& runtime_config,
                          std::shared_ptr<Context>& context, std::shared_ptr<BaseWeight>& weight) {
  KLLM_LOG_INFO << "Start to init model instance " << model_name;
  int worker_id = context->device->GetDeviceId();
  KLLM_LOG_INFO << "Start to create empty model weight on device " << worker_id;
  weight = CreateModelWeight<WeightType>(worker_id, model_config, runtime_config, context);
}

void WeightInstance::CreateWeightInstances() {
  std::string unified_model_type = model_config_.type;
  // unify it to lower case
  std::transform(unified_model_type.begin(), unified_model_type.end(), unified_model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (unified_model_type.find("llama4") != std::string::npos) {
    CreateWeightInstance<Llama4Weight>(unified_model_type, model_config_, runtime_config_, context_, weight_);
  } else if (unified_model_type.find("llama") != std::string::npos) {
    CreateWeightInstance<LlamaWeight>(unified_model_type, model_config_, runtime_config_, context_, weight_);
  } else if (unified_model_type.find("qwen3_moe") != std::string::npos) {
    CreateWeightInstance<Qwen3MoeWeight>(unified_model_type, model_config_, runtime_config_, context_, weight_);
  } else if (unified_model_type.find("qwen2_moe") != std::string::npos) {
    CreateWeightInstance<Qwen2MoeWeight>(unified_model_type, model_config_, runtime_config_, context_, weight_);
  } else if (unified_model_type.find("glm4_moe") != std::string::npos) {
    // GLM4-MoE uses custom weight class to handle dense+MoE hybrid architecture
    CreateWeightInstance<Glm4MoeWeight>(unified_model_type, model_config_, runtime_config_, context_, weight_);
  } else if (unified_model_type.find("baichuan") != std::string::npos) {
    CreateWeightInstance<BaichuanWeight>(unified_model_type, model_config_, runtime_config_, context_, weight_);
  } else if (unified_model_type.find("gpt") != std::string::npos ||
             unified_model_type.find("fairseq-transformer") != std::string::npos) {
    CreateWeightInstance<GPTWeight>(unified_model_type, model_config_, runtime_config_, context_, weight_);
  } else if (unified_model_type.find("hunyuan") != std::string::npos && model_config_.is_moe) {
    CreateWeightInstance<HunyuanLargeWeight>(unified_model_type, model_config_, runtime_config_, context_, weight_);
  } else {
    // Optional weights map
    auto optional_file = Singleton<OptionalFile>::GetInstance();
    std::string& weight_map =
        optional_file->GetOptionalFile(model_config_.path, "weight_map", unified_model_type + "_weight_map.json");
    if (weight_map != "") {
      CreateWeightInstance<LlamaWeight>(unified_model_type, model_config_, runtime_config_, context_, weight_);
    } else {
      KLLM_THROW(fmt::format("Model type {} is not supported.", unified_model_type));
    }
  }
}

void WeightInstance::Load() {
  const char* const enable_old_loader = std::getenv("ENABLE_OLD_LOADER");
  bool use_old_loader = (enable_old_loader != nullptr && strcmp(enable_old_loader, "1") == 0);

  ModelConfigParser model_config_parser;
  std::shared_ptr<BaseModelConfig> model_config;
  Status status =
      model_config_parser.ParseModelConfig(model_config_.path, runtime_config_.parallel_basic_config, model_config);

  // Pass is_embedding_model from runtime ModelConfig to the weight loader's BaseModelConfig.
  // TODO(long-term): Consider passing is_embedding_model through ParseModelConfig parameters
  // instead of post-hoc assignment, to avoid this "patch-style" parameter passing.
  if (status.OK() && model_config) {
    model_config->is_embedding_model = model_config_.is_embedding_model;
  }

  if (status.OK() && use_old_loader && model_config->model_arch == ModelArchitecture::ARCH_DEEPSEEK) {
    KLLM_LOG_WARNING << "DeepSeek is disabled for old model loader now. Ignore env variable `ENABLE_OLD_LOADER`";
    use_old_loader = false;
  }

  if (status.OK() && IsCompatibleWithNewLoader(model_config) && !use_old_loader) {
    KLLM_LOG_INFO << "Using new loader to load model weights";
    ModelWeightLoader weight_loader(Singleton<Environment>::GetInstance(), context_);

    std::shared_ptr<ModelWeight> dev_weight = std::make_shared<ModelWeight>();
    status = weight_loader.LoadWeight(model_config, dev_weight);
    STATUS_CHECK_FAILURE(status);
    weight_ = dev_weight;
  } else {
    KLLM_LOG_INFO << "Using old loader to load model weights";
    CreateWeightInstances();
    bool loaded_from_cache = false;
    LoadWeightsAndModelsMap(loaded_from_cache);
    if (loaded_from_cache) {
      KLLM_LOG_INFO << "Model was loaded from cache";
    }
  }
}

void WeightInstance::SetEmbeddingsConfig() {
  if (weight_) {
    weight_->SetEmbeddingsConfig();
  }
}
/*
 * embed_token.weight 和 lm_head.weight 的检查和替换逻辑如下表所示：
 * （第一列为模型config.json中是否存在参数tie_word_embeddings）
 * （第二列为参数tie_word_embeddings实际默认值）
 *  (第三列为是否存在lm_head.weight)
 *  (第四列为embed_token.weight是否替换lm_head.weight)
 *   +-----------+--------+---------------+-------------+
 *   | exist tie | value  | exist lm_head | is replace  |
 *   +-----------+--------+---------------+-------------+
 *   |           |        |     true      |     NO      |
 *   |           |  true  +---------------+-------------|
 *   |           |        |     false     |     YES     |
 *   | false     +--------+---------------+-------------+
 *   |           |        |     true      |     NO      |
 *   |           |  false +---------------+-------------|
 *   |           |        |     false     |     YES     |
 *   +-----------+--------+---------------+-------------+
 *   |           |        |     true      |     YES     |
 *   |           |  true  +---------------+-------------|
 *   |           |        |     false     |     YES     |
 *   |  true     +--------+---------------+-------------+
 *   |           |  false |     true      |     NO      |
 *   +-----------+--------+---------------+-------------+
 */
void WeightInstance::CheckTieEmbeddings(int weight_file_size) {
  if (weight_file_size <= 1 || model_config_.exist_tie_embeddings_param) {
    return;
  }
  // When the quantity of weight files exceeds 1, retrieve the "index.json" file mapping the names of the weights
  // under the model path.
  for (const auto& entry : std::filesystem::directory_iterator(model_config_.path)) {
    std::string index_filename = entry.path().filename().string();
    if (index_filename.size() > 11 && index_filename.substr(index_filename.size() - 11) == ".index.json" &&
        index_filename.substr(6) != ".etag.") {
      std::ifstream file(entry.path());
      nlohmann::json weights_index_json;
      file >> weights_index_json;
      if (!weights_index_json["weight_map"].contains("lm_head.weight") &&
          !weights_index_json["weight_map"].contains("transformer.output_layer.weight") &&
          !weights_index_json["weight_map"].contains("language_model.lm_head.weight")) {
        SetEmbeddingsConfig();
        KLLM_LOG_INFO
            << "tie_word_embeddings param and lm_head.weight are not exist, replace it with embedd_tokens.weight";
        break;
      }
    }
  }
}

void WeightInstance::CheckTieEmbeddings(const std::vector<std::string>& custom_name_list) {
  if (!model_config_.exist_tie_embeddings_param) {
    // When the quantity of weight files is equal to 1, the weight file should be loaded directly before the name search
    // is performed.
    const std::string lm_head_weight = "lm_head.weight";
    const auto exist_lm_head = std::find(custom_name_list.begin(), custom_name_list.end(), lm_head_weight);
    if (exist_lm_head == custom_name_list.end()) {
      SetEmbeddingsConfig();
      KLLM_LOG_INFO
          << "tie_word_embeddings param and lm_head.weight are not exist, replace it with the embedd_tokens.weight";
    }
  }
}

void WeightInstance::LoadWeightsAndModelsMap(bool& loaded_from_cache) {
  if (std::getenv("ENABLE_MODEL_CACHE") != nullptr) {
    loaded_from_cache = TryToLoadWeightsFromCache();
  }
  if (!loaded_from_cache) {
    LoadWeights();
  }
  if (std::getenv("ENABLE_MODEL_CACHE") != nullptr && !loaded_from_cache) {
    SaveWeightsToCache();
  }

  ProcessWeights();
}

bool WeightInstance::SaveWeightsToCache() {
  std::vector<std::future<void>> save_weight_tasks;
  save_weight_tasks.reserve(context_->global->GetTensorParallelSize());

  for (size_t worker_id = 0; worker_id < context_->global->GetTensorParallelSize(); ++worker_id) {
    save_weight_tasks.emplace_back(loader_weight_threadpool_->Submit([worker_id, this]() {
      StreamSynchronize(this->context_->device->GetMemoryManageStream());
      this->weight_->SaveWeightsToCacheFolder();
      StreamSynchronize(this->context_->device->GetMemoryManageStream());
    }));
  }
  for (auto&& save_weight_task : save_weight_tasks) {
    save_weight_task.get();
  }
  return true;
}

bool WeightInstance::TryToLoadWeightsFromCache() {
  auto start_time = std::chrono::high_resolution_clock::now();
  std::vector<std::future<bool>> load_weight_tasks;
  for (size_t worker_id = 0; worker_id < context_->global->GetTensorParallelSize(); ++worker_id) {
    load_weight_tasks.push_back(loader_weight_threadpool_->Submit([worker_id, this]() {
      bool res = this->weight_->TryToLoadWeightsFromCache();
      StreamSynchronize(this->context_->device->GetMemoryManageStream());
      return res;
    }));
  }
  bool is_success = true;
  for (auto&& load_weight_task : load_weight_tasks) {
    is_success = is_success && load_weight_task.get();
  }
  std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start_time;
  KLLM_LOG_INFO << fmt::format("load cache cost: {} seconds", elapsed.count());
  if (is_success) {
    KLLM_LOG_INFO << "Load model from cache successfully";
  }
  return is_success;
}

void WeightInstance::LoadWeights() {
  const auto start_time = std::chrono::high_resolution_clock::now();
  ModelFileFormat model_file_format;
  const std::vector<std::string>& weights_file_list = SearchLocalPath(model_config_.path, model_file_format);
  const size_t weight_file_size = weights_file_list.size();
  CheckTieEmbeddings(weight_file_size);

  std::vector<std::future<void>> load_weight_tasks;
  std::vector<std::mutex> tp_mutex(context_->global->GetTensorParallelSize());

  for (const std::string& file_name : weights_file_list) {
    std::shared_ptr<BaseFileTensorLoader> weights_loader = nullptr;
    if (model_file_format == SAFETENSORS) {
      weights_loader = std::make_shared<SafeTensorsLoader>(file_name, model_config_.load_bias);
    } else if (model_file_format == GGUF) {
      weights_loader = std::make_shared<GGUFFileTensorLoader>(file_name, model_config_.load_bias);
    } else {
      weights_loader = std::make_shared<PytorchFileTensorLoader>(file_name, model_config_.load_bias);
    }

    std::vector<std::string> custom_name_list;
    GetCustomNameList(model_config_.path, model_config_.type, weights_loader->GetTensorNameList(), custom_name_list,
                      model_file_format);

    if (weight_file_size == 1) {
      CheckTieEmbeddings(custom_name_list);
    }

    for (size_t tp_i = 0; tp_i < context_->global->GetTensorParallelSize(); ++tp_i) {
      load_weight_tasks.emplace_back(
          loader_weight_threadpool_->Submit([tp_i, this, weights_loader, custom_name_list, &tp_mutex]() {
            // The current implementation can only process serially per card.
            std::lock_guard<std::mutex> lock(tp_mutex[tp_i]);
            const auto& weight_name_list = weights_loader->GetTensorNameList();
            this->weight_->LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list);
            StreamSynchronize(this->context_->device->GetMemoryManageStream());
          }));
    }

    if (model_file_format != SAFETENSORS) {
      for (auto&& get_weight_task : load_weight_tasks) {
        get_weight_task.get();
      }
      load_weight_tasks.clear();
    }
  }

  for (auto&& get_weight_task : load_weight_tasks) {
    get_weight_task.get();
  }

  const std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start_time;
  KLLM_LOG_INFO << fmt::format("load cost: {} seconds", elapsed.count());
}

void WeightInstance::ProcessWeights() {
  const auto start_time = std::chrono::high_resolution_clock::now();
  std::vector<std::future<void>> process_weight_tasks;
  process_weight_tasks.reserve(context_->global->GetTensorParallelSize());
  for (size_t worker_id = 0; worker_id < context_->global->GetTensorParallelSize(); ++worker_id) {
    process_weight_tasks.emplace_back(
        loader_weight_threadpool_->Submit([worker_id, this]() { this->weight_->ProcessWeights(); }));
  }
  for (auto&& process_weight_task : process_weight_tasks) {
    process_weight_task.get();
  }
  const std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start_time;
  KLLM_LOG_INFO << fmt::format("process cost: {} seconds", elapsed.count());
}

bool WeightInstance::IsCompatibleWithNewLoader(std::shared_ptr<BaseModelConfig> model_config) {
// Not support HUAWEI ascend NPU
#ifdef ENABLE_ACL
  return false;
#endif

  // Null check should be first
  if (model_config == nullptr) {
    return false;
  }

  // Temporarily support partly llama, qwen-dense, deepseek v3 and chatglm on nvidia-GPU
  switch (model_config->model_arch) {
    case ModelArchitecture::ARCH_CHATGLM: {
      // Support ChatGLM with normal precision (no quantization)
      // Currently only support non-quantized ChatGLM models
      return true;
    }

    case ModelArchitecture::ARCH_LLAMA: {
      std::shared_ptr<LlamaModelConfig> llama_model_config = std::dynamic_pointer_cast<LlamaModelConfig>(model_config);
      if (!llama_model_config->is_quant) {
        return true;
      }
      // not support quantized llama model
      return false;
    }

    case ModelArchitecture::ARCH_QWEN: {
      return true;
    }

    case ModelArchitecture::ARCH_INTERNLM2:
      [[fallthrough]];
    case ModelArchitecture::ARCH_INTERNLMXCOMPOSER2:
      [[fallthrough]];
    case ModelArchitecture::ARCH_MINICPM:
      [[fallthrough]];
    case ModelArchitecture::ARCH_DEEPSEEK:
      [[fallthrough]];
    case ModelArchitecture::ARCH_QWEN2_MOE:
      [[fallthrough]];
    case ModelArchitecture::ARCH_QWEN3_MOE:
      [[fallthrough]];
    case ModelArchitecture::ARCH_MIXTRAL:
      [[fallthrough]];
    case ModelArchitecture::ARCH_GLM4_MOE:
      [[fallthrough]];
    case ModelArchitecture::ARCH_ARC_HUNYUAN_VIDEO:
      return true;

    case ModelArchitecture::ARCH_HUNYUAN_LARGE: {
      // TODO(zhongzhicao): support hunyuan-large
      return false;
    }

    case ModelArchitecture::ARCH_HUNYUAN_TURBO: {
      return true;
    }

    default:
      return false;
  }
}

}  // namespace ksana_llm
