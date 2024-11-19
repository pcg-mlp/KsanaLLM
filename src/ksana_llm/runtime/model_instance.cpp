/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/model_instance.h"
#include <future>
#include <memory>
#include <vector>

#include "ksana_llm/runtime/worker.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/get_custom_weight_name.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/optional_file.h"
#include "ksana_llm/utils/status.h"
#include "nlohmann/json.hpp"

#include "ksana_llm/models/baichuan/baichuan_weight.h"
#include "ksana_llm/models/chatglm/chatglm_weight.h"
#include "ksana_llm/models/gpt/gpt_weight.h"
#include "ksana_llm/models/llama/llama_weight.h"
#include "ksana_llm/models/mixtral/mixtral_weight.h"
#include "ksana_llm/models/qwen/qwen_weight.h"
#include "ksana_llm/models/qwen2_moe/qwen2_moe_weight.h"

#include "ksana_llm/models/baichuan/baichuan_model.h"
#include "ksana_llm/models/chatglm/chatglm_model.h"
#include "ksana_llm/models/gpt/gpt_model.h"
#include "ksana_llm/models/llama/llama_model.h"
#include "ksana_llm/models/mixtral/mixtral_model.h"
#include "ksana_llm/models/qwen/qwen_model.h"
#include "ksana_llm/models/qwen2_moe/qwen2_moe_model.h"

namespace ksana_llm {

std::vector<std::shared_ptr<BaseModel>> ModelInstance::models_;
std::vector<std::shared_ptr<BaseWeight>> ModelInstance::weights_;

void ModelInstance::InitPlugin() {
  // search optional plugin
  std::string plugin_name = model_config_.type;
  if (model_config_.type == "qwen" && model_config_.is_visual && model_config_.hidden_units == 4096) {
    plugin_name = "qwenvl";
  }

  auto optional_file = Singleton<OptionalFile>::GetInstance();
  std::string& plugin_path =
      optional_file->GetOptionalFile(model_config_.path, "ksana_plugin/" + plugin_name, "ksana_plugin.py");
  py::gil_scoped_acquire acquire;
  // try to load plugin
  try {
    py::module importlib_util = py::module::import("importlib.util");
    py::object spec = importlib_util.attr("spec_from_file_location")("ksana_plugin", plugin_path);
    py::object module = importlib_util.attr("module_from_spec")(spec);
    spec.attr("loader").attr("exec_module")(module);

    plugin_ = std::make_shared<py::object>(module.attr("KsanaPlugin")());

    KLLM_LOG_INFO << "Using Plugin";
  } catch (const py::error_already_set& e) {
    KLLM_LOG_WARNING << "Error loading plugin: " << e.what();
    PyErr_Clear();
  }
  // if load plugin success, try to init plugin
  if (plugin_) {
    py::dict kwargs;
    kwargs["model_path"] = model_config_.path;
    kwargs["enable_trt"] = model_config_.enable_trt;
    kwargs["preprocess"] = true;
    try {
      py::object result = plugin_->attr("init_plugin")(**kwargs);
      if (!result.is_none()) {
        if (py::isinstance<py::dict>(result)) {
          py::dict result_dict = result.cast<py::dict>();
          if (result_dict.contains("reserved_device_memory_ratio")) {
            float reserved_device_memory_ratio = result_dict["reserved_device_memory_ratio"].cast<float>();
            KLLM_LOG_DEBUG << "Plugin fixed reserved_device_memory_ratio : " << reserved_device_memory_ratio;
            Singleton<Environment>::GetInstance()->SetReservedDeviceRatio(reserved_device_memory_ratio);
          }
        }
      }
    } catch (const py::error_already_set& e) {
      KLLM_LOG_WARNING << "Error initializing plugin: " << e.what();
      PyErr_Clear();
    }
  }
  g_plugin = plugin_;
}

void ModelInstance::Load() {
  InitPlugin();

  std::string unified_model_type = model_config_.type;
  // unify it to lower case
  std::transform(unified_model_type.begin(), unified_model_type.end(), unified_model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (unified_model_type.find("llama") != std::string::npos) {
    type = "llama";
    CreateModelInstance<LlamaModel, LlamaWeight>(unified_model_type);
  } else if (unified_model_type.find("qwen2_moe") != std::string::npos) {
    type = "qwen2_moe";
    CreateModelInstance<Qwen2MoeModel, Qwen2MoeWeight>(unified_model_type);
  } else if (unified_model_type.find("qwen") != std::string::npos) {
    type = "qwen";
    CreateModelInstance<QwenModel, QwenWeight>(unified_model_type);
  } else if (unified_model_type.find("baichuan") != std::string::npos) {
    type = "baichuan";
    CreateModelInstance<BaichuanModel, BaichuanWeight>(unified_model_type);
  } else if (unified_model_type.find("chatglm") != std::string::npos) {
    type = "chatglm";
    CreateModelInstance<ChatglmModel, ChatglmWeight>(unified_model_type);
  } else if (unified_model_type.find("gpt") != std::string::npos ||
             unified_model_type.find("fairseq-transformer") != std::string::npos) {
    type = "gpt";
    CreateModelInstance<GPTModel, GPTWeight>(unified_model_type);
  } else if (unified_model_type.find("mixtral") != std::string::npos) {
    type = "mixtral";
    CreateModelInstance<MixtralModel, MixtralWeight>(unified_model_type);
  } else {
    // Optional weights map
    auto optional_file = Singleton<OptionalFile>::GetInstance();
    std::string& weight_map =
        optional_file->GetOptionalFile(model_config_.path, "weight_map", unified_model_type + "_weight_map.json");
    if (weight_map != "") {
      type = "llama";
      CreateModelInstance<LlamaModel, LlamaWeight>(unified_model_type);
    } else {
      KLLM_THROW(fmt::format("Model type {} is not supported.", unified_model_type));
    }
  }
}

std::vector<float*> ModelInstance::GetLogitsPtr() {
  std::vector<float*> results;
  for (auto& model : models_) {
    results.push_back(model->GetLogitsPtr());
  }
  return results;
}

std::vector<Status> ModelInstance::Forward(std::shared_ptr<WorkerGroup> worker_group, InferStage stage,
                                           std::vector<ForwardRequest>& forward_reqs) {
  std::vector<Status> results;
  for (int worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
    results.push_back(
        worker_group->GetWorker(worker_id)->Forward(models_[worker_id], weights_[worker_id], stage, forward_reqs));
  }
  return results;
}

std::vector<std::future<Status>> ModelInstance::ForwardAsync(std::shared_ptr<WorkerGroup> worker_group,
                                                             InferStage stage,
                                                             std::vector<ForwardRequest>& forward_reqs) {
  std::vector<std::future<Status>> results;
  for (int worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
    results.push_back(
        worker_group->GetWorker(worker_id)->ForwardAsync(models_[worker_id], weights_[worker_id], stage, forward_reqs));
  }
  return results;
}

void ModelInstance::SetEmbeddingsConfig() {
  for (auto& weight : weights_) {
    if (weight) {
      weight->SetEmbeddingsConfig();
    }
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
void ModelInstance::CheckTieEmbeddings(int weight_file_size) {
  if (weight_file_size <= 1 || model_config_.exist_tie_embeddings_param) {
    return;
  }
  // When the quantity of weight files exceeds 1, retrieve the "index.json" file mapping the names of the weights
  // under the model path.
  for (const auto& entry : std::filesystem::directory_iterator(model_config_.path)) {
    std::string index_filename = entry.path().filename().string();
    if (index_filename.size() > 11 && index_filename.substr(index_filename.size() - 11) == ".index.json") {
      std::ifstream file(entry.path());
      nlohmann::json weights_index_json;
      file >> weights_index_json;
      if (!weights_index_json["weight_map"].contains("lm_head.weight") &&
          !weights_index_json["weight_map"].contains("transformer.output_layer.weight")) {
        SetEmbeddingsConfig();
        KLLM_LOG_INFO
            << "tie_word_embeddings param and lm_head.weight are not exist, replace it with embedd_tokens.weight";
        break;
      }
    }
  }
}

void ModelInstance::CheckTieEmbeddings(std::vector<std::string>& custom_name_list) {
  if (!model_config_.exist_tie_embeddings_param) {
    // When the quantity of weight files is equal to 1, the weight file should be loaded directly before the name search
    // is performed.
    std::string lm_head_weight = "lm_head.weight";
    auto exist_lm_head = std::find(custom_name_list.begin(), custom_name_list.end(), lm_head_weight);
    if (exist_lm_head == custom_name_list.end()) {
      SetEmbeddingsConfig();
      KLLM_LOG_INFO
          << "tie_word_embeddings param and lm_head.weight are not exist, replace it with the embedd_tokens.weight";
    }
  }
}

void ModelInstance::LoadWeightsAndModelsMap() {
  ModelFileFormat model_file_format;
  std::vector<std::string> weights_file_list = SearchLocalPath(model_config_.path, model_file_format);
  int weight_file_size = weights_file_list.size();
  CheckTieEmbeddings(weight_file_size);

  for (std::string& file_name : weights_file_list) {
    std::shared_ptr<BaseFileTensorLoader> weights_loader = nullptr;
    if (model_file_format == SAFETENSORS) {
      weights_loader = std::make_shared<SafeTensorsLoader>(file_name);
    } else if (model_file_format == GGUF) {
      weights_loader = std::make_shared<GGUFFileTensorLoader>(file_name);
    } else {
      weights_loader = std::make_shared<PytorchFileTensorLoader>(file_name);
    }
    std::vector<std::string> weight_name_list = weights_loader->GetTensorNameList();
    std::vector<std::string> custom_name_list;
    GetCustomNameList(weight_name_list, custom_name_list, model_config_.path, model_config_.type, model_file_format);
    if (weight_file_size == 1) {
      CheckTieEmbeddings(custom_name_list);
    }

    std::vector<std::future<void>> get_weight_tasks;

    for (int worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
      get_weight_tasks.push_back(
          loader_weight_threadpool_->Submit([worker_id, this, &weights_loader, &weight_name_list, &custom_name_list]() {
            this->weights_[worker_id]->LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list);
            StreamSynchronize(this->context_->GetMemoryManageStreams()[worker_id]);
            return;
          }));
    }
    for (auto&& get_weight_task : get_weight_tasks) {
      get_weight_task.get();
    }
  }
  std::vector<std::future<void>> process_weight_tasks;
  for (int worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
    process_weight_tasks.push_back(loader_weight_threadpool_->Submit([worker_id, this]() {
      this->weights_[worker_id]->ProcessWeights();
      return;
    }));
  }
  for (auto&& process_weight_task : process_weight_tasks) {
    process_weight_task.get();
  }
}

}  // namespace ksana_llm
