/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/models/llama/llama_weight.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/memory_utils.h"

namespace numerous_llm {

/* weight_name 与存储的模型文件的关联映射关系
 * gather_embedding                   : model.wte.weight.bin
 * input_layernorm          + layer x : model.layers.x.input_layernorm.weight.bin
 * post_attention_layernorm + layer x : model.layers.x.post_attention_layernorm.weight.bin
 * attention.dense + rank r + layer x : model.layers.x.attention.dense.weight.r.bin
 * attention.query_key_value + r +  x : model.layers.x.attention.query_key_value.weight.r.bin
 * mlp.gate_proj +  rank r  + layer x : model.layers.x.mlp.gate_proj.weight.r.bin
 * mlp.up_proj   +  rank r  + layer x : model.layers.x.mlp.up_proj.weight.r.bin
 * mlp.down_proj +  rank r  + layer x : model.layers.x.mlp.down_proj.weight.r.bin
 * norm                               : model.final_layernorm.weight
 * lm_head                            : model.lm_head.weight
 */
template <typename T>
std::pair<const char*, const char*> LlamaWeight<T>::binfile_map_[] = {{"gather_embedding", "model.wte.weight.bin"},
                                                                      {"lm_head", "model.lm_head.weight.bin"},
                                                                      {"norm", "model.final_layernorm.weight.bin"}};

template <typename T>
LlamaWeight<T>::~LlamaWeight() {
  GetBlockManager()->SetDeviceId(rank_);
  for (auto& [key, tensor] : weights_map_) {
    const std::vector<int>& block_ids = tensor.GetBlockIds();
    NLLM_CHECK_WITH_INFO(block_ids.size() == 1, "Contiguous must have only one block.");
    GetBlockManager()->FreeContiguous(block_ids.front());
  }
}

template <typename T>
LlamaWeight<T>::LlamaWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context)
    : context_(context) {
  model_path_ = model_config.path;
  rank_ = rank;

  if (!LoadLlamaWeightsMap(model_config).OK()) {
    NLLM_LOG_ERROR << fmt::format("Load model config file error.");
    exit(-1);
  }
}

template <typename T>
Status LlamaWeight<T>::LoadLlamaWeightsMap(const ModelConfig& model_config) {
  DataType weight_data_type = model_config.weight_data_type;
  int head_num = model_config.head_num;
  int size_per_head = model_config.size_per_head;
  int hidden_units = head_num * size_per_head;
  int inter_size = model_config.inter_size;
  int num_layer = model_config.num_layer;
  int rotary_embedding = model_config.rotary_embedding;
  int vocab_size = model_config.vocab_size;
  int tensor_para_size = model_config.tensor_para_size;
  AddWeightTensor("gather_embedding", {vocab_size, hidden_units}, weight_data_type);
  AddWeightTensor("norm", {hidden_units}, weight_data_type);
  AddWeightTensor("lm_head", {hidden_units, vocab_size}, weight_data_type);
  for (int l = 0; l < num_layer; ++l) {
    AddWeightTensor(ConcatLayerName("input_layernorm", l), {hidden_units}, weight_data_type);
    AddWeightTensor(ConcatLayerName("post_attention_layernorm", l), {hidden_units}, weight_data_type);
    AddWeightTensor(ConcatLayerName("attention.dense", l), {hidden_units / tensor_para_size, hidden_units},
                    weight_data_type);
    AddWeightTensor(ConcatLayerName("attention.query_key_value", l),
                    {hidden_units, 3 * hidden_units / tensor_para_size}, weight_data_type);
    AddWeightTensor(ConcatLayerName("mlp.down_proj", l), {inter_size / tensor_para_size, hidden_units},
                    weight_data_type);
    AddWeightTensor(ConcatLayerName("mlp.gate_proj", l), {hidden_units, inter_size / tensor_para_size},
                    weight_data_type);
    AddWeightTensor(ConcatLayerName("mlp.up_proj", l), {hidden_units, inter_size / tensor_para_size}, weight_data_type);
  }
  return Status();
}

template <typename T>
Status LlamaWeight<T>::LoadWeightFromBin(Tensor tensor, std::string binfile) {
  if (tensor.shape.size() > 2) {
    NLLM_LOG_ERROR << fmt::format("shape should have less than two dims");
    return Status(RET_INVALID_ARGUMENT, "[ERROR] shape should have less than two dims \n");
  }

  size_t dim0 = tensor.shape[0];
  size_t dim1 = tensor.shape.size() > 1 ? tensor.shape[1] : 1;
  size_t size = dim0 * dim1;
  if (size == 0) {
    NLLM_LOG_ERROR << fmt::format("shape is zero, skip loading weight from  file {}", binfile);
    return Status(RET_INVALID_ARGUMENT, "shape is zero, skip loading weight from file " + binfile);
  }

  std::ifstream in(binfile, std::ios::in | std::ios::binary);
  if (!in.is_open()) {
    NLLM_LOG_ERROR << fmt::format("file {} cannot be opened, loading model fails!", binfile);
    return Status(RET_INVALID_ARGUMENT, "file " + binfile + " cannot be opened, loading model fails!");
  }
  in.seekg(0, in.beg);

  size_t loaded_data_size = tensor.GetTotalBytes();
  if (loaded_data_size == 0) {
    NLLM_LOG_ERROR << fmt::format("tensor total bytes = 0");
    return Status(RET_INVALID_ARGUMENT, "[ERROR] tensor " + binfile + " total bytes = 0\n");
  }

  std::vector<char> host_array(loaded_data_size);
  in.read(host_array.data(), loaded_data_size);

  size_t in_get_size = in.gcount();
  if (in_get_size != loaded_data_size) {
    NLLM_LOG_ERROR << fmt::format("file {} only has {}, but request {}, loading model fails!", binfile, in_get_size,
                                  loaded_data_size);
    return Status(RET_INVALID_ARGUMENT, "file " + binfile + " only has " + std::to_string(in_get_size) + ", but " +
                                            "request " + std::to_string(loaded_data_size) + ", loading model fails!");
  }
  T* tensor_ptr = tensor.GetPtr<T>();
  cudaMemcpy(tensor_ptr, host_array.data(), loaded_data_size, cudaMemcpyHostToDevice);
  return Status();
}

template <typename T>
Status LlamaWeight<T>::AddWeightTensor(std::string weight_name, std::vector<size_t> shapes, DataType dtype) {
  size_t length = Tensor::GetTypeSize(dtype);
  for (auto& dim : shapes) {
    length *= dim;
  }

  std::string binfile_name = GetBinfileName(weight_name);
  int block_id;
  GetBlockManager()->SetDeviceId(rank_);
  GetBlockManager()->AllocateContiguous(length, block_id);

  weights_map_.emplace(weight_name, Tensor(MEMORY_GPU, STORAGE_CONTIGUOUS, dtype, shapes, {block_id}));

  STATUS_CHECK_RETURN(LoadWeightFromBin(weights_map_[weight_name], binfile_name));
  return Status();
}

template <typename T>
std::string LlamaWeight<T>::ConcatLayerName(std::string layer_flag, int& layer_index) {
  std::string layer_name = std::to_string(layer_index) + "." + layer_flag;
  return layer_name;
}

template <typename T>
std::string LlamaWeight<T>::GetBinfileName(std::string weight_name) {
  std::string binfile_name = model_path_ + "/";
  bool match_weight_name = false;
  for (const auto& pair : binfile_map_) {
    if (std::strcmp(pair.first, weight_name.c_str()) == 0) {
      match_weight_name = true;
      binfile_name += pair.second;
      break;
    }
  }
  if (!match_weight_name) {
    binfile_name += "model.layers." + weight_name + ".weight";
    std::string weight_flag = weight_name.substr(weight_name.find_last_of('.') + 1);
    if (weight_flag != "input_layernorm" && weight_flag != "post_attention_layernorm") {
      binfile_name += "." + std::to_string(rank_);
    }
    binfile_name += ".bin";
  }
  return binfile_name;
}

template <typename T>
Tensor LlamaWeight<T>::GetModelWeights(const std::string& weight_name) {
  if (!weights_map_.count(weight_name)) {
    NLLM_LOG_WARNING << fmt::format("weight name {} not in weights map", weight_name);
    return Tensor();
  }
  return weights_map_[weight_name];
}

template class LlamaWeight<float>;
template class LlamaWeight<half>;

}  // namespace numerous_llm
