/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/models/llama/llama_weight.h"

namespace numerous_llm {

LlamaWeight::~LlamaWeight() {
  // TODO: 引用计数
}

LlamaWeight::LlamaWeight(const ModelConfig& model_config) {
  model_path_ = model_config.path;
  weight_data_type_ = model_config.weight_data_type;
  head_num_ = model_config.head_num;
  size_per_head_ = model_config.size_per_head;
  hidden_units_ = head_num_ * size_per_head_;
  inter_size_ = model_config.inter_size;
  num_layer_ = model_config.num_layer;
  rotary_embedding_ = model_config.rotary_embedding;
  vocab_size_ = model_config.vocab_size;
  rank_ = model_config.rank;
  tensor_para_size_ = model_config.tensor_para_size;

  if (!LoadLlamaWeightsMap().OK()) {
    printf("ERROR");
    exit(-1);
  }
}

Status LlamaWeight::LoadLlamaWeightsMap() {
  AddWeightTensor("gather_embedding", {vocab_size_,  hidden_units_}, weight_data_type_);
  AddWeightTensor("norm", {hidden_units_}, weight_data_type_);
  AddWeightTensor("lm_head", {vocab_size_,  hidden_units_}, weight_data_type_);
  for (int l = 0; l < num_layer_; ++l) {
    AddWeightTensor(ConcatLayerName("input_layernorm", l), {hidden_units_}, weight_data_type_);
    AddWeightTensor(ConcatLayerName("post_attention_layernorm", l), {hidden_units_}, weight_data_type_);
    AddWeightTensor(ConcatLayerName("attention.dense", l),
                  {hidden_units_ / tensor_para_size_, hidden_units_}, weight_data_type_);
    AddWeightTensor(ConcatLayerName("attention.query_key_value", l),
                  {hidden_units_, 3 * hidden_units_ / tensor_para_size_}, weight_data_type_);
    AddWeightTensor(ConcatLayerName("mlp.down_proj", l),
                  {inter_size_ / tensor_para_size_, hidden_units_}, weight_data_type_);
    AddWeightTensor(ConcatLayerName("mlp.gate_proj", l),
                  {hidden_units_, inter_size_ / tensor_para_size_}, weight_data_type_);
    AddWeightTensor(ConcatLayerName("mlp.up_proj", l),
                  {hidden_units_, inter_size_ / tensor_para_size_}, weight_data_type_);
  }
  return Status();
}

Status LlamaWeight::LoadWeightFromBin(Tensor tensor, std::string binfile) {
  if (tensor.shape.size() > 2) {
    printf("[ERROR] shape should have less than two dims \n");
    return Status(RET_INVALID_ARGUMENT, "[ERROR] shape should have less than two dims \n");
  }

  size_t dim0 = tensor.shape[0];
  size_t dim1 = tensor.shape.size() > 1 ? tensor.shape[1] : 1;
  size_t size = dim0 * dim1;
  if (size == 0) {
    printf("[WARNING] shape is zero, skip loading weight from  file %s\n", binfile.c_str());
    return Status(RET_INVALID_ARGUMENT, "shape is zero, skip loading weight from file  " + binfile);
  }

  std::ifstream in(binfile, std::ios::in | std::ios::binary);
  if (!in.is_open()) {
    printf("file %s cannot be opened, loading model fails! \n", binfile.c_str());
    return Status(RET_INVALID_ARGUMENT, "file " + binfile + " cannot be opened, loading model fails!");
  }
  in.seekg(0, in.beg);

  size_t loaded_data_size = tensor.GetTotalBytes();
  if (loaded_data_size == 0) {
    printf("[ERROR] tensor total bytes = 0\n");
    return Status(RET_INVALID_ARGUMENT, "[ERROR] tensor " + binfile + " total bytes = 0\n");
  }

  std::vector<char> host_array(loaded_data_size);
  in.read(host_array.data(), loaded_data_size);

  size_t in_get_size = in.gcount();
  if (in_get_size != loaded_data_size) {
    printf("[WARNING] file %s only has %ld, but request %ld, loading model fails! \n",
           binfile.c_str(), in_get_size, loaded_data_size);
    return Status(RET_INVALID_ARGUMENT, "file " + binfile + " only has " + std::to_string(in_get_size) + ", but "
                + "request " + std::to_string(loaded_data_size) + ", loading model fails!");
  }
  void* tensor_ptr = tensor.GetPtr<void>();
  cudaMemcpy(tensor_ptr, host_array.data(), loaded_data_size, cudaMemcpyHostToDevice);
  return Status();
}

Status LlamaWeight::AddWeightTensor(std::string weight_name, std::vector<size_t> shapes, DataType dtype) {
  size_t length = Tensor::GetTypeSize(dtype);
  for (auto& dim : shapes) {
    length *= dim;
  }

  std::string binfile_name = GetBinfileName(weight_name);
  int block_id;
  // printf("%s try to allocate %d\n", weight_name.c_str(), length);
  STATUS_CHECK_RETURN(DEVICE_EXECUTE(rank_, BlockManager, AllocateContiguous, length, block_id));
  weights_map_.emplace(weight_name, Tensor(MEMORY_GPU, STORAGE_CONTIGUOUS, dtype, shapes, {block_id}));

  STATUS_CHECK_RETURN(LoadWeightFromBin(weights_map_[weight_name], binfile_name));
  return Status();
}

std::string LlamaWeight::ConcatLayerName(std::string layer_flag, int& layer_index) {
  std::string layer_name = std::to_string(layer_index) + "." + layer_flag;
  return layer_name;
}

std::string LlamaWeight::GetBinfileName(std::string weight_name) {
  std::string binfile_name = model_path_ + "/";
  if (binfile_map_.count(weight_name)) {
    binfile_name += binfile_map_[weight_name];
  } else {
    binfile_name += "model.layers." + weight_name + ".weight";
    std::string weight_flag = weight_name.substr(weight_name.find_last_of('.') + 1);
    if (weight_flag != "input_layernorm" && weight_flag != "post_attention_layernorm") {
      binfile_name +=  "." + std::to_string(rank_);
    }
    binfile_name += ".bin";
  }
  return binfile_name;
}

Tensor LlamaWeight::GetModelWeights(std::string& weight_name) {
  if (!weights_map_.count(weight_name)) {
    return Tensor();
  }
  return weights_map_[weight_name];
}

}  // namespace numerous_llm
