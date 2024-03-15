/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/llama/llama_weight.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ksana_llm {

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

int CheckQKVWeight(const std::string& str) {
  std::string suffix = "_proj.weight";
  if (str.length() < suffix.length() + 1 || str.compare(str.length() - suffix.length(), suffix.length(), suffix)) {
    return 0;
  }
  std::vector<char> qkv_list = {'q', 'k', 'v'};
  for (int i = 0; i < 3; ++i) {
    if (str[str.length() - suffix.length() - 1] == qkv_list[i]) {
      return i + 1;
    }
  }
  return 0;
}

template <typename T>
std::vector<std::string> LlamaWeight<T>::SearchLocalPath(const std::string& model_path, bool& is_safetensors) {
  std::vector<std::string> bin_file_list;
  std::vector<std::string> safetensors_list;
  std::vector<std::string> black_list = {"training_args.bin", "optimizer.bin"};
  for (const auto& entry : std::filesystem::directory_iterator(model_path)) {
    if (entry.is_regular_file()) {
      std::string file_name = entry.path().filename().string();
      std::string extension = entry.path().extension().string();
      if (extension == ".bin") {
        bool is_black_file = false;
        for (std::string& black_file_name : black_list) {
          if (entry.path().filename().string() == black_file_name) {
            is_black_file = true;
            break;
          }
        }
        if (!is_black_file) {
          bin_file_list.emplace_back(entry.path().string());
        }
      } else if (extension == ".safetensors") {
        safetensors_list.emplace_back(entry.path().string());
      }
    }
  }
  if (safetensors_list.size() > 0) {
    is_safetensors = true;
    return safetensors_list;
  }
  return bin_file_list;
}

template <typename T>
Status LlamaWeight<T>::LoadWeightsFromFile(std::shared_ptr<BaseFileTensorLoader>& weights_loader) {
  GetBlockManager()->SetDeviceId(rank_);
  std::vector<std::string> tensor_name_list = weights_loader->GetTensorNameList();

  for (std::string& tensor_name : tensor_name_list) {
    // TODO: 扩展支持 bfp16
    /* tensor_para_offset 用于标记读取 weights_data 时是否做分卡处理:
     *     input_layernorm:          不做分卡处理
     *     post_attention_layernorm: 不做分卡处理
     *     self_attn.o_proj:         先转置,再按 axis=0 切分
     *     self_attn.qkv_proj:       先按 axis=0 切分, 再 permute((2, 0, 1))
     *     mlp.down_proj:            先转置,再按 axis=0 切分
     *     mlp.up_proj:              先按 axis=0 切分, 再转置
     *     mlp.gate_proj:            先按 axis=0 切分, 再转置
     *     lm_head:                  不做分卡处理, 需转置
     *     norm:                     不做分卡处理
     *     embedding:                不做分卡处理
     */
    bool transpose_first = false; // 使用 transpose_first 表明转置(若存在)是否在分卡(若存在)之前
    size_t tensor_para_offset = 0;
    if (tensor_name.find("_proj.weight") != std::string::npos) {
      tensor_para_offset = rank_;
      if (tensor_name.find("o_proj") != std::string::npos || tensor_name.find("down_proj") != std::string::npos) {
        transpose_first = true;
      }
    }

    // get weight's data ptr
    void* weight_ptr = weights_loader->GetTensor(tensor_name);

    // copy host data to device
    int qkv_offset;
    if (weights_map_.count(tensor_name)) {
      if (transpose_first) {
        size_t src_pitch = weights_map_[tensor_name].shape[0] * tensor_para_size_  * sizeof(T);
        size_t dst_pitch = weights_map_[tensor_name].shape[0] * sizeof(T);
        tensor_para_offset *= dst_pitch;
        CUDA_CHECK(cudaMemcpy2DAsync(weights_map_[tensor_name].GetPtr<void>(), dst_pitch, weight_ptr + tensor_para_offset,
                                     src_pitch, dst_pitch, weights_map_[tensor_name].shape[1], cudaMemcpyHostToDevice,
                                     context_->GetComputeStreams()[rank_]));
      } else {
        tensor_para_offset *= weights_map_[tensor_name].GetTotalBytes();
        CUDA_CHECK(cudaMemcpyAsync(weights_map_[tensor_name].GetPtr<void>(), weight_ptr + tensor_para_offset,
                   weights_map_[tensor_name].GetTotalBytes(), cudaMemcpyHostToDevice, context_->GetComputeStreams()[rank_]));
      }
    } else if ((qkv_offset = CheckQKVWeight(tensor_name))) {
      std::string qkv_name = tensor_name.substr(0, tensor_name.find_last_of('_') - 1) + "query_key_value.weight";
      Tensor& qkv_weight_tensor = weights_map_[qkv_name];
      size_t single_proj_size = qkv_weight_tensor.GetTotalBytes() / 3;
      size_t saved_offset = (qkv_offset - 1) * single_proj_size;
      tensor_para_offset *= single_proj_size;
      CUDA_CHECK(cudaMemcpyAsync(qkv_weight_tensor.GetPtr<void>() + saved_offset, weight_ptr + tensor_para_offset,
                                 single_proj_size, cudaMemcpyHostToDevice, context_->GetComputeStreams()[rank_]));
    } else {
      NLLM_LOG_DEBUG << "state_dict[" << tensor_name << "] will not be used";
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(context_->GetComputeStreams()[rank_]));
  return Status();
}

template <typename T>
Status LlamaWeight<T>::PermuteTensor(int hidden_units, int inter_size, int num_layer, int vocab_size) {
  GetBlockManager()->SetDeviceId(rank_);

  // permute qkv_tensor: permute((2, 0, 1))
  Tensor& last_qkv_tensor = weights_map_["empty_qkv_tensor"];
  for (size_t layer_idx = 0; layer_idx < num_layer; ++layer_idx) {
    std::string qkv_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.query_key_value.weight";
    Tensor& qkv_weight_tensor = weights_map_[qkv_name];
    InvokePermute(qkv_weight_tensor.GetPtr<void>(), last_qkv_tensor.GetPtr<void>(),
                  {3, hidden_units / tensor_para_size_, hidden_units}, {2, 0, 1}, context_->GetComputeStreams()[rank_]);
    Tensor t = last_qkv_tensor;
    last_qkv_tensor = qkv_weight_tensor;
    weights_map_[qkv_name] = t;
  }

  // permute gate_proj, up_proj, down_proj: permute(1, 0)
  Tensor& last_mlp_tensor = weights_map_["empty_mlp_tensor"];
  for (size_t layer_idx = 0; layer_idx < num_layer; ++layer_idx) {
    std::string down_proj_name = "model.layers." + std::to_string(layer_idx) + ".mlp.down_proj.weight";
    Tensor& down_weight_tensor = weights_map_[down_proj_name];
    InvokePermute(down_weight_tensor.GetPtr<void>(), last_mlp_tensor.GetPtr<void>(),
                  {hidden_units, inter_size / tensor_para_size_}, {1, 0}, context_->GetComputeStreams()[rank_]);
    Tensor t = last_mlp_tensor;
    last_mlp_tensor = down_weight_tensor;
    weights_map_[down_proj_name] = t;
    weights_map_[down_proj_name].shape = {inter_size / tensor_para_size_, hidden_units};

    std::string gate_proj_name = "model.layers." + std::to_string(layer_idx) + ".mlp.gate_proj.weight";
    Tensor& gate_weight_tensor = weights_map_[gate_proj_name];
    InvokePermute(gate_weight_tensor.GetPtr<void>(), last_mlp_tensor.GetPtr<void>(),
                  {inter_size / tensor_para_size_, hidden_units}, {1, 0}, context_->GetComputeStreams()[rank_]);
    t = last_mlp_tensor;
    last_mlp_tensor = gate_weight_tensor;
    weights_map_[gate_proj_name] = t;
    weights_map_[gate_proj_name].shape = {hidden_units, inter_size / tensor_para_size_};

    std::string up_proj_name = "model.layers." + std::to_string(layer_idx) + ".mlp.up_proj.weight";
    Tensor& up_weight_tensor = weights_map_[up_proj_name];
    InvokePermute(up_weight_tensor.GetPtr<void>(), last_mlp_tensor.GetPtr<void>(),
                  {inter_size / tensor_para_size_, hidden_units}, {1, 0}, context_->GetComputeStreams()[rank_]);
    t = last_mlp_tensor;
    last_mlp_tensor = up_weight_tensor;
    weights_map_[up_proj_name] = t;
    weights_map_[up_proj_name].shape = {hidden_units, inter_size / tensor_para_size_};
  }

  // permute o_proj: permute(1, 0)
  Tensor& last_o_proj_tensor = weights_map_["empty_o_proj_tensor"];
  for (size_t layer_idx = 0; layer_idx < num_layer; ++layer_idx) {
    std::string o_proj_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.weight";
    Tensor& o_proj_weight_tensor = weights_map_[o_proj_name];
    InvokePermute(o_proj_weight_tensor.GetPtr<void>(), last_o_proj_tensor.GetPtr<void>(),
                  {hidden_units, hidden_units / tensor_para_size_}, {1, 0}, context_->GetComputeStreams()[rank_]);
    Tensor t = last_o_proj_tensor;
    last_o_proj_tensor = o_proj_weight_tensor;
    weights_map_[o_proj_name] = t;
  }

  // permute lm_head: permute(1, 0)
  Tensor& lm_head_tensor = weights_map_["lm_head.weight"];
  Tensor& lm_head_transpose_tensor = weights_map_["empty_lm_head_tensor"];
  InvokePermute(lm_head_tensor.GetPtr<void>(), lm_head_transpose_tensor.GetPtr<void>(), {vocab_size, hidden_units},
                {1, 0}, context_->GetComputeStreams()[rank_]);
  Tensor t = lm_head_transpose_tensor;
  lm_head_transpose_tensor = lm_head_tensor;
  weights_map_["lm_head.weight"] = t;

  // Free useless tensor
  CUDA_CHECK(cudaStreamSynchronize(context_->GetComputeStreams()[rank_]));
  GetBlockManager()->SetDeviceId(rank_);
  GetBlockManager()->FreeContiguous(last_mlp_tensor.GetBlockIds()[0]);
  GetBlockManager()->FreeContiguous(last_o_proj_tensor.GetBlockIds()[0]);
  GetBlockManager()->FreeContiguous(last_qkv_tensor.GetBlockIds()[0]);
  GetBlockManager()->FreeContiguous(lm_head_transpose_tensor.GetBlockIds()[0]);

  return Status();
}

template <typename T>
Status LlamaWeight<T>::LoadLlamaWeightsMap(const ModelConfig& model_config) {
  DataType weight_data_type = model_config.weight_data_type;
  int head_num = model_config.head_num;
  int hidden_units = model_config.hidden_units;
  int inter_size = model_config.inter_size;
  int num_layer = model_config.num_layer;
  int rotary_embedding = model_config.rotary_embedding;
  int vocab_size = model_config.vocab_size;
  tensor_para_size_ = model_config.tensor_para_size;

  AddWeightTensor("model.embed_tokens.weight", {vocab_size, hidden_units}, weight_data_type);
  AddWeightTensor("model.norm.weight", {hidden_units}, weight_data_type);
  AddWeightTensor("empty_qkv_tensor", {hidden_units, 3 * hidden_units / tensor_para_size_}, weight_data_type);
  AddWeightTensor("empty_mlp_tensor", {hidden_units, inter_size / tensor_para_size_}, weight_data_type);
  AddWeightTensor("empty_lm_head_tensor", {hidden_units, vocab_size}, weight_data_type);
  AddWeightTensor("empty_o_proj_tensor", {hidden_units / tensor_para_size_, hidden_units}, weight_data_type);
  for (int l = 0; l < num_layer; ++l) {
    AddWeightTensor(ConcatLayerName("input_layernorm", l), {hidden_units}, weight_data_type);
    AddWeightTensor(ConcatLayerName("post_attention_layernorm", l), {hidden_units}, weight_data_type);
    AddWeightTensor(ConcatLayerName("mlp.down_proj", l), {inter_size / tensor_para_size_, hidden_units},
                    weight_data_type);
    AddWeightTensor(ConcatLayerName("mlp.gate_proj", l), {hidden_units, inter_size / tensor_para_size_},
                    weight_data_type);
    AddWeightTensor(ConcatLayerName("mlp.up_proj", l), {hidden_units, inter_size / tensor_para_size_},
                    weight_data_type);
    AddWeightTensor(ConcatLayerName("self_attn.o_proj", l), {hidden_units / tensor_para_size_, hidden_units},
                    weight_data_type);
    AddWeightTensor(ConcatLayerName("self_attn.query_key_value", l),
                    {hidden_units, 3 * hidden_units / tensor_para_size_}, weight_data_type);
  }
  AddWeightTensor("lm_head.weight", {vocab_size, hidden_units}, weight_data_type);

  bool is_safetensors = false;
  std::vector<std::string> weights_file_list = SearchLocalPath(model_path_, is_safetensors);
  for (std::string& file_name : weights_file_list) {
    std::shared_ptr<BaseFileTensorLoader> weights_loader = nullptr;
    if (is_safetensors) {
      weights_loader = std::make_shared<SafeTensorsLoader>(file_name);
    } else {
      weights_loader = std::make_shared<PytorchFileTensorLoader>(file_name);
    }
    LoadWeightsFromFile(weights_loader);
  }
  PermuteTensor(hidden_units, inter_size, num_layer, vocab_size);
  return Status();
}

template <typename T>
bool LlamaWeight<T>::IsLoaded() {
  return weights_had_loaded_;
}

template <typename T>
Status LlamaWeight<T>::AddWeightTensor(std::string weight_name, std::vector<size_t> shapes, DataType dtype) {
  size_t length = Tensor::GetTypeSize(dtype);
  for (auto& dim : shapes) {
    length *= dim;
  }

  int block_id;
  GetBlockManager()->SetDeviceId(rank_);
  GetBlockManager()->AllocateContiguous(length, block_id);

  weights_map_.emplace(weight_name, Tensor(MEMORY_GPU, STORAGE_CONTIGUOUS, dtype, shapes, {block_id}));
  return Status();
}

template <typename T>
std::string LlamaWeight<T>::ConcatLayerName(std::string layer_flag, int& layer_index) {
  std::string layer_name = "model.layers." + std::to_string(layer_index) + "." + layer_flag + ".weight";
  return layer_name;
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

}  // namespace ksana_llm
