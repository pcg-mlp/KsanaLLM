/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/llama/llama_weight.h"
#include "ksana_llm/utils/common_device.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

#include "ksana_llm/kernels/cast.h"
#include "ksana_llm/kernels/permute.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/nn/functional/normalization.h>

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
    const int block_id = tensor.GetBlockId();
    GetBlockManager()->FreeContiguous(block_id);
  }
}

template <typename T>
LlamaWeight<T>::LlamaWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context)
    : context_(context), model_config_(model_config) {
  model_path_ = model_config.path;
  rank_ = rank;
  if (!LoadLlamaWeightsMap(model_config).OK()) {
    NLLM_LOG_ERROR << fmt::format("Load model config file error.");
    exit(-1);
  }
}

int CheckQKVWeight(const std::string& str) {
  std::string suffix = "_proj.weight";
  if (str.find("_proj.bias") != std::string::npos) {
    suffix = "_proj.bias";
  }
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
      if (file_name.length() >= 6 && file_name.compare(0, 6, ".etag.") == 0) {
        // skip etag file
        continue;
      } else if (extension == ".bin") {
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
  for (size_t idx = 0; idx < tensor_name_list.size(); ++idx) {
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
    std::string& tensor_name = tensor_name_list[idx];
    bool transpose_first = false;  // 使用 transpose_first 表明转置(若存在)是否在分卡(若存在)之前
    size_t tensor_para_offset = 0;
    std::vector<size_t> weight_shape = weights_loader->GetTensorShape(tensor_name);
    if (tensor_name.find("_proj.weight") != std::string::npos || tensor_name.find("_proj.bias") != std::string::npos ||
        tensor_name.find("self_attn.W_pack") != std::string::npos ||
        tensor_name.find("embed_tokens") != std::string::npos ||
        tensor_name.find("lm_head.weight") != std::string::npos) {
      tensor_para_offset = rank_;
      if (tensor_name.find("o_proj") != std::string::npos || tensor_name.find("down_proj") != std::string::npos ||
          tensor_name.find("embed_tokens") != std::string::npos) {
        transpose_first = true;
        weight_shape[1] /= tensor_para_size_;
      } else {
        weight_shape[0] = DivRoundUp(weight_shape[0], tensor_para_size_);
      }
    }

    // get weight's data ptr
    void* weight_ptr;
    size_t weight_size;
    std::tie(weight_ptr, weight_size) = weights_loader->GetTensor(tensor_name);
    DataType weight_data_type = weights_loader->GetTensorDataType(tensor_name);

    // copy host data to device
    int qkv_offset;
    if ((qkv_offset = CheckQKVWeight(tensor_name))) {
      bool is_bias = (tensor_name.find("_proj.bias") != std::string::npos);
      std::string qkv_name = tensor_name.substr(0, tensor_name.find_last_of('_') - 1) + "query_key_value" +
                             (is_bias ? ".bias" : ".weight");
      if (!weights_map_.count(qkv_name)) {
        weight_shape.insert(weight_shape.begin(), 3);
        AddWeightTensor(qkv_name, weight_shape, weight_data_type_);
      }
      weights_data_type_map_[qkv_name] = weight_data_type;
      Tensor& qkv_weight_tensor = weights_map_[qkv_name];
      size_t single_proj_size = qkv_weight_tensor.GetTotalBytes() / 3;
      size_t saved_offset = (qkv_offset - 1) * single_proj_size;
      tensor_para_offset *= single_proj_size;
      GetBlockManager()->SetDeviceId(rank_);
      MemcpyAsync(qkv_weight_tensor.GetPtr<void>() + saved_offset, weight_ptr + tensor_para_offset, single_proj_size,
                  MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
    } else if (tensor_name.find("_proj.weight") != std::string::npos ||
               tensor_name.find("_proj.bias") != std::string::npos ||
               tensor_name.find("_layernorm.weight") != std::string::npos || tensor_name == "lm_head.weight" ||
               tensor_name == "model.norm.weight" || tensor_name == "model.embed_tokens.weight") {
      AddWeightTensor(tensor_name, weight_shape, weight_data_type_);
      weights_data_type_map_[tensor_name] = weight_data_type;
      if (transpose_first) {
        size_t src_pitch = weights_map_[tensor_name].shape[1] * tensor_para_size_ * sizeof(T);
        size_t dst_pitch = weights_map_[tensor_name].shape[1] * sizeof(T);
        tensor_para_offset *= dst_pitch;
        GetBlockManager()->SetDeviceId(rank_);
        Memcpy2DAsync(weights_map_[tensor_name].GetPtr<void>(), dst_pitch, weight_ptr + tensor_para_offset, src_pitch,
                      dst_pitch, weights_map_[tensor_name].shape[0], MEMCPY_HOST_TO_DEVICE,
                      context_->GetMemoryManageStreams()[rank_]);
      } else {
        tensor_para_offset *= weights_map_[tensor_name].GetTotalBytes();
        GetBlockManager()->SetDeviceId(rank_);
        size_t sub_bytes = 0;
        if (rank_ == (tensor_para_size_ - 1) && tensor_name.find("lm_head.weight") != std::string::npos) {
          sub_bytes = weights_map_[tensor_name].GetTotalBytes() * tensor_para_size_ - weight_size;
        }
        MemcpyAsync(weights_map_[tensor_name].GetPtr<void>(), weight_ptr + tensor_para_offset,
                    weights_map_[tensor_name].GetTotalBytes() - sub_bytes, MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[rank_]);
      }
    } else if (tensor_name.find("self_attn.W_pack.weight") != std::string::npos) {
      bool is_bias = (tensor_name.find(".bias") != std::string::npos);
      std::string qkv_name = tensor_name.substr(0, tensor_name.find_last_of('_') - 1) + "query_key_value" +
                             (is_bias ? ".bias" : ".weight");
      weights_data_type_map_[qkv_name] = weight_data_type;
      if (!weights_map_.count(qkv_name)) {
        weight_shape.insert(weight_shape.begin(), 3);
        weight_shape[1] /= 3;
        AddWeightTensor(qkv_name, weight_shape, weight_data_type_);
      }
      Tensor& qkv_weight_tensor = weights_map_[qkv_name];
      size_t src_pitch = weight_shape[1] * weight_shape[2] * tensor_para_size_ * sizeof(T);
      size_t dst_pitch = weight_shape[1] * weight_shape[2] * sizeof(T);
      tensor_para_offset *= dst_pitch;
      GetBlockManager()->SetDeviceId(rank_);
      Memcpy2DAsync(qkv_weight_tensor.GetPtr<void>(), dst_pitch, weight_ptr + tensor_para_offset, src_pitch, dst_pitch,
                    weight_shape[0], MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
    } else {
      NLLM_LOG_DEBUG << "state_dict[" << tensor_name << "] will not be used";
    }
  }
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
    Permute(qkv_weight_tensor, last_qkv_tensor, {2, 0, 1}, context_->GetMemoryManageStreams()[rank_]);
    Tensor t = last_qkv_tensor;
    last_qkv_tensor = qkv_weight_tensor;
    t.shape = {qkv_weight_tensor.shape[2], qkv_weight_tensor.shape[0] * qkv_weight_tensor.shape[1]};
    weights_map_[qkv_name] = t;
  }

  // permute gate_proj, up_proj, down_proj: permute(1, 0)
  Tensor& last_mlp_tensor = weights_map_["empty_mlp_tensor"];
  for (size_t layer_idx = 0; layer_idx < num_layer; ++layer_idx) {
    std::string down_proj_name = "model.layers." + std::to_string(layer_idx) + ".mlp.down_proj.weight";
    Tensor& down_weight_tensor = weights_map_[down_proj_name];
    Permute(down_weight_tensor, last_mlp_tensor, {1, 0}, context_->GetMemoryManageStreams()[rank_]);
    Tensor t = last_mlp_tensor;
    last_mlp_tensor = down_weight_tensor;
    t.shape = {down_weight_tensor.shape[1], down_weight_tensor.shape[0]};
    weights_map_[down_proj_name] = t;

    std::string gate_proj_name = "model.layers." + std::to_string(layer_idx) + ".mlp.gate_proj.weight";
    Tensor& gate_weight_tensor = weights_map_[gate_proj_name];
    Permute(gate_weight_tensor, last_mlp_tensor, {1, 0}, context_->GetMemoryManageStreams()[rank_]);
    t = last_mlp_tensor;
    last_mlp_tensor = gate_weight_tensor;
    t.shape = {gate_weight_tensor.shape[1], gate_weight_tensor.shape[0]};
    weights_map_[gate_proj_name] = t;

    std::string up_proj_name = "model.layers." + std::to_string(layer_idx) + ".mlp.up_proj.weight";
    Tensor& up_weight_tensor = weights_map_[up_proj_name];
    Permute(up_weight_tensor, last_mlp_tensor, {1, 0}, context_->GetMemoryManageStreams()[rank_]);
    t = last_mlp_tensor;
    last_mlp_tensor = up_weight_tensor;
    t.shape = {up_weight_tensor.shape[1], up_weight_tensor.shape[0]};
    weights_map_[up_proj_name] = t;
  }

  // permute o_proj: permute(1, 0)
  Tensor& last_o_proj_tensor = weights_map_["empty_o_proj_tensor"];
  for (size_t layer_idx = 0; layer_idx < num_layer; ++layer_idx) {
    std::string o_proj_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.weight";
    Tensor& o_proj_weight_tensor = weights_map_[o_proj_name];
    Permute(o_proj_weight_tensor, last_o_proj_tensor, {1, 0}, context_->GetMemoryManageStreams()[rank_]);
    Tensor t = last_o_proj_tensor;
    last_o_proj_tensor = o_proj_weight_tensor;
    t.shape = {o_proj_weight_tensor.shape[1], o_proj_weight_tensor.shape[0]};
    weights_map_[o_proj_name] = t;
  }

  // permute lm_head: permute(1, 0)
  Tensor& lm_head_tensor = weights_map_["lm_head.weight"];
  Tensor& lm_head_transpose_tensor = weights_map_["empty_lm_head_tensor"];
  Permute(lm_head_tensor, lm_head_transpose_tensor, {1, 0}, context_->GetMemoryManageStreams()[rank_]);
  Tensor t = lm_head_transpose_tensor;
  lm_head_transpose_tensor = lm_head_tensor;
  t.shape = {lm_head_tensor.shape[1], lm_head_tensor.shape[0]};
  weights_map_["lm_head.weight"] = t;

  // Free useless tensor
  GetBlockManager()->SetDeviceId(rank_);
  GetBlockManager()->FreeContiguous(last_mlp_tensor.GetBlockId());
  GetBlockManager()->FreeContiguous(last_o_proj_tensor.GetBlockId());
  GetBlockManager()->FreeContiguous(last_qkv_tensor.GetBlockId());
  GetBlockManager()->FreeContiguous(lm_head_transpose_tensor.GetBlockId());
  weights_map_.erase("empty_qkv_tensor");
  weights_map_.erase("empty_mlp_tensor");
  weights_map_.erase("empty_lm_head_tensor");
  weights_map_.erase("empty_o_proj_tensor");
  return Status();
}

template <typename T>
Status LlamaWeight<T>::LoadLlamaWeightsMap(const ModelConfig& model_config) {
  weight_data_type_ = model_config.weight_data_type;
  int head_num = model_config.head_num;
  int hidden_units = model_config.hidden_units;
  int inter_size = model_config.inter_size;
  int num_layer = model_config.num_layer;
  int rotary_embedding = model_config.rotary_embedding;
  int vocab_size = model_config.vocab_size;
  model_name_ = model_config.name;
  tensor_para_size_ = model_config.tensor_para_size;

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
    StreamSynchronize(context_->GetComputeStreams()[rank_]);
  }

  CreateTensorWithSameShape("model.layers.0.self_attn.o_proj.weight", "empty_o_proj_tensor");
  CreateTensorWithSameShape("model.layers.0.self_attn.query_key_value.weight", "empty_qkv_tensor");
  CreateTensorWithSameShape("model.layers.0.mlp.down_proj.weight", "empty_mlp_tensor");
  CreateTensorWithSameShape("lm_head.weight", "empty_lm_head_tensor");

  PermuteTensor(hidden_units, inter_size, num_layer, vocab_size);

  // Convert BFP16 to FP16
  for (auto& data_type_iter : weights_data_type_map_) {
    if (data_type_iter.second == TYPE_BF16) {
      Tensor& tensor = weights_map_[data_type_iter.first];
      tensor.dtype = DataType::TYPE_BF16;
      GetBlockManager()->SetDeviceId(rank_);
      CastInplace(tensor, DataType::TYPE_FP16, context_->GetMemoryManageStreams()[rank_]);
      tensor.dtype = TYPE_FP16;
      // We use vocab_size to determine whether it is the Baichuan2 model.
      // If vocab_size is equal to 125,696, then it is the Baichuan2 model.
      // And Unlike Baichuan1, the Baichuan2 model requires normalizing the head weights. Refer to
      // https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/blob/84603cde5ebffb6084e476cfaeceaf0b8b91fe54/modeling_baichuan.py#L508
      if (model_config_.type == "baichuan" && data_type_iter.first == "lm_head.weight" && vocab_size == 125696) {
        StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
        auto options = torch::TensorOptions().device(torch::kCUDA, rank_).dtype(torch::kFloat16);
        torch::Tensor in = torch::from_blob(tensor.GetPtr<void>(), {tensor.shape[0], tensor.shape[1]}, options);
        auto out = torch::nn::functional::normalize(in, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));
        MemcpyAsync(tensor.GetPtr<void>(), out.data_ptr(), sizeof(T) * tensor.shape[0] * tensor.shape[1],
                    MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
      }
    }
  }
  StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
  return Status();
}

template <typename T>
bool LlamaWeight<T>::IsLoaded() {
  return weights_had_loaded_;
}

template <typename T>
Status LlamaWeight<T>::AddWeightTensor(std::string weight_name, std::vector<size_t> shapes, DataType dtype) {
  size_t length = GetTypeSize(dtype);
  for (auto& dim : shapes) {
    length *= dim;
  }
  int block_id;
  GetBlockManager()->SetDeviceId(rank_);
  GetBlockManager()->AllocateContiguous(length, block_id);

  weights_map_.emplace(weight_name, Tensor(MemoryDevice::MEMORY_DEVICE, dtype, shapes, block_id));
  return Status();
}

template <typename T>
Status LlamaWeight<T>::CreateTensorWithSameShape(const std::string& origin_tensor_name,
                                                 const std::string& copy_tensor_name) {
  if (!weights_map_.count(origin_tensor_name)) {
    NLLM_LOG_ERROR << fmt::format("Create tensor {} faild: tensor {} not in weights map", copy_tensor_name,
                                  origin_tensor_name);
    exit(-1);
  }
  Tensor& origin_tensor = weights_map_[origin_tensor_name];
  AddWeightTensor(copy_tensor_name, origin_tensor.shape, origin_tensor.dtype);
  return Status();
}

template <typename T>
std::string LlamaWeight<T>::ConcatLayerName(std::string layer_flag, int& layer_index, bool is_bias) {
  std::string layer_name =
      "model.layers." + std::to_string(layer_index) + "." + layer_flag + (is_bias ? ".bias" : ".weight");
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
#ifdef ENABLE_CUDA
template class LlamaWeight<half>;
#endif
#ifdef ENABLE_ACL
template class LlamaWeight<aclFloat16>;
#endif

}  // namespace ksana_llm
