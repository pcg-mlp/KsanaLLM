#ifndef CREATE_TEST_MODEL_H
#define CREATE_TEST_MODEL_H

#include <filesystem>
#include <random>
#include "ksana_llm/utils/environment.h"

using namespace ksana_llm;

inline std::vector<uint16_t> get_random_data(int length) {
  std::vector<uint16_t> data(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  for (int i = 0; i < length; ++i) {
    float value = dis(gen);
    uint32_t f32 = *reinterpret_cast<uint32_t*>(&value);
    uint32_t f16 = ((f32 & 0x7fffffff) >> 13) - (0x38000000 >> 13);
    data[i] = static_cast<uint16_t>(f16 | (f32 & 0x80000000) >> 16);
  }
  return data;
}

inline void write_data_to_file(const std::string& saved_dir, const std::string& filename,
                               const std::vector<uint16_t>& data) {
  std::ofstream file(saved_dir + "/" + filename, std::ios::binary);
  if (file) {
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    file.close();
  } else {
    NLLM_LOG_ERROR << fmt::format("Failed to open file: {}/{}", saved_dir, filename);
  }
}

inline void create_model(ModelConfig& model_config) {
  std::string saved_dir = model_config.path;
  int hidden_units = model_config.head_num * model_config.size_per_head;
  int inter_size = model_config.inter_size;
  int num_layer = model_config.num_layer;
  int vocab_size = model_config.vocab_size;
  int tensor_para_size = model_config.tensor_para_size;
  write_data_to_file(saved_dir, "model.wte.weight.bin", get_random_data(vocab_size * hidden_units));
  write_data_to_file(saved_dir, "model.lm_head.weight.bin", get_random_data(vocab_size * hidden_units));
  write_data_to_file(saved_dir, "model.final_layernorm.weight.bin", get_random_data(hidden_units));
  for (int layer = 0; layer < num_layer; ++layer) {
    write_data_to_file(saved_dir, "model.layers." + std::to_string(layer) + ".input_layernorm.weight.bin",
                       get_random_data(hidden_units));
    write_data_to_file(saved_dir, "model.layers." + std::to_string(layer) + ".post_attention_layernorm.weight.bin",
                       get_random_data(hidden_units));
    for (int rank = 0; rank < tensor_para_size; ++rank) {
      write_data_to_file(
          saved_dir,
          "model.layers." + std::to_string(layer) + ".attention.dense.weight." + std::to_string(rank) + ".bin",
          get_random_data(hidden_units / tensor_para_size * hidden_units));
      write_data_to_file(saved_dir,
                         "model.layers." + std::to_string(layer) + ".attention.query_key_value.weight." +
                             std::to_string(rank) + ".bin",
                         get_random_data(3 * hidden_units / tensor_para_size * hidden_units));
      write_data_to_file(
          saved_dir, "model.layers." + std::to_string(layer) + ".mlp.gate_proj.weight." + std::to_string(rank) + ".bin",
          get_random_data(inter_size / tensor_para_size * hidden_units));
      write_data_to_file(
          saved_dir, "model.layers." + std::to_string(layer) + ".mlp.up_proj.weight." + std::to_string(rank) + ".bin",
          get_random_data(inter_size / tensor_para_size * hidden_units));
      write_data_to_file(
          saved_dir, "model.layers." + std::to_string(layer) + ".mlp.down_proj.weight." + std::to_string(rank) + ".bin",
          get_random_data(inter_size / tensor_para_size * hidden_units));
    }
  }
}

#endif
