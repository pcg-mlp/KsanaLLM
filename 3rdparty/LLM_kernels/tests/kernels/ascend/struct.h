#include <memory>
#include <string>
#include "aclnn/acl_meta.h"

namespace AclnnLlama {

struct ModelConfig {
 public:
  std::string model_path;
  int num_layers;
  int num_heads;
  int head_dims;
  int hidden_size;
  int64_t vocab_size;
  int64_t ffn_intermediate_size;
  int64_t max_tokens_num;
  float rope_theta;
  float rope_scaling_factor;
};

class TensorWeight {
 public:
  explicit TensorWeight(const std::vector<int64_t>& shape, aclDataType dtype, aclFormat fmt);

  void CreateAclTensor();

  aclTensor* GetAclTensor() {
    if (acl_tensor_ == nullptr) {
      CreateAclTensor();
    }
    return acl_tensor_;
  }

  ~TensorWeight();

 public:
  std::vector<int64_t> shape_;
  aclDataType dtype_;
  aclFormat fmt_ = aclFormat::ACL_FORMAT_ND;

  void* data_dev_ = nullptr;
  aclTensor* acl_tensor_ = nullptr;
};

using TensorWeightPtr = std::unique_ptr<TensorWeight>;

class DecoderLayerWeight {
 public:
  TensorWeightPtr q_proj = nullptr;
  TensorWeightPtr k_proj = nullptr;
  TensorWeightPtr v_proj = nullptr;
  TensorWeightPtr qkv_proj = nullptr;
  TensorWeightPtr o_proj = nullptr;
  TensorWeightPtr gate_proj = nullptr;
  TensorWeightPtr up_proj = nullptr;
  TensorWeightPtr down_proj = nullptr;
  TensorWeightPtr attn_rms = nullptr;
  TensorWeightPtr attn_post_rms = nullptr;
  void* total_key_cache = nullptr;
  void* total_val_cache = nullptr;
};

class LlamaWeight {
 public:
  std::vector<DecoderLayerWeight> decoder_layer_weight;
  TensorWeightPtr lm_head_rms = nullptr;
  void* input_ids_dev = nullptr;
  TensorWeightPtr emb_weights = nullptr;
  TensorWeightPtr lm_head = nullptr;
  int64_t posIndex;
};

}  // namespace AclnnLlama
