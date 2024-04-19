#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "struct.h"

#include "common.h"

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

#include "csrc/kernels/ascend/activation/activation.h"
#include "csrc/kernels/ascend/argmax/argmax.h"
#include "csrc/kernels/ascend/attention/attention.h"
#include "csrc/kernels/ascend/cat/cat.h"
#include "csrc/kernels/ascend/elementwise/elementwise.h"
#include "csrc/kernels/ascend/embedding/embedding.h"
#include "csrc/kernels/ascend/gather/gather.h"
#include "csrc/kernels/ascend/layernorm/layernorm.h"
#include "csrc/kernels/ascend/matmul/matmul.h"
#include "csrc/kernels/ascend/pointwise/pointwise.h"
#include "csrc/kernels/ascend/transpose/transpose.h"

#define SUCCESS 0
#define FAILED 1

using namespace AclnnLlama;
constexpr int64_t max_tokens_num = 2048;
constexpr int max_prompt_len = 32;
constexpr int max_ans_len = 32;
const float rope_theta = 10000.0;
const float rope_scaling_factor = 1.0;

int bs = 1;
int hidden_size = 32 * 128;
constexpr int num_heads = 32;
constexpr int num_layers = 32;
constexpr int head_dims = 128;
constexpr int vocab_size = 32001;
constexpr int64_t ffn_intermediate_size = 11008;

aclDataType dtype = aclDataType::ACL_FLOAT16;
aclFormat fmt = aclFormat::ACL_FORMAT_ND;

static std::unordered_map<int, std::string> INT2STR = {
    {0, "0"},   {1, "1"},   {2, "2"},   {3, "3"},   {4, "4"},   {5, "5"},   {6, "6"},   {7, "7"},
    {8, "8"},   {9, "9"},   {10, "10"}, {11, "11"}, {12, "12"}, {13, "13"}, {14, "14"}, {15, "15"},
    {16, "16"}, {17, "17"}, {18, "18"}, {19, "19"}, {20, "20"}, {21, "21"}, {22, "22"}, {23, "23"},
    {24, "24"}, {25, "25"}, {26, "26"}, {27, "27"}, {28, "28"}, {29, "29"}, {30, "30"}, {31, "31"}};

using namespace llm_kernels::utils;

TensorWeight::TensorWeight(const std::vector<int64_t>& shape, aclDataType dtype, aclFormat fmt)
    : shape_(shape), dtype_(dtype), fmt_(fmt) {}

void TensorWeight::CreateAclTensor() {
  if (data_dev_ == nullptr) {
    auto size = GetShapeSize(shape_) * DT2LONG.at(dtype_);
    ACL_CHECK_RET(aclrtMalloc(&data_dev_, size, ACL_MEM_MALLOC_NORMAL_ONLY));
  }
  CreateAclTensorWithData(shape_, &data_dev_, dtype_, fmt_, &acl_tensor_);
}

TensorWeight::~TensorWeight() {
  if (acl_tensor_ != nullptr) {
    aclDestroyTensor(acl_tensor_);
  }
  if (data_dev_ != nullptr) {
    aclrtFree(data_dev_);
    data_dev_ = nullptr;
  }
}

void LmHead(aclTensor** gatherOutput, const int bs, const int64_t seq_len, const int64_t hidden_size,
            const aclTensor* lm_head_rms_weight, const aclTensor* lm_head_weight, void** tmpDev1, void** tmpDev2,
            void** outDev, aclrtStream& stream) {
  // gatherOutput - > rmsnormOutput
  aclTensor* rmsnormOutput = nullptr;
  std::vector<int64_t> rmsnormOutputShape = {bs, seq_len, hidden_size};
  CreateAclTensorWithData(rmsnormOutputShape, tmpDev1, dtype, fmt, &rmsnormOutput);
  llm_kernels::ascend::RMSLayerNorm(*gatherOutput, lm_head_rms_weight, &rmsnormOutput, stream,
                                    llm_kernels::utils::GetTestWorkSpaceFunc);
  aclDestroyTensor(*gatherOutput);

  // rmsnormOutput - > matmul_output
  std::vector<int64_t> matmul_outputShape = {bs, seq_len, vocab_size};
  aclTensor* matmul_output = nullptr;
  CreateAclTensorWithData(matmul_outputShape, tmpDev2, dtype, fmt, &matmul_output);
  int mm_type = 0;
  llm_kernels::ascend::MatMul(rmsnormOutput, lm_head_weight, mm_type, &matmul_output, stream,
                              llm_kernels::utils::GetTestWorkSpaceFunc);
  aclDestroyTensor(rmsnormOutput);

  // matmul_output - > castOutput
  std::vector<int64_t> castOutputShape = {bs, seq_len, vocab_size};
  aclTensor* castOutput = nullptr;
  CreateAclTensorWithData(castOutputShape, tmpDev1, aclDataType::ACL_FLOAT, fmt, &castOutput);
  llm_kernels::ascend::Cast(matmul_output, aclDataType::ACL_FLOAT, &castOutput, stream,
                            llm_kernels::utils::GetTestWorkSpaceFunc);
  aclDestroyTensor(matmul_output);

  // castOutput - > gather1Output
  std::vector<int64_t> gather1IndexShape = {1};
  std::vector<int64_t> gather1OutputShape = {bs, 1, vocab_size};
  void* gather1IndexDev = nullptr;
  aclTensor* gather1Index = nullptr;
  aclTensor* gather1Output = nullptr;
  int64_t gather1IndexData = 15;  // magic number, why??
  CreateAclTensor(gather1IndexData, gather1IndexShape, &gather1IndexDev, aclDataType::ACL_INT64, fmt, &gather1Index);
  CreateAclTensorWithData(gather1OutputShape, tmpDev2, aclDataType::ACL_FLOAT, fmt, &gather1Output);
  int64_t gather1Dim = 1;
  llm_kernels::ascend::Gather(castOutput, gather1Dim, gather1Index, &gather1Output, stream,
                              llm_kernels::utils::GetTestWorkSpaceFunc);
  aclDestroyTensor(castOutput);
  aclDestroyTensor(gather1Index);
  aclrtFree(gather1IndexDev);
  gather1IndexDev = nullptr;

  // gather1Output - > argMaxOutput
  std::vector<int64_t> argMaxOutputShape = {bs, 1, 1};
  void* argMaxOutputDev = nullptr;
  aclTensor* argMaxOutput = nullptr;
  CreateAclTensor(argMaxOutputShape, &argMaxOutputDev, aclDataType::ACL_INT64, fmt, &argMaxOutput);
  int64_t argMaxDim = -1;
  bool argMaxKeepdim = true;
  llm_kernels::ascend::ArgMax(gather1Output, argMaxDim, argMaxKeepdim, &argMaxOutput, stream,
                              llm_kernels::utils::GetTestWorkSpaceFunc);
  aclDestroyTensor(gather1Output);

  // argMaxOutput(reshape1Input) - > reshape1Output
  std::vector<int64_t> reshape1OutputShape = {bs, 1};
  aclTensor* reshape1Input = nullptr;
  aclTensor* reshape1Output = nullptr;
  CreateAclTensorWithData(reshape1OutputShape, &argMaxOutputDev, aclDataType::ACL_INT64, fmt, &reshape1Input);
  CreateAclTensorWithData(reshape1OutputShape, outDev, aclDataType::ACL_INT64, fmt, &reshape1Output);
  llm_kernels::ascend::Transpose(reshape1Input, &reshape1Output, stream, llm_kernels::utils::GetTestWorkSpaceFunc);
  // PrintTensor(reshape1Output, stream, "reshape1Output");

  aclDestroyTensor(reshape1Input);
  aclDestroyTensor(argMaxOutput);
  aclDestroyTensor(reshape1Output);
  aclrtFree(argMaxOutputDev);
  argMaxOutputDev = nullptr;
}

// output, tmp[2]
void LlamaAttn(aclTensor* rmsnormOutput, int seq_len, int posIndex, DecoderLayerInfo& decoderLayerInfo,
               std::vector<void*>& tmp_buffers, aclTensor** oproj_output, const bool is_context_stage,
               std::unique_ptr<llm_kernels::ascend::FlashAttentionACL>& flash_attn, aclrtStream& stream) {
  /// matmul
  aclTensor* matmulQKVOutput = nullptr;
  std::vector<int64_t> matmulQKVOutputShape = {bs, seq_len, 3 * hidden_size};
  CreateAclTensorWithData(matmulQKVOutputShape, &tmp_buffers[0], dtype, fmt, &matmulQKVOutput);
  int mm_type = 0;
  llm_kernels::ascend::MatMul(rmsnormOutput, decoderLayerInfo.qkv_proj->GetAclTensor(), mm_type, &matmulQKVOutput,
                              stream, llm_kernels::utils::GetTestWorkSpaceFunc);
  aclDestroyTensor(rmsnormOutput);

  // / flash atten
  // 1-4
  // TODO: generate sin and code, when rope init
  aclTensor* attnOutput = nullptr;
  flash_attn->Forward(matmulQKVOutput, posIndex, &decoderLayerInfo.total_key_cache, &decoderLayerInfo.total_val_cache,
                      tmp_buffers, &attnOutput, is_context_stage, stream, llm_kernels::utils::GetTestWorkSpaceFunc);
  aclDestroyTensor(matmulQKVOutput);

  // o_proj matmul: reshapeOutput -> oproj_output
  auto hidden_size = num_heads * head_dims;
  std::vector<int64_t> oproj_outputShape = {bs, seq_len, hidden_size};
  CreateAclTensorWithData(oproj_outputShape, &tmp_buffers[2], dtype, fmt, oproj_output);
  llm_kernels::ascend::MatMul(attnOutput, decoderLayerInfo.o_proj->GetAclTensor(), mm_type, oproj_output, stream,
                              llm_kernels::utils::GetTestWorkSpaceFunc);
  aclDestroyTensor(attnOutput);
}

void LlamaMLP(const aclTensor* post_norm_output, const int bs, const int64_t seq_len, const int64_t hidden_size,
              const int64_t ffn_size, DecoderLayerInfo& decoderLayerInfo, void** tmp_buffer_vocab1,
              void** tmp_buffer_vocab2, void** tmp_buffer_vocab3, aclTensor** down_output, aclrtStream& stream) {
  // need 3 tmp bufers
  auto dtype = aclDataType::ACL_FLOAT16;
  auto fmt = aclFormat::ACL_FORMAT_ND;
  aclOpExecutor* executor;
  int mm_type = 0;

  // input(post_norm_output) -> gate_output
  aclTensor* gate_output = nullptr;
  std::vector<int64_t> gate_output_shape = {bs, seq_len, ffn_size};
  CreateAclTensorWithData(gate_output_shape, tmp_buffer_vocab1, dtype, fmt, &gate_output);
  llm_kernels::ascend::MatMul(post_norm_output, decoderLayerInfo.gate_proj->GetAclTensor(), mm_type, &gate_output,
                              stream, llm_kernels::utils::GetTestWorkSpaceFunc);

  // gate_output -> silu_output
  aclTensor* silu_output = nullptr;
  std::vector<int64_t> silu_output_shape = {bs, seq_len, ffn_size};
  CreateAclTensorWithData(silu_output_shape, tmp_buffer_vocab2, dtype, fmt, &silu_output);
  llm_kernels::ascend::Silu(gate_output, &silu_output, stream, llm_kernels::utils::GetTestWorkSpaceFunc);
  aclDestroyTensor(gate_output);

  // post_norm_output -> up_output
  aclTensor* up_output = nullptr;
  std::vector<int64_t> up_output_shape = {bs, seq_len, ffn_size};
  CreateAclTensorWithData(up_output_shape, tmp_buffer_vocab1, dtype, fmt, &up_output);
  llm_kernels::ascend::MatMul(post_norm_output, decoderLayerInfo.up_proj->GetAclTensor(), mm_type, &up_output, stream,
                              llm_kernels::utils::GetTestWorkSpaceFunc);

  // up_output * silu_output -> mul_output
  aclTensor* mul_output = nullptr;
  std::vector<int64_t> mul_output_shape = {bs, seq_len, ffn_size};
  CreateAclTensorWithData(mul_output_shape, tmp_buffer_vocab3, dtype, fmt, &mul_output);
  llm_kernels::ascend::Mul(up_output, silu_output, &mul_output, stream, llm_kernels::utils::GetTestWorkSpaceFunc);
  aclDestroyTensor(up_output);
  aclDestroyTensor(silu_output);

  // mul_output -> down_output
  std::vector<int64_t> down_output_shape = {bs, seq_len, hidden_size};
  CreateAclTensorWithData(down_output_shape, tmp_buffer_vocab1, dtype, fmt, down_output);
  llm_kernels::ascend::MatMul(mul_output, decoderLayerInfo.down_proj->GetAclTensor(), mm_type, down_output, stream,
                              llm_kernels::utils::GetTestWorkSpaceFunc);
  aclDestroyTensor(mul_output);
}

void LlamaDecode(aclTensor* decoderLayerInput, void** decoderLayerOutputDev, int seq_len, int posIndex,
                 DecoderLayerInfo& decoderLayerInfo, const bool is_context_stage,
                 std::unique_ptr<llm_kernels::ascend::FlashAttentionACL>& flash_attn, aclrtStream& stream) {
  size_t maxDevSize;
  if (is_context_stage) {
    maxDevSize = bs * seq_len * ffn_intermediate_size * sizeof(uint16_t);
  } else {
    maxDevSize = bs * hidden_size * (seq_len + max_ans_len) * sizeof(uint16_t);
  }
  auto qkvoutSize = bs * seq_len * 3 * hidden_size * sizeof(uint16_t);

  std::vector<void*> tmp_buffers(5, nullptr);
  ACL_CHECK_RET(aclrtMalloc(&(tmp_buffers[0]), std::max(maxDevSize, qkvoutSize), ACL_MEM_MALLOC_NORMAL_ONLY));
  for (int i = 1; i < 5; ++i) {
    ACL_CHECK_RET(aclrtMalloc(&(tmp_buffers[i]), maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY));
  }

  // / step.1 input layernorm
  // input(bs, seq_len, hiddens) - >  rmsnormOutput(bs, seq_len, hiddens)
  aclTensor* rmsnormOutput = nullptr;
  std::vector<int64_t> rmsnormOutputShape = {bs, seq_len, hidden_size};
  CreateAclTensorWithData(rmsnormOutputShape, &tmp_buffers[1], dtype, fmt, &rmsnormOutput);
  llm_kernels::ascend::RMSLayerNorm(decoderLayerInput, decoderLayerInfo.attn_rms->GetAclTensor(), &rmsnormOutput,
                                    stream, llm_kernels::utils::GetTestWorkSpaceFunc);
  // PrintTensor(rmsnormOutput, stream, "rmsnormOutput");

  // / step.2 llama attn
  aclTensor* oproj_output = nullptr;
  LlamaAttn(rmsnormOutput, seq_len, posIndex, decoderLayerInfo, tmp_buffers, &oproj_output, is_context_stage,
            flash_attn, stream);
  // PrintTensor(oproj_output, stream, "oproj_output");

  // / step.3 add: decoderLayerInput + oproj_output -> add_output
  aclTensor* add_output = nullptr;
  uint16_t one_in_fp16 = 0b11110000000000;
  aclScalar* add_alpha = aclCreateScalar(&one_in_fp16, dtype);
  std::vector<int64_t> add_outputShape = {bs, seq_len, hidden_size};
  CreateAclTensorWithData(add_outputShape, &tmp_buffers[0], dtype, fmt, &add_output);
  llm_kernels::ascend::Add(decoderLayerInput, oproj_output, add_alpha, &add_output, stream,
                           llm_kernels::utils::GetTestWorkSpaceFunc);
  aclDestroyTensor(oproj_output);

  // 1-4 buffer can use
  // / step.4 post_attention_layernorm
  aclTensor* post_norm_output = nullptr;  // tmp[1]
  CreateAclTensorWithData(rmsnormOutputShape, &tmp_buffers[1], dtype, fmt, &post_norm_output);
  llm_kernels::ascend::RMSLayerNorm(add_output, decoderLayerInfo.attn_post_rms->GetAclTensor(), &post_norm_output,
                                    stream, llm_kernels::utils::GetTestWorkSpaceFunc);

  // / step.5 MLP
  // 2-4 buffer can use
  aclTensor* down_output = nullptr;  // => tmp[2]
  LlamaMLP(post_norm_output, bs, seq_len, hidden_size, ffn_intermediate_size, decoderLayerInfo, &tmp_buffers[2],
           &tmp_buffers[3], &tmp_buffers[4], &down_output, stream);
  aclDestroyTensor(post_norm_output);

  // / step.6 add: down_output + add_output-> last_add_output
  aclTensor* last_add_output = nullptr;
  std::vector<int64_t> last_add_output_shape = {bs, seq_len, hidden_size};
  CreateAclTensorWithData(last_add_output_shape, decoderLayerOutputDev, dtype, fmt, &last_add_output);
  llm_kernels::ascend::Add(down_output, add_output, add_alpha, &last_add_output, stream,
                           llm_kernels::utils::GetTestWorkSpaceFunc);
  aclDestroyScalar(add_alpha);
  aclDestroyTensor(add_output);
  aclDestroyTensor(down_output);
  // PrintTensor(last_add_output, stream, "last_add_output");

  // release dev mem for infer
  for (auto& tmp : tmp_buffers) {
    aclrtFree(tmp);
    tmp = nullptr;
  }
}

void ExcuteLlamaInc(aclTensor* llama2ndInput, void** llama2ndOutputDev, LlamaInfo& llamaInfo,
                    std::unique_ptr<llm_kernels::ascend::FlashAttentionACL>& flash_attn, aclrtStream& stream) {
  auto time_start = GetCurrentTimeInUs();
  aclOpExecutor* executor;
  int seq_len = 1;
  auto maxDevSize = seq_len * vocab_size * sizeof(float);
  void* tmp_dev_a = nullptr;
  void* tmp_dev_b = nullptr;
  ACL_CHECK_RET(aclrtMalloc(&tmp_dev_a, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CHECK_RET(aclrtMalloc(&tmp_dev_b, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY));

  std::vector<int64_t> gatherOutputShape = {bs, seq_len, hidden_size};
  aclTensor* gatherOutput = nullptr;
  CreateAclTensorWithData(gatherOutputShape, &tmp_dev_a, dtype, fmt, &gatherOutput);
  int64_t sinDim = 0;
  llm_kernels::ascend::Gather(llamaInfo.emb_weights->GetAclTensor(), sinDim, llama2ndInput, &gatherOutput, stream,
                              llm_kernels::utils::GetTestWorkSpaceFunc);
  for (int i = 0; i < num_layers; i++) {
    std::cout << " layer :" << i << std::endl;
    // PrintTensor(gatherOutput, stream, "inc_gather ");
    LlamaDecode(gatherOutput, &tmp_dev_a, seq_len, llamaInfo.posIndex, llamaInfo.decoderLayerInfo[i], false, flash_attn,
                stream);
  }
  // PrintTensor(gatherOutput, stream, "last_inc_gather");
  LmHead(&gatherOutput, bs, seq_len, hidden_size, llamaInfo.lm_head_rms->GetAclTensor(),
         llamaInfo.lm_head->GetAclTensor(), &tmp_dev_b, &tmp_dev_a, llama2ndOutputDev, stream);

  // release dev mem for infer
  aclrtFree(tmp_dev_a);
  tmp_dev_a = nullptr;
  aclrtFree(tmp_dev_b);
  tmp_dev_b = nullptr;
  auto time_end = GetCurrentTimeInUs();
  std::cout << "inc total time " << (time_end - time_start) / 1000.0 << " ms\n" << std::endl;
}

void ExcuteLlamaPrompt(LlamaInfo& llamaInfo, int64_t prompt_len, void** llama1stOutDev,
                       std::unique_ptr<llm_kernels::ascend::FlashAttentionACL>& flash_attn, aclrtStream& stream) {
  auto time_start = GetCurrentTimeInUs();
  aclOpExecutor* executor;
  // malloc dev mem for infer
  int64_t seq_len = prompt_len;

  auto buffer_size_hidden = bs * seq_len * hidden_size;
  auto buffer_size_vocab = bs * seq_len * vocab_size;
  auto buffer_size = std::max(buffer_size_hidden, buffer_size_vocab) * sizeof(float);
  void* tmp_dev_a = nullptr;
  void* tmp_dev_b = nullptr;
  ACL_CHECK_RET(aclrtMalloc(&tmp_dev_a, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CHECK_RET(aclrtMalloc(&tmp_dev_b, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  // head
  // inputIds + embedTokens -> gatherOutput
  std::vector<int64_t> inputIdsShape = {bs, seq_len};
  std::vector<int64_t> gatherOutputShape = {bs, seq_len, hidden_size};
  aclTensor* inputIds = nullptr;
  aclTensor* gatherOutput = nullptr;
  CreateAclTensorWithData(inputIdsShape, &llamaInfo.inputIdsDev, aclDataType::ACL_INT64, fmt, &inputIds);
  CreateAclTensorWithData(gatherOutputShape, &tmp_dev_a, dtype, fmt, &gatherOutput);
  int64_t sinDim = 0;
  llm_kernels::ascend::Gather(llamaInfo.emb_weights->GetAclTensor(), sinDim, inputIds, &gatherOutput, stream,
                              llm_kernels::utils::GetTestWorkSpaceFunc);
  aclDestroyTensor(inputIds);

  for (int i = 0; i < num_layers; i++) {
    // std::cout << " layer :" << i << std::endl;
    // PrintTensor(gatherOutput, stream, "prompt_gather ");
    int posIndex = prompt_len - 1;
    LlamaDecode(gatherOutput, &tmp_dev_a, prompt_len, posIndex, llamaInfo.decoderLayerInfo[i], true, flash_attn,
                stream);
  }
  // PrintTensor(gatherOutput, stream, "last_prompt_gather");
  LmHead(&gatherOutput, bs, seq_len, hidden_size, llamaInfo.lm_head_rms->GetAclTensor(),
         llamaInfo.lm_head->GetAclTensor(), &tmp_dev_b, &tmp_dev_a, llama1stOutDev, stream);
  // release dev mem for infer
  aclrtFree(tmp_dev_a);
  tmp_dev_a = nullptr;
  aclrtFree(tmp_dev_b);
  tmp_dev_b = nullptr;
  auto time_end = GetCurrentTimeInUs();
  std::cout << "prompt total time " << (time_end - time_start) / 1000.0 << " ms\n" << std::endl;
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  // init acl resource
  ACL_CHECK_RET(aclInit(nullptr));
  ACL_CHECK_RET(aclrtSetDevice(deviceId));
  ACL_CHECK_RET(aclrtCreateContext(context, deviceId));
  ACL_CHECK_RET(aclrtSetCurrentContext(*context));
  ACL_CHECK_RET(aclrtCreateStream(stream));
  aclrtRunMode runMode;
  aclrtGetRunMode(&runMode);
  ACL_CHECK_EQ(runMode, ACL_HOST);
  return 0;
}

void MergeQKVWeight(aclTensor* qWeight, aclTensor* kWeight, aclTensor* vWeight, DecoderLayerInfo& dlInfo,
                    aclrtStream& stream) {
  std::vector<int64_t> qkv_weight_shape = {hidden_size, 3 * hidden_size};
  auto size = GetShapeSize(qkv_weight_shape) * DT2LONG.at(dtype);
  dlInfo.qkv_proj = std::make_unique<TensorWeight>(qkv_weight_shape, dtype, fmt);
  dlInfo.qkv_proj->CreateAclTensor();
  std::vector<const aclTensor*> inputs{qWeight, kWeight, vWeight};
  int64_t catDim = -1;
  llm_kernels::ascend::Cat(inputs, catDim, &(dlInfo.qkv_proj->acl_tensor_), stream,
                           llm_kernels::utils::GetTestWorkSpaceFunc);
}

void PrepareWeight(LlamaInfo& llamaInfo, int num_layers, std::unordered_map<std::string, aclTensor*>& weight,
                   aclrtStream& stream) {
  std::cout << "[INFO] weight preparing..." << std::endl;
  aclDataType dtype = aclDataType::ACL_FLOAT16;
  aclFormat fmt = aclFormat::ACL_FORMAT_ND;

  std::vector<int64_t> embedTokensWeightShape = {vocab_size, hidden_size};
  std::string embedTokensWeightPath = "../llama_weight/model.embed_tokens.weight.bin";
  llamaInfo.emb_weights = std::make_unique<TensorWeight>(embedTokensWeightShape, dtype, fmt);
  ACL_CHECK_RET(
      ReadDataToDevice(embedTokensWeightPath, embedTokensWeightShape, &(llamaInfo.emb_weights->data_dev_), dtype, fmt));

  std::vector<int64_t> stateMemShape = {bs, num_layers, max_prompt_len + max_ans_len, head_dims};
  size_t stateMemSize = GetShapeSize(stateMemShape) * sizeof(uint16_t);
  llamaInfo.decoderLayerInfo.resize(num_layers);
  for (int i = 0; i < num_layers; i++) {
    DecoderLayerInfo& dlInfo = llamaInfo.decoderLayerInfo[i];
    ACL_CHECK_RET(aclrtMalloc(&(dlInfo.total_key_cache), stateMemSize, ACL_MEM_MALLOC_NORMAL_ONLY));
    ACL_CHECK_RET(aclrtMalloc(&(dlInfo.total_val_cache), stateMemSize, ACL_MEM_MALLOC_NORMAL_ONLY));

    std::string head = "../llama_weight/model.layers." + INT2STR[i];
    // std::string dst_layer_name = "model.layers." + INT2STR[i];

    std::vector<int64_t> attn_weight_shape = {hidden_size, hidden_size};
    std::string matmul1WeightPath = head + ".self_attn.q_proj.weight.bin";
    dlInfo.q_proj = std::make_unique<TensorWeight>(attn_weight_shape, dtype, fmt);
    ACL_CHECK_RET(ReadDataToDevice(matmul1WeightPath, attn_weight_shape, &(dlInfo.q_proj->data_dev_), dtype, fmt));

    std::string matmul2WeightPath = head + ".self_attn.k_proj.weight.bin";
    dlInfo.k_proj = std::make_unique<TensorWeight>(attn_weight_shape, dtype, fmt);
    ACL_CHECK_RET(ReadDataToDevice(matmul2WeightPath, attn_weight_shape, &(dlInfo.k_proj->data_dev_), dtype, fmt));

    std::string matmul3WeightPath = head + ".self_attn.v_proj.weight.bin";
    dlInfo.v_proj = std::make_unique<TensorWeight>(attn_weight_shape, dtype, fmt);
    ACL_CHECK_RET(ReadDataToDevice(matmul3WeightPath, attn_weight_shape, &(dlInfo.v_proj->data_dev_), dtype, fmt));
    MergeQKVWeight(dlInfo.q_proj->GetAclTensor(), dlInfo.k_proj->GetAclTensor(), dlInfo.v_proj->GetAclTensor(), dlInfo,
                   stream);

    std::string matmulOWeightPath = head + ".self_attn.o_proj.weight.bin";
    dlInfo.o_proj = std::make_unique<TensorWeight>(attn_weight_shape, dtype, fmt);
    ACL_CHECK_RET(ReadDataToDevice(matmulOWeightPath, attn_weight_shape, &(dlInfo.o_proj->data_dev_), dtype, fmt));

    std::vector<int64_t> matmul5WeightShape = {hidden_size, ffn_intermediate_size};
    std::string matmul5WeightPath = head + ".mlp.gate_proj.weight.bin";
    dlInfo.gate_proj = std::make_unique<TensorWeight>(matmul5WeightShape, dtype, fmt);
    ACL_CHECK_RET(ReadDataToDevice(matmul5WeightPath, matmul5WeightShape, &(dlInfo.gate_proj->data_dev_), dtype, fmt));

    std::vector<int64_t> matmul6WeightShape = {hidden_size, ffn_intermediate_size};
    std::string matmul6WeightPath = head + ".mlp.up_proj.weight.bin";
    dlInfo.up_proj = std::make_unique<TensorWeight>(matmul6WeightShape, dtype, fmt);
    ACL_CHECK_RET(ReadDataToDevice(matmul6WeightPath, matmul6WeightShape, &(dlInfo.up_proj->data_dev_), dtype, fmt));

    std::vector<int64_t> matmul7WeightShape = {ffn_intermediate_size, hidden_size};
    std::string matmul7WeightPath = head + ".mlp.down_proj.weight.bin";
    dlInfo.down_proj = std::make_unique<TensorWeight>(matmul7WeightShape, dtype, fmt);
    ACL_CHECK_RET(ReadDataToDevice(matmul7WeightPath, matmul7WeightShape, &(dlInfo.down_proj->data_dev_), dtype, fmt));

    std::vector<int64_t> rms_weight_shape = {1, 1, hidden_size};
    std::string rm1mulWeightPath = head + ".input_layernorm.weight.bin";
    dlInfo.attn_rms = std::make_unique<TensorWeight>(rms_weight_shape, dtype, fmt);
    ACL_CHECK_RET(ReadDataToDevice(rm1mulWeightPath, rms_weight_shape, &(dlInfo.attn_rms->data_dev_), dtype, fmt));

    std::string rm2mulWeightPath = head + ".post_attention_layernorm.weight.bin";
    dlInfo.attn_post_rms = std::make_unique<TensorWeight>(rms_weight_shape, dtype, fmt);
    ACL_CHECK_RET(ReadDataToDevice(rm2mulWeightPath, rms_weight_shape, &(dlInfo.attn_post_rms->data_dev_), dtype, fmt));
  }

  std::string rmmulWeightPath = "../llama_weight/model.norm.weight.bin";
  std::vector<int64_t> mulWeightShape = {1, 1, hidden_size};
  llamaInfo.lm_head_rms = std::make_unique<TensorWeight>(mulWeightShape, dtype, fmt);
  ACL_CHECK_RET(ReadDataToDevice(rmmulWeightPath, mulWeightShape, &(llamaInfo.lm_head_rms->data_dev_), dtype, fmt));

  std::string lmHeadWeight = "../llama_weight/lm_head.weight.bin";
  std::vector<int64_t> lmHeadWeightShape = {hidden_size, vocab_size};
  llamaInfo.lm_head = std::make_unique<TensorWeight>(lmHeadWeightShape, dtype, fmt);
  ACL_CHECK_RET(ReadDataToDevice(lmHeadWeight, lmHeadWeightShape, &(llamaInfo.lm_head->data_dev_), dtype, fmt));
}

int main() {
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  ACL_CHECK_RET(Init(deviceId, &context, &stream));

  int32_t majorVersion = 0;
  int32_t minorVersion = 0;
  int32_t patchVersion = 0;
  ACL_CHECK_RET(aclrtGetVersion(&majorVersion, &minorVersion, &patchVersion));
  std::cout << "[INFO] ACL version " << majorVersion << "." << minorVersion << "." << patchVersion << std::endl;

  // prepare input data & weight
  LlamaInfo llamaInfo;
  std::unordered_map<std::string, aclTensor*> weight;
  PrepareWeight(llamaInfo, num_layers, weight, stream);
  std::cout << "[INFO] data prepared, ready to infer llama." << std::endl;

  auto inferOutSize = sizeof(int64_t);
  void* inferOutHostData = nullptr;
  ACL_CHECK_RET(aclrtMallocHost(&inferOutHostData, inferOutSize));

  int i = 0;
  std::stringstream sstream;
  while (1) {
    // If the file name is exit.bin, exit the loop
    struct stat buffer;
    auto exit_value = stat("../input/exit.bin", &buffer);
    if (!exit_value) {
      break;
    }

    std::string inputIdsPath;
    sstream.str("");
    sstream.clear();
    sstream << "../input/input_ids_" << i << ".bin";
    sstream >> inputIdsPath;
    exit_value = stat(inputIdsPath.c_str(), &buffer);
    if (exit_value) {
      sleep(1);
      continue;
    }
    // get real input sequence length, update
    int64_t prompt_len = 32;
    size_t fileSize = 0;
    auto retRead = ReadFile("../input/prompt_len.bin", fileSize, &prompt_len, sizeof(int64_t));
    CHECK_RET(retRead == true, LOG_PRINT("ReadFile prompt_len failed. ERROR: %d\n", retRead); return !retRead);

    std::vector<int64_t> inputIdsShape = {1, prompt_len};
    ACL_CHECK_RET(ReadDataToDevice(inputIdsPath, inputIdsShape, &llamaInfo.inputIdsDev, aclDataType::ACL_INT64, fmt));
    llamaInfo.posIndex = prompt_len;

    std::unique_ptr<llm_kernels::ascend::FlashAttentionACL> flash_attn(new llm_kernels::ascend::FlashAttentionACL());
    flash_attn->Init(max_tokens_num, head_dims, num_heads, num_heads, rope_theta, rope_scaling_factor, dtype, stream,
                     llm_kernels::utils::GetTestWorkSpaceFunc);

    std::vector<int64_t> llama1stOutShape = {1, 1};
    void* llama1stOutDev = nullptr;
    aclTensor* llama1stOut = nullptr;
    CreateAclTensor(llama1stOutShape, &llama1stOutDev, aclDataType::ACL_INT64, fmt, &llama1stOut);
    ExcuteLlamaPrompt(llamaInfo, prompt_len, &llama1stOutDev, flash_attn, stream);
    std::cout << "[INFO] llama 1st run SUCCESS" << std::endl;

    std::vector<int64_t> answerList;
    ACL_CHECK_RET(aclrtMemcpy(inferOutHostData, inferOutSize, llama1stOutDev, inferOutSize, ACL_MEMCPY_DEVICE_TO_HOST));
    int64_t inferOut = *(reinterpret_cast<int64_t*>(inferOutHostData));
    answerList.push_back(inferOut);
    int endcnt = 3;
    for (int i = 0; (i < max_ans_len) && endcnt; i++) {
      ExcuteLlamaInc(llama1stOut, &llama1stOutDev, llamaInfo, flash_attn, stream);
      llamaInfo.posIndex++;
      std::cout << "[INFO] llama 2nd run " << i << " SUCCESS" << std::endl;
      ACL_CHECK_RET(
          aclrtMemcpy(inferOutHostData, inferOutSize, llama1stOutDev, inferOutSize, ACL_MEMCPY_DEVICE_TO_HOST));
      inferOut = *(reinterpret_cast<int64_t*>(inferOutHostData));
      std::cout << "posIndex: " << llamaInfo.posIndex << ", out: " << inferOut << std::endl;
      answerList.push_back(inferOut);
      if (inferOut == 13) {
        endcnt--;
      }
    }
    // release resource
    aclDestroyTensor(llama1stOut);
    aclrtFree(llama1stOutDev);
    llama1stOutDev = nullptr;

    std::string outputPath = "";
    sstream.str("");
    sstream.clear();
    sstream << "../output/fin_result_" << i << ".bin";
    sstream >> outputPath;
    WriteFile(outputPath, answerList.data(), inferOutSize * answerList.size());
    i++;
  }

  return 0;
}
