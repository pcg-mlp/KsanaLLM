/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/weight_method/weight_method.h"

namespace ksana_llm {

WeightMethod::WeightMethod(std::shared_ptr<CommonModelWeightLoader> common_weight_loader, int tp)
    : common_method_(std::make_shared<CommonMethod>(common_weight_loader, tp)),
      awq_method_(nullptr),
      w4a8_awq_method_(std::make_shared<W4A8AWQMethod>(common_weight_loader, tp)),
      w4a8_klx_method_(std::make_shared<W4A8KlxMethod>(common_weight_loader, tp)),
      gptq_method_(nullptr),
      rtn_method_(std::make_shared<RtnMethod>(common_weight_loader, tp)),
      fp8_e4m3_method_(nullptr) {}

WeightMethod::WeightMethod(std::shared_ptr<CommonModelWeightLoader> common_weight_loader, int tp,
                           const QuantConfig& quant_config)
    : common_method_(std::make_shared<CommonMethod>(common_weight_loader, tp)),
      awq_method_(std::make_shared<AwqMethod>(common_weight_loader, tp, quant_config)),
      w4a8_awq_method_(std::make_shared<W4A8AWQMethod>(common_weight_loader, tp)),
      w4a8_klx_method_(std::make_shared<W4A8KlxMethod>(common_weight_loader, tp)),
      gptq_method_(std::make_shared<GptqMethod>(common_weight_loader, tp, quant_config)),
      rtn_method_(std::make_shared<RtnMethod>(common_weight_loader, tp)),
      fp8_e4m3_method_(std::make_shared<Fp8E4m3Method>(common_weight_loader, tp, quant_config)) {}

}  // namespace ksana_llm