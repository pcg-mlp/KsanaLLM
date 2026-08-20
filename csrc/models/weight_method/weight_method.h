/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <any>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "models/weight_method/awq_method.h"
#include "models/weight_method/common_method.h"
#include "models/weight_method/fp8_e4m3_method.h"
#include "models/weight_method/gptq_method.h"
#include "models/weight_method/rtn_method.h"
#include "models/weight_method/w4a8_awq_method.h"
#include "models/weight_method/w4a8_klx_method.h"

namespace ksana_llm {

class FunctionHelper {
 private:
  std::function<std::any(std::vector<std::any>&)> func_;

 public:
  // 注册任意签名的成员函数
  template <typename ClassType, typename ReturnType, typename... Args>
  void Add(ClassType* obj, ReturnType (ClassType::*func)(Args...)) {
    func_ = [obj, func](std::vector<std::any>& args) -> std::any {
      return callHelper(obj, func, args, std::index_sequence_for<Args...>{});
    };
  }

  // 调用函数
  template <typename ReturnType, typename... Args>
  ReturnType Run(Args&&... args) {
    if (!func_) {
      throw std::runtime_error("Function not registered");
    }
    std::vector<std::any> argVec = {std::any(std::ref(args))...};
    std::any result = func_(argVec);
    return std::any_cast<ReturnType>(result);
  }

 private:
  template <typename ClassType, typename ReturnType, typename... Args, std::size_t... Is>
  static std::any callHelper(ClassType* obj, ReturnType (ClassType::*func)(Args...), std::vector<std::any>& args,
                             std::index_sequence<Is...>) {
    return (obj->*func)(std::any_cast<std::reference_wrapper<std::remove_reference_t<Args>>>(args[Is]).get()...);
  }
};

// Forward declaration
struct QuantConfig;

class WeightMethod {
 public:
  using RegistryType = std::unordered_map<std::string, std::unordered_map<std::string, FunctionHelper>>;

  WeightMethod(std::shared_ptr<CommonModelWeightLoader> common_weight_loader, int tp);
  WeightMethod(std::shared_ptr<CommonModelWeightLoader> common_weight_loader, int tp, const QuantConfig& quant_config);

  std::shared_ptr<CommonMethod> GetCommonMethod() const { return common_method_; }
  std::shared_ptr<AwqMethod> GetAwqMethod() const { return awq_method_; }
  std::shared_ptr<W4A8AWQMethod> GetW4A8AWQMethod() const { return w4a8_awq_method_; }
  std::shared_ptr<W4A8KlxMethod> GetW4A8KlxMethod() const { return w4a8_klx_method_; }
  std::shared_ptr<GptqMethod> GetGptqMethod() const { return gptq_method_; }
  std::shared_ptr<RtnMethod> GetRtnMethod() const { return rtn_method_; }
  std::shared_ptr<Fp8E4m3Method> GetFp8E4m3Method() const { return fp8_e4m3_method_; }
  RegistryType& GetRegistry() { return registry_; }

 public:
  // 两层 map 结构: registry_[name][stage] 存储 FunctionHelper 对象
  RegistryType registry_;

 private:
  std::shared_ptr<CommonMethod> common_method_;
  std::shared_ptr<AwqMethod> awq_method_;
  std::shared_ptr<W4A8AWQMethod> w4a8_awq_method_;
  std::shared_ptr<W4A8KlxMethod> w4a8_klx_method_;
  std::shared_ptr<GptqMethod> gptq_method_;
  std::shared_ptr<RtnMethod> rtn_method_;
  std::shared_ptr<Fp8E4m3Method> fp8_e4m3_method_;
};

}  // namespace ksana_llm