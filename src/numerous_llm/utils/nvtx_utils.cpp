/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "numerous_llm/utils/nvtx_utils.h"

#include <nvToolsExt.h>
#include "loguru.hpp"

namespace numerous_llm {

std::string GetNVTXScopeName() { return nvtx_scope_name; }

void AddNVTXScope(std::string name) {
  nvtx_scope_name = fmt::format("{}{}/", nvtx_scope_name, name);
  return;
}

void SetNVTXScope(std::string name) {
  nvtx_scope_name = fmt::format("{}/", name);
  return;
}

void ResetNVTXScope() {
  nvtx_scope_name = "";
  return;
}

void SetNVTXDomainDeviceID(int deviceId) {
  domain_device_id = deviceId;
  return;
}

int GetNVTXDomainDeviceID() { return domain_device_id; }

void ResetDeviceDomain() {
  domain_device_id = 0;
  return;
}

bool IsEnableNVTX() {
  if (!has_read_nvtx_env) {
    static char* ft_nvtx_env_char = std::getenv("ENABLE_NVTX");
    is_enable_ft_nvtx = (ft_nvtx_env_char != nullptr && std::string(ft_nvtx_env_char) == "ON") ? true : false;
    has_read_nvtx_env = true;
  }
  return is_enable_ft_nvtx;
}

void StartNvtxRange(std::string name) {
  nvtxStringHandle_t name_id = nvtxDomainRegisterStringA(NULL, (GetNVTXScopeName() + name).c_str());
  nvtxEventAttributes_t event_attr = {0};
  event_attr.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
  event_attr.message.registered = name_id;
  event_attr.payloadType = NVTX_PAYLOAD_TYPE_INT32;
  event_attr.payload.iValue = GetNVTXDomainDeviceID();
  nvtxRangePushEx(&event_attr);
}

void EndNVTXRange() { nvtxRangePop(); }

}  // namespace numerous_llm