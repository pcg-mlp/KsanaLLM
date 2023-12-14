/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>

namespace numerous_llm {

static std::string nvtx_scope_name;
std::string GetNVTXScopeName();
void AddNVTXScope(std::string name);
void SetNVTXScope(std::string name);
void ResetNVTXScope();

static int domain_device_id = 0;
void SetNVTXDomainDeviceID(int deviceId);
int GetNVTXDomainDeviceID();
void ResetDeviceDomain();

bool IsEnableNVTX();

static bool has_read_nvtx_env = false;
static bool is_enable_ft_nvtx = false;

void StartNvtxRange(std::string name);
void EndNVTXRange();

#define START_NVTX_RANGE(name) \
  {                            \
    if (IsEnableNVTX()) {      \
      StartNvtxRange(name);    \
    }                          \
  }

#define END_NVTX_RANGE    \
  {                       \
    if (IsEnableNVTX()) { \
      EndNVTXRange();     \
    }                     \
  }

}  // namespace numerous_llm