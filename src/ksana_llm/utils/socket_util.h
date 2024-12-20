/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <unistd.h>

#include <string>

#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// Get available interface and ip
Status GetAvailableInterfaceAndIP(std::string& interface, std::string& ip);

// Get available port.
Status GetAvailablePort(uint16_t& port);

}  // namespace ksana_llm
