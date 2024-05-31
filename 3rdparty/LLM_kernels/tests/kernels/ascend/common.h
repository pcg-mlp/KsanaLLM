#pragma once

#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "csrc/utils/ascend/common.h"

int ReadDataToDevice(const std::string dataPath, const std::vector<int64_t> &shape, void **deviceAddr,
                     aclDataType dataType, aclFormat dataFormat, bool on_device = false);

bool ReadFile(const std::string &filePath, size_t fileSize, void *buffer, size_t bufferSize);

bool WriteFile(const std::string &filePath, const void *buffer, size_t size);

std::time_t GetCurrentTimeInUs();
