
#include "common.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>

#include "csrc/kernels/ascend/pointwise/pointwise.h"

using namespace llm_kernels::utils;

bool ReadFile(const std::string& filePath, size_t fileSize, void* buffer, size_t bufferSize) {
  struct stat sBuf;
  int fileStatus = stat(filePath.data(), &sBuf);
  if (fileStatus == -1) {
    ERROR_LOG("failed to get file %s", filePath.c_str());
    return false;
  }
  if (S_ISREG(sBuf.st_mode) == 0) {
    ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
    return false;
  }

  std::ifstream file;
  file.open(filePath, std::ios::binary);
  if (!file.is_open()) {
    ERROR_LOG("Open file failed. path = %s", filePath.c_str());
    return false;
  }

  std::filebuf* buf = file.rdbuf();
  size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
  if (size == 0) {
    ERROR_LOG("file size is 0");
    file.close();
    return false;
  }
  if (size > bufferSize) {
    ERROR_LOG("file size is larger than buffer size");
    file.close();
    return false;
  }
  buf->pubseekpos(0, std::ios::in);
  buf->sgetn(static_cast<char*>(buffer), size);
  fileSize = size;
  file.close();
  return true;
}

bool WriteFile(const std::string& filePath, const void* buffer, size_t size) {
  if (buffer == nullptr) {
    ERROR_LOG("Write file failed. buffer is nullptr");
    return false;
  }

  int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
  if (fd < 0) {
    ERROR_LOG("Open file failed. path = %s", filePath.c_str());
    return false;
  }

  auto writeSize = write(fd, buffer, size);
  (void)close(fd);
  if (writeSize != size) {
    ERROR_LOG("Write file Failed.");
    return false;
  }

  return true;
}

int ReadDataToDevice(const std::string dataPath, const std::vector<int64_t>& src_shape, void** deviceAddr,
                     aclDataType dataType, aclFormat dataFormat, bool on_device) {
  auto size = GetShapeSize(src_shape) * DT2LONG.at(dataType);
  void* hostData = nullptr;
  if (on_device) {
    ACL_CHECK_RET(aclrtMalloc(&hostData, size, ACL_MEM_MALLOC_NORMAL_ONLY));
  } else {
    ACL_CHECK_RET(aclrtMallocHost(&hostData, size));
  }
  size_t fileSize = 0;
  auto retRead = ReadFile(dataPath, fileSize, reinterpret_cast<void*>(hostData), size);
  CHECK_RET(retRead == true, ERROR_LOG("aclrtMalloc failed. ERROR: %d\n", retRead); return !retRead);
  ACL_CHECK_RET(aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMemcpy(*deviceAddr, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE));
  if (on_device) {
    ACL_CHECK_RET(aclrtFree(hostData));
    hostData = nullptr;
  } else {
    ACL_CHECK_RET(aclrtFreeHost(hostData));
    hostData = nullptr;
  }
  return 0;
}

std::time_t GetCurrentTimeInUs() {
  std::chrono::high_resolution_clock::duration d = std::chrono::high_resolution_clock::now().time_since_epoch();
  std::chrono::microseconds micro_sec = std::chrono::duration_cast<std::chrono::microseconds>(d);
  return micro_sec.count();
}
