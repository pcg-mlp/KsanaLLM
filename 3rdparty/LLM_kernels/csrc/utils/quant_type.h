#pragma once

namespace llm_kernels {
namespace utils {

enum KVCacheType {
  kAuto = 0,
  kFp8E4M3 = 1,
  kFp8E5M2 = 2,
};

}  // namespace utils
}  // namespace llm_kernels