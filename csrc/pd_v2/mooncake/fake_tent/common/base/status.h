/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// FAKE mooncake::Status — minimal stand-in for
// common/base/status.h, used only by the pd_v2 connector coverage test.
// Mirrors the real Status surface the pd_v2 wrappers touch: ok() + code().
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace mooncake {

class Status final {
 public:
  enum class Code : uint16_t {
    kOk = 0,
    kInvalidArgument = 1,
    kRuntime = 2,
  };

  Status() = default;
  explicit Status(Code code) : code_(code) {}
  Status(Code code, std::string_view message) : code_(code), message_(message) {}

  Code code() const { return code_; }
  bool ok() const { return Code::kOk == code_; }
  const std::string& message() const { return message_; }

  static Status OK() { return Status(); }

 private:
  Code code_ = Code::kOk;
  std::string message_;
};

}  // namespace mooncake
