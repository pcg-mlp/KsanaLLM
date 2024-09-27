/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/status.h"

namespace ksana_llm {

Status::Status(RetCode code, const std::string &message) {
  if (code == RetCode::RET_SUCCESS) {
    return;
  }

  state_ = std::make_shared<State>();
  state_->code = code;
  state_->message = message;
}

Status::Status(const Status &status) : Status(status.GetCode(), status.GetMessage()) {}

const std::string &Status::GetEmptyString() {
  static std::string empty;
  return empty;
}

std::string Status::ToString() const {
  if (OK()) {
    return "OK";
  }

  char tmp[32];
  std::snprintf(tmp, sizeof(tmp), "ret: %d, err: ", state_->code);
  return std::string(tmp) + state_->message;
}

}  // namespace ksana_llm
