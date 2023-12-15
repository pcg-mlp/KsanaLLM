/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/utils/request.h"

namespace numerous_llm {

IdGenerator Request::id_generator_;

Request::Request() { req_id = id_generator_.Gen(); }

}  // namespace numerous_llm
