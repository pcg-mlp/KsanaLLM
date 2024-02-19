/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/request.h"

namespace ksana_llm {

IdGenerator Request::id_generator_;

Request::Request() { req_id = id_generator_.Gen(); }

}  // namespace ksana_llm
