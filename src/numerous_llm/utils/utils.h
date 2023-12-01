// Copyright 2023 Tencent Inc.  All rights reserved.

#pragma once

namespace numerous_llm {

// Example:
//   class T {
//    public:
//     DELETE_COPY_AND_MOVE(T);
//
//     ...
//   };
#define DELETE_COPY(class_name)           \
  class_name(const class_name&) = delete; \
  class_name& operator=(const class_name&) = delete
#define DELETE_MOVE(class_name)      \
  class_name(class_name&&) = delete; \
  class_name& operator=(class_name&&) = delete
#define DELETE_COPY_AND_MOVE(class_name) \
  DELETE_COPY(class_name);               \
  DELETE_MOVE(class_name)

// like python "for i in range(begin, end)"
#define FOR_RANGE(type, i, begin, end) for (type i = begin; i < end; ++i)

} // namespace numerous_llm