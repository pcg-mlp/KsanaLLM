/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

class DynamicBatching {
 public:
  DynamicBatching();
  ~DynamicBatching();

 private:
  // The maxinum batch size that can be batched.
  unsigned int max_batch_size_;

  // The maxinum waiting ms.
  unsigned int max_waiting_ms_;
};
