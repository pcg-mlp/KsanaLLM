/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "tests/test.h"

#include <pybind11/embed.h>

namespace py = pybind11;

// Keep alive this python interpreter during the whole test process.
class PythonEnvironment : public testing::Environment {
 public:
  virtual ~PythonEnvironment() = default;

  virtual void SetUp() { py::initialize_interpreter(); }

  virtual void TearDown() { py::finalize_interpreter(); }
};

int main(int argc, char** argv) {
  PythonEnvironment* env = new PythonEnvironment();
  testing::AddGlobalTestEnvironment(env);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
