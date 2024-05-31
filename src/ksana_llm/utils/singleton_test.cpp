/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/singleton.h"

#include "tests/test.h"

namespace ksana_llm {

// Faked type used for template
struct FakedSingletonType {
  int val = 8;
};

// Facked normal class used for test
class TestSingletonNormalClass {
 public:
  int Fun() { return 1; }
};

// Facked template class used for test
template <typename T>
class TestSingletonTemplateClass {
 public:
  T Fun(T val) { return val; }
};

TEST(Singleton, TestSingleton) {
  int val = Singleton<TestSingletonNormalClass>::GetInstance()->Fun();
  EXPECT_EQ(val, 1);

  int int_val = Singleton<TestSingletonTemplateClass<int>>::GetInstance()->Fun(2);
  EXPECT_EQ(int_val, 2);

  std::vector<float> vector_val =
      Singleton<TestSingletonTemplateClass<std::vector<float>>>::GetInstance()->Fun(std::vector<float>(4, 3.14));
  EXPECT_EQ(vector_val.size(), size_t(4));

  FakedSingletonType struct_val =
      Singleton<TestSingletonTemplateClass<FakedSingletonType>>::GetInstance()->Fun(FakedSingletonType());
  EXPECT_EQ(struct_val.val, 8);
}

}  // namespace ksana_llm
