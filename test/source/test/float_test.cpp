/*==============================================================================
#
# File name     :   Math.hpp
#
# Author        :   Bensuperpc
# Date          :   18/11/2023
#
# Maintainers   :
#
# Summary       :   Math program
#
#
#=============================================================================*/

#include "benlib/math/float.hpp"

#include "gtest/gtest.h"

TEST(areEqual, basic_float_1) {
  float data1 = 1.5f;
  float data2 = 1.5f;
  bool result = benlib::math::fp::areEqual<float>(data1, data2);

  EXPECT_EQ(result, true);
}

TEST(areEqual, basic_float_2) {
  float data1 = 1.51541f;
  float data2 = 1.5f;
  bool result = benlib::math::fp::areEqual<float>(data1, data2);

  EXPECT_EQ(result, false);
}

TEST(areEqual, basic_double_1) {
  double data1 = 1.5f;
  double data2 = 1.5f;
  bool result = benlib::math::fp::areEqual<double>(data1, data2);

  EXPECT_EQ(result, true);
}

TEST(areEqual, basic_double_2) {
  double data1 = 1.51541f;
  double data2 = 1.5f;
  bool result = benlib::math::fp::areEqual<double>(data1, data2);

  EXPECT_EQ(result, false);
}

auto main(int argc, char** argv) -> int {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
