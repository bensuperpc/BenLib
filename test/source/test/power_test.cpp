/**
 * @file test_power.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2021-04-01
 *
 * MIT License
 *
 */

#include "benlib/math/power.hpp"

#include "gtest/gtest.h"

TEST(power, basic_uint64_t_1) {
  auto data1 = benlib::math::power<uint64_t>(2, 16);
  auto data2 = static_cast<uint64_t>(65536);

  EXPECT_EQ(data1, data2);
}

TEST(power, basic_uint64_t_2) {
  auto data1 = benlib::math::power<uint64_t>(2, 32);
  auto data2 = static_cast<uint64_t>(4294967296);

  EXPECT_EQ(data1, data2);
}

TEST(power, basic_uint64_t_3) {
  auto data1 = benlib::math::power<uint64_t>(55, 0);
  auto data2 = static_cast<uint64_t>(1);

  EXPECT_EQ(data1, data2);
}

TEST(power, basic_int_4) {
  auto data1 = benlib::math::power<int>(-5, 3);
  auto data2 = static_cast<int>(-125);

  EXPECT_EQ(data1, data2);
}

auto main(int argc, char** argv) -> int {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
