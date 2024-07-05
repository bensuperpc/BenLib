/**
 * @file test_cylinder.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2021-04-01
 *
 * MIT License
 *
 */

#include "benlib/math/cylinder.hpp"

#include "gtest/gtest.h"

/**
 * @brief Construct a new boost auto test case object
 *
 */
TEST(cylinder_volume, basic_uint64_t_1) {
  auto data1 = benlib::math::cylinder::cylinderVolume<uint64_t>(1000, 5);
  auto data2 = static_cast<uint64_t>(15707963);

  EXPECT_EQ(data1, data2);
}

TEST(cylinder_volume, basic_uint64_t_2) {
  auto data1 = benlib::math::cylinder::cylinderVolume<uint64_t>(151, 10);
  auto data2 = static_cast<uint64_t>(716314);

  EXPECT_EQ(data1, data2);
}

TEST(cylinder_volume, basic_uint64_t_3) {
  auto data1 = benlib::math::cylinder::cylinderVolume<uint64_t>(0, 0);
  auto data2 = static_cast<uint64_t>(0);

  EXPECT_EQ(data1, data2);
}

TEST(cylinder_volume, basic_int_1) {
  auto data1 = benlib::math::cylinder::cylinderVolume<int>(32, 3);
  auto data2 = static_cast<int>(9650);

  EXPECT_EQ(data1, data2);
}

TEST(cylinder_volume, basic_int_2) {
  auto data1 = benlib::math::cylinder::cylinderVolume<int>(51, 2);
  auto data2 = static_cast<int>(16342);

  EXPECT_EQ(data1, data2);
}

TEST(cylinder_volume, basic_int_3) {
  auto data1 = benlib::math::cylinder::cylinderVolume<int>(0, 0);
  auto data2 = static_cast<int>(0);

  EXPECT_EQ(data1, data2);
}

TEST(cylinder_surface, basic_uint64_t_1) {
  auto data1 = benlib::math::cylinder::cylinderSurface<uint64_t>(10000, 9);
  auto data2 = static_cast<uint64_t>(628884017);

  EXPECT_EQ(data1, data2);
}

TEST(cylinder_surface, basic_uint64_t_2) {
  auto data1 = benlib::math::cylinder::cylinderSurface<uint64_t>(151, 6);
  auto data2 = static_cast<uint64_t>(148955);

  EXPECT_EQ(data1, data2);
}

TEST(cylinder_surface, basic_uint64_t_3) {
  auto data1 = benlib::math::cylinder::cylinderSurface<uint64_t>(0, 0);
  auto data2 = static_cast<uint64_t>(0);

  EXPECT_EQ(data1, data2);
}

TEST(cylinder_surface, basic_int_1) {
  auto data1 = benlib::math::cylinder::cylinderSurface<int>(6598, 4);
  auto data2 = static_cast<int>(273695526);

  EXPECT_EQ(data1, data2);
}

TEST(cylinder_surface, basic_int_2) {
  auto data1 = benlib::math::cylinder::cylinderSurface<int>(151, 6);
  auto data2 = static_cast<int>(148955);

  EXPECT_EQ(data1, data2);
}

TEST(cylinder_surface, basic_int_3) {
  auto data1 = benlib::math::cylinder::cylinderSurface<int>(0, 0);
  auto data2 = static_cast<int>(0);

  EXPECT_EQ(data1, data2);
}

auto main(int argc, char** argv) -> int {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
