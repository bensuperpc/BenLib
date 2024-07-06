/**
 * @file angle_test.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2024-07-06
 *
 * MIT License
 *
 */

#include "benlib/math/geometry/angle.hpp"

#include "gtest/gtest.h"

TEST(AngleRadToDeg, basic_float_1) {
    float dataRad = 1.0;
    float dataDeg = 57.29577951308232;

    EXPECT_NEAR(benlib::math::geometry::angle::radToDeg<float>(dataRad), dataDeg, 0.001);
}

TEST(AngleRadToDeg, basic_double_1) {
    double dataRad = 2.0;
    double dataDeg = 114.59155902616465;

    EXPECT_NEAR(benlib::math::geometry::angle::radToDeg<double>(dataRad), dataDeg, 0.0001);
}

TEST(AngleDegToRad, basic_float_1) {
    float dataDeg = 57.29577951308232;
    float dataRad = 1.0;

    EXPECT_NEAR(benlib::math::geometry::angle::degToRad<float>(dataDeg), dataRad, 0.001);
}

TEST(AngleDegToRad, basic_double_1) {
    double dataDeg = 114.59155902616465;
    double dataRad = 2.0;

    EXPECT_NEAR(benlib::math::geometry::angle::degToRad<double>(dataDeg), dataRad, 0.0001);
}

auto main(int argc, char** argv) -> int {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
