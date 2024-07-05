/**
 * @file test_sphere.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2021-04-01
 *
 * MIT License
 *
 */

#include "benlib/math/sphere.hpp"

#include "gtest/gtest.h"

TEST(sphereVolume, basic_uint64_t_1) {
    auto data1 = benlib::math::sphere::sphereVolume<uint64_t>(10000);
    auto data2 = static_cast<uint64_t>(4188790204786);

    EXPECT_EQ(data1, data2);
}

TEST(sphereVolume, basic_uint64_t_2) {
    auto data1 = benlib::math::sphere::sphereVolume<uint64_t>(151);
    auto data2 = static_cast<uint64_t>(14421799);

    EXPECT_EQ(data1, data2);
}

TEST(sphereVolume, basic_uint64_t_3) {
    auto data1 = benlib::math::sphere::sphereVolume<uint64_t>(0);
    auto data2 = static_cast<uint64_t>(0);

    EXPECT_EQ(data1, data2);
}

TEST(sphereSurface, basic_uint64_t_1) {
    auto data1 = benlib::math::sphere::sphereSurface<uint64_t>(10000);
    auto data2 = static_cast<uint64_t>(125663);

    EXPECT_EQ(data1, data2);
}

TEST(sphereSurface, basic_uint64_t_2) {
    auto data1 = benlib::math::sphere::sphereSurface<uint64_t>(151);
    auto data2 = static_cast<uint64_t>(1897);

    EXPECT_EQ(data1, data2);
}

TEST(sphereSurface, basic_uint64_t_3) {
    auto data1 = benlib::math::sphere::sphereSurface<uint64_t>(0);
    auto data2 = static_cast<uint64_t>(0);

    EXPECT_EQ(data1, data2);
}

/*
#if BOOST_COMP_GNUC
// 1086832411937628777542115
BOOST_AUTO_TEST_CASE(test_sphere_volume_3)
{
    BOOST_REQUIRE_MESSAGE(
        static_cast<boost::multiprecision::uint1024_t>(benlib::math::sphere::sphereVolume<boost::multiprecision::float128>(EARTH_RADIUS)
/ 1000000)
            == 1086832411937628777,
        static_cast<boost::multiprecision::uint1024_t>(benlib::math::sphere::sphereVolume<boost::multiprecision::float128>(EARTH_RADIUS)
/ 1000000)
            << " instead: " << 1086832411937628777);
}
#endif
*/

auto main(int argc, char** argv) -> int {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
