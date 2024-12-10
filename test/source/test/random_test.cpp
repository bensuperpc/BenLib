/**
 * @file random_test.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief Source: https://stackoverflow.com/q/27228813
 * @version 1.0.0
 * @date 2022-03-21
 *
 * MIT License
 *
 */

#include <array>
#include <cstdlib>

#include "benlib/math/random.hpp"

#include "gtest/gtest.h"

#define EXPECT_IN_RANGE(VAL, MIN, MAX) \
    EXPECT_GE((VAL), (MIN));           \
    EXPECT_LE((VAL), (MAX))

#define ASSERT_IN_RANGE(VAL, MIN, MAX) \
    ASSERT_GE((VAL), (MIN));           \
    ASSERT_LE((VAL), (MAX))

/**
 * @brief Construct a new test case
 *
 */
TEST(random, basic_int_1) {
    auto min = 0;
    auto max = 100;

    auto result = benlib::math::rand::random<int>(min, max);
    EXPECT_IN_RANGE(result, min, max);
}

/**
 * @brief Construct a new test case
 *
 */
TEST(random, basic_int_2) {
    auto min = 0;
    auto max = 100;

    auto result = benlib::math::rand::random<int, false>(min, max);
    EXPECT_IN_RANGE(result, min, max);
}

/**
 * @brief Construct a new test case
 *
 */
TEST(random, basic_int_3) {
    auto min = 0;
    auto max = 100;
    auto result = -1;

    result = benlib::math::rand::random<int, false>(min, max);
    EXPECT_IN_RANGE(result, min, max);
}

/**
 * @brief Construct a new test case
 *
 */
TEST(random, basic_float_1) {
    auto min = 0.0f;
    auto max = 1.0f;

    auto result = benlib::math::rand::random<float>(min, max);
    EXPECT_IN_RANGE(result, min, max);
}

/**
 * @brief Construct a new test case
 *
 */
TEST(random, basic_float_2) {
    auto min = 0.0f;
    auto max = 1.0f;

    auto result = benlib::math::rand::random<float, false>(min, max);
    EXPECT_IN_RANGE(result, min, max);
}

/**
 * @brief Construct a new test case
 *
 */
TEST(random, basic_float_3) {
    float min = 0.0f;
    float max = 1.0f;

    float result = -1.0;

    result = benlib::math::rand::random<float, false>(min, max);
    EXPECT_IN_RANGE(result, min, max);
}

/**
 * @brief Construct a new test case
 *
 */
TEST(random, basic_vec_int_1) {
    std::vector<int> vec = {-1, -1, -1, -1, -1};
    const auto min = 0;
    const auto max = 100;

    benlib::math::rand::random<int, true>(vec, 0, 100);
    for (auto& vec_ : vec)  // access by reference to avoid copying
    {
        EXPECT_IN_RANGE(vec_, min, max);
    }
}

/**
 * @brief Construct a new test case
 *
 */
TEST(random, basic_vec_int_2) {
    std::vector<int> vec = {-1, -1, -1, -1, -1};
    const auto min = 0;
    const auto max = 160;

    benlib::math::rand::random<int, true>(vec, min, max);
    for (auto& vec_ : vec) {
        EXPECT_IN_RANGE(vec_, min, max);
    }
}

/**
 * @brief Construct a new test case
 *
 */
TEST(random, basic_vec_float_1) {
    std::vector<float> vec = {-1.0, -1.0, -1.0, -1.0, -1.0};
    const auto min = 0.0;
    const auto max = 1.0;

    benlib::math::rand::random<float, true>(vec, min, max);
    for (auto& vec_ : vec) {
        EXPECT_IN_RANGE(vec_, min, max);
    }
}

/**
 * @brief Construct a new test case
 *
 */
TEST(random, basic_vec_float_2) {
    std::vector<float> vec = {-2.0, -2.0, -2.0, -2.0, -2.0};
    const auto min = -1.0;
    const auto max = 1.0;

    benlib::math::rand::random<float, true>(vec, min, max);
    for (auto& vec_ : vec) {
        EXPECT_IN_RANGE(vec_, min, max);
    }
}

/**
 * @brief Construct a new test case
 *
 */
TEST(random, basic_vec_double_1) {
    std::vector<double> vec = {-1.0, -1.0, -1.0, -1.0, -1.0};
    const auto min = 0.0;
    const auto max = 1.0;

    benlib::math::rand::random<double, true>(vec, min, max);
    for (auto& vec_ : vec) {
        EXPECT_IN_RANGE(vec_, min, max);
    }
}

/**
 * @brief Construct a new test case
 *
 */
TEST(random, basic_arr_double_1) {
    std::array<double, 5> vec = {-1.0, -1.0, -1.0, -1.0, -1.0};
    const auto min = 0.0;
    const auto max = 1.0;

    benlib::math::rand::random<double, true>(vec.data(), 5, min, max);
    for (auto& vec_ : vec) {
        EXPECT_IN_RANGE(vec_, min, max);
    }
}

/**
 * @brief Construct a new test case
 *
 */
TEST(random, basic_arr_int_1) {
    std::array<int, 5> vec = {-1, -1, -1, -1, -1};
    const auto min = 0;
    const auto max = 1;

    benlib::math::rand::random<int, true>(vec.data(), 5, min, max);
    for (auto& vec_ : vec) {
        EXPECT_IN_RANGE(vec_, min, max);
    }
}

/*
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
*/

auto main(int argc, char** argv) -> int {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
