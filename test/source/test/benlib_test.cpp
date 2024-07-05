#include "benlib/benlib.hpp"

#include "gtest/gtest.h"

TEST(BenLib, basic_1) {
    EXPECT_EQ(get_benlib(), "benlib");
}

auto main(int argc, char** argv) -> int {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
