/**
 * @file keyword_test.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2021-04-01
 *
 * MIT License
 *
 */

#include "benlib/pattern/gettersetter.hpp"

#include "gtest/gtest.h"

class GetterSetterClass
{
private:
public:
    GetterSetterClass() {
    }
    ~GetterSetterClass() {
    }
    GETTERSETTER(int, Value, _value)
    GETTERSETTER(std::string, Name, _name)
};

TEST(GetterSetter, Basic_1) {
    GetterSetterClass test;
    test.setValue(5);
    test.setName("Hello");
    EXPECT_EQ(test.getValue(), 5);
    EXPECT_EQ(test.getName(), "Hello");
}

auto main(int argc, char** argv) -> int {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
