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

#include "benlib/keyword/keywordhandler.hpp"

#include "gtest/gtest.h"

TEST(keywordhandler, basic_1) {
    KeywordHandler keywordHandler;

    size_t count = 0;
    std::function<void()> func = [&]() { count++; };

    keywordHandler.addKeyword("hello", func);

    keywordHandler.callKeyword("hello");
    keywordHandler.callKeyword("hello");
    EXPECT_EQ(count, 2);

    keywordHandler.removeKeyword("hello");
    keywordHandler.callKeyword("hello");
    EXPECT_EQ(count, 2);
}

TEST(keywordhandler, basic_2) {
    KeywordHandler keywordHandler;

    size_t count = 0;
    std::function<void()> func1 = [&]() { count++; };
    std::function<void()> func2 = [&]() { count++; };

    keywordHandler.addKeyword("func1", func1);
    keywordHandler.addKeyword("func2", func2);

    keywordHandler.callKeyword("func1");
    keywordHandler.callKeyword("func2");
    EXPECT_EQ(count, 2);

    keywordHandler.removeKeyword("func1");

    keywordHandler.callKeyword("func1");
    keywordHandler.callKeyword("func2");
    EXPECT_EQ(count, 3);

    keywordHandler.removeKeyword("func2");

    keywordHandler.callKeyword("func1");
    keywordHandler.callKeyword("func2");

    EXPECT_EQ(count, 3);
}

TEST(keywordhandler, multiple_same_key_1) {
    KeywordHandler keywordHandler;

    size_t countFunc1 = 0;
    size_t countFunc2 = 0;
    std::function<void(void)> func1 = [&]() { countFunc1++; };
    std::function<void(void)> func2 = [&]() { countFunc2++; };

    keywordHandler.addKeyword("func", func1);
    keywordHandler.addKeyword("func", func2);

    keywordHandler.callKeyword("func");
    keywordHandler.callKeyword("func");

    EXPECT_EQ(countFunc1, 2);
    EXPECT_EQ(countFunc2, 2);
}

TEST(keywordhandler, multiple_same_key_2) {
    KeywordHandler keywordHandler;

    size_t countFunc1 = 0;
    size_t countFunc2 = 0;
    size_t countFunc3 = 0;
    size_t countFunc4 = 0;
    std::function<void(void)> func1 = [&]() { countFunc1++; };
    std::function<void(void)> func2 = [&]() { countFunc2++; };
    std::function<void(void)> func3 = [&]() { countFunc3++; };
    std::function<void(void)> func4 = [&]() { countFunc4++; };

    keywordHandler.addKeyword("func", func1);
    keywordHandler.addKeyword("func", func2);
    keywordHandler.addKeyword("func", func3);
    keywordHandler.addKeyword("func", func4);

    keywordHandler.callKeyword("func");
    keywordHandler.callKeyword("func");

    EXPECT_EQ(countFunc1, 2);
    EXPECT_EQ(countFunc2, 2);
    EXPECT_EQ(countFunc3, 2);
    EXPECT_EQ(countFunc4, 2);
}

TEST(keywordhandler, multi_param_1) {
    KeywordHandler keywordHandler;

    size_t countFunc1 = 0;
    size_t countFunc2 = 0;
    std::function<void(int, int)> func1 = [&](int a, int b) { countFunc1 = a + b; };
    std::function<void(int, int)> func2 = [&](int a, int b) { countFunc2 = a * b; };

    keywordHandler.addKeyword("func1", func1);
    keywordHandler.addKeyword("func2", func2);

    keywordHandler.callKeyword("func1", 2, 3);
    keywordHandler.callKeyword("func2", 2, 3);

    EXPECT_EQ(countFunc1, 5);
    EXPECT_EQ(countFunc2, 6);
}

TEST(keywordhandler, multi_param_same_key_1) {
    KeywordHandler keywordHandler;

    size_t countFunc1a = 0;
    size_t countFunc2b = 0;
    std::function<void(int, int)> func1 = [&](int a, int b) { countFunc1a = a + b; };
    std::function<void(int, int, int)> func2 = [&](int a, int b, int c) { countFunc2b = a * b * c; };

    keywordHandler.addKeyword("func", func1);
    keywordHandler.addKeyword("func", func2);

    keywordHandler.callKeyword("func", 2, 3);
    keywordHandler.callKeyword("func", 2, 3, 4);

    EXPECT_EQ(countFunc1a, 5);
    EXPECT_EQ(countFunc2b, 24);
}

auto main(int argc, char** argv) -> int {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
