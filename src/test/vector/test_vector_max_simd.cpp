#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE vector_max_simd

#include <algorithm>
#include <boost/predef.h>
#include <random>

#if BOOST_COMP_GNUC
extern "C"
{
#    include "quadmath.h"
}
#endif

#include <boost/multiprecision/cpp_int.hpp>
#if BOOST_COMP_GNUC
#    include <boost/multiprecision/float128.hpp>
#endif
#include <boost/test/unit_test.hpp>
#include "vector/vector_avx.hpp"

BOOST_AUTO_TEST_CASE(test_vector_max_simd_1)
{
    const size_t i = 8192;
    int *n = new int[i];
    for (size_t x = 0; x < i; ++x) {
        n[x] = rand() % 2000000000;
    }
#if (__AVX2__ || __AVX__)
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_normal(n, i) == my::vector_avx::find_max_avx(n, i), "find_max_normal != find_max_avx");
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_sse(n, i) == my::vector_avx::find_max_avx(n, i), "find_max_sse != find_max_avx");
#endif
#if (__SSE3__ || __SSE2__)
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_normal(n, i) == my::vector_avx::find_max_sse(n, i), "find_max_normal != find_max_sse");
#endif
}

BOOST_AUTO_TEST_CASE(test_vector_max_simd_2)
{
    const size_t i = 8192;
    int *n = new int[i];
    for (size_t x = 0; x < i; ++x) {
        n[x] = rand() % 8000;
    }
#if (__AVX2__ || __AVX__)
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_normal(n, i) == my::vector_avx::find_max_avx(n, i), "find_max_normal != find_max_avx");
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_sse(n, i) == my::vector_avx::find_max_avx(n, i), "find_max_sse != find_max_avx");
#endif
#if (__SSE3__ || __SSE2__)
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_normal(n, i) == my::vector_avx::find_max_sse(n, i), "find_max_normal != find_max_sse");
#endif
}

BOOST_AUTO_TEST_CASE(test_vector_max_simd_3)
{
    const size_t i = 1024;
    int *n = new int[i];
    for (size_t x = 0; x < i; ++x) {
        n[x] = rand() % 1000;
    }
#if (__AVX2__ || __AVX__)
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_normal(n, i) == my::vector_avx::find_max_avx(n, i), "find_max_normal != find_max_avx");
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_sse(n, i) == my::vector_avx::find_max_avx(n, i), "find_max_sse != find_max_avx");
#endif
#if (__SSE3__ || __SSE2__)
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_normal(n, i) == my::vector_avx::find_max_sse(n, i), "find_max_normal != find_max_sse");
#endif
}

BOOST_AUTO_TEST_CASE(test_vector_max_simd_4)
{
    const size_t i = 8192;
    int *n = new int[i];
    for (size_t x = 0; x < i; ++x) {
        n[x] = 1000;
    }
#if (__AVX2__ || __AVX__)
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_normal(n, i) == my::vector_avx::find_max_avx(n, i), "find_max_normal != find_max_avx");
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_sse(n, i) == my::vector_avx::find_max_avx(n, i), "find_max_sse != find_max_avx");
#endif
#if (__SSE3__ || __SSE2__)
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_normal(n, i) == my::vector_avx::find_max_sse(n, i), "find_max_normal != find_max_sse");
#endif
}

BOOST_AUTO_TEST_CASE(test_vector_max_simd_5)
{
    const size_t i = 16384;
    int *n = new int[i];
    for (size_t x = 0; x < i; ++x) {
        n[x] = 0;
    }
#if (__AVX2__ || __AVX__)
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_normal(n, i) == my::vector_avx::find_max_avx(n, i), "find_max_normal != find_max_avx");
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_sse(n, i) == my::vector_avx::find_max_avx(n, i), "find_max_sse != find_max_avx");
#endif
#if (__SSE3__ || __SSE2__)
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_normal(n, i) == my::vector_avx::find_max_sse(n, i), "find_max_normal != find_max_sse");
#endif
}

BOOST_AUTO_TEST_CASE(test_vector_max_simd_6)
{
    const size_t i = 16384;
    int *n = new int[i];
    for (size_t x = 0; x < i; ++x) {
        n[x] = 2147483647;
    }
#if (__AVX2__ || __AVX__)
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_normal(n, i) == my::vector_avx::find_max_avx(n, i), "find_max_normal != find_max_avx");
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_sse(n, i) == my::vector_avx::find_max_avx(n, i), "find_max_sse != find_max_avx");
#endif
#if (__SSE3__ || __SSE2__)
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_normal(n, i) == my::vector_avx::find_max_sse(n, i), "find_max_normal != find_max_sse");
#endif
}

BOOST_AUTO_TEST_CASE(test_vector_max_simd_7)
{
    const size_t i = 16384;
    int *n = new int[i];
    for (size_t x = 0; x < i; ++x) {
        n[x] = -2147483647;
    }
#if (__AVX2__ || __AVX__)
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_normal(n, i) == my::vector_avx::find_max_avx(n, i), "find_max_normal != find_max_avx");
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_sse(n, i) == my::vector_avx::find_max_avx(n, i), "find_max_sse != find_max_avx");
#endif
#if (__SSE3__ || __SSE2__)
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_normal(n, i) == my::vector_avx::find_max_sse(n, i), "find_max_normal != find_max_sse");
#endif
}

BOOST_AUTO_TEST_CASE(test_vector_max_simd_8)
{
    const size_t i = 16384;
    int *n = new int[i];
    for (size_t x = 0; x < i; ++x) {
        n[x] = rand() % 100000;
    }
#if (__AVX2__ || __AVX__)
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_normal(n, i) == my::vector_avx::find_max_avx(n, i), "find_max_normal != find_max_avx");
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_sse(n, i) == my::vector_avx::find_max_avx(n, i), "find_max_sse != find_max_avx");
#endif
#if (__SSE3__ || __SSE2__)
    BOOST_REQUIRE_MESSAGE(my::vector_avx::find_max_normal(n, i) == my::vector_avx::find_max_sse(n, i), "find_max_normal != find_max_sse");
#endif
}