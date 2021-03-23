#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE vector_copy

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
#include "vector/vector_imp.hpp"

BOOST_AUTO_TEST_CASE(test_vector_cache_friendly_copy_1)
{
    std::vector<uint64_t> mat2(50);
    std::iota(mat2.begin(), mat2.end(), 0);

    auto mat1 = mat2;
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(mat2), std::end(mat2), rng);

    // BOOST_FAIL( "Test is not ready yet" );
    BOOST_REQUIRE_MESSAGE(mat1 != mat2, "mat1 == mat2"
                                            << " instead: "
                                            << "mat1 != mat2");
    my::vector::cache_friendly_copy<uint64_t>(mat1, mat2);
    BOOST_REQUIRE_EQUAL_COLLECTIONS(mat2.begin(), mat2.end(), mat1.begin(), mat1.end());
}

BOOST_AUTO_TEST_CASE(test_vector_cache_friendly_copy_2)
{
    std::vector<uint64_t> mat2(200);
    std::iota(mat2.begin(), mat2.end(), 0);

    auto mat1 = mat2;
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(mat2), std::end(mat2), rng);

    // BOOST_FAIL( "Test is not ready yet" );
    BOOST_REQUIRE_MESSAGE(mat1 != mat2, "mat1 == mat2"
                                            << " instead: "
                                            << "mat1 != mat2");
    my::vector::cache_friendly_copy<uint64_t>(mat1, mat2);
    BOOST_REQUIRE_EQUAL_COLLECTIONS(mat2.begin(), mat2.end(), mat1.begin(), mat1.end());
}

BOOST_AUTO_TEST_CASE(test_vector_cache_friendly_copy_2d_1)
{
    auto &&mat1 = my::vector::generate_matrix<uint64_t>(50, 50, 0);
    auto &&mat2 = my::vector::generate_matrix<uint64_t>(50, 50, 42);

    // BOOST_FAIL( "Test is not ready yet" );
    BOOST_REQUIRE_MESSAGE(mat1 != mat2, "mat1 == mat2"
                                            << " instead: "
                                            << "mat1 != mat2");
    my::vector::cache_friendly_copy<uint64_t>(mat1, mat2);
    BOOST_REQUIRE_MESSAGE(mat1 == mat2, "mat1 != mat2"
                                            << " instead: "
                                            << "mat1 == mat2");
}

BOOST_AUTO_TEST_CASE(test_vector_cache_friendly_copy_2d_2)
{
    auto &&mat1 = my::vector::generate_matrix<uint64_t>(50, 50, 0);
    auto &&mat2 = my::vector::generate_matrix<uint64_t>(50, 50, 42);

    // BOOST_FAIL( "Test is not ready yet" );
    BOOST_REQUIRE_MESSAGE(mat1 != mat2, "mat1 == mat2"
                                            << " instead: "
                                            << "mat1 != mat2");
    my::vector::cache_friendly_copy<uint64_t>(mat1, mat2);
    // BOOST_REQUIRE_EQUAL_COLLECTIONS(mat2.begin(), mat2.end(),mat1.begin(), mat1.end());
    BOOST_REQUIRE_MESSAGE(mat1 == mat2, "mat1 != mat2"
                                            << " instead: "
                                            << "mat1 == mat2");
}
/*
BOOST_AUTO_TEST_CASE(test_vector_std_copy_2d_1)
{
    auto &&mat1 = my::vector::generate_matrix<uint64_t>(50, 50, 0);
    auto &&mat2 = my::vector::generate_matrix<uint64_t>(50, 50, 42);

    // BOOST_FAIL( "Test is not ready yet" );
    BOOST_REQUIRE_MESSAGE(mat1 != mat2, "mat1 == mat2"
                                            << " instead: "
                                            << "mat1 != mat2");
    my::vector::std_copy<uint64_t>(mat1, mat2);
    BOOST_REQUIRE_MESSAGE(mat1 == mat2, "mat1 != mat2"
                                            << " instead: "
                                            << "mat1 == mat2");
}
*/