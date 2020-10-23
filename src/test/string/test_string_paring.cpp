#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE vector_sort

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
#include "../../../lib/vector/vector.hpp"

BOOST_AUTO_TEST_CASE(test_vector_sort_sort_1)
{
    std::vector<uint64_t> unordered(50);
    std::iota(unordered.begin(), unordered.end(), 0);

    auto ordered = unordered;
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(unordered), std::end(unordered), rng);
    // BOOST_FAIL( "Test is not ready yet" );
    BOOST_REQUIRE_MESSAGE(ordered != unordered, "ordered == unordered"
                                                    << " instead: "
                                                    << "ordered != unordered");
    my::vector::sort_sort<uint64_t>(unordered);

    BOOST_REQUIRE_EQUAL_COLLECTIONS(unordered.begin(), unordered.end(), ordered.begin(), ordered.end());
}

BOOST_AUTO_TEST_CASE(test_vector_sort_sort_2)
{
    std::vector<uint64_t> unordered(200);
    std::iota(unordered.begin(), unordered.end(), 0);

    auto ordered = unordered;
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(unordered), std::end(unordered), rng);
    BOOST_REQUIRE_MESSAGE(ordered != unordered, "ordered == unordered"
                                                    << " instead: "
                                                    << "ordered != unordered");
    my::vector::sort_sort<uint64_t>(unordered);

    BOOST_REQUIRE_EQUAL_COLLECTIONS(unordered.begin(), unordered.end(), ordered.begin(), ordered.end());
}
