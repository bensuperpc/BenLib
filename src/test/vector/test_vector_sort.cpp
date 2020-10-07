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

BOOST_AUTO_TEST_CASE(test_vector_sort_qsort_1)
{
    std::vector<uint64_t> unordered(50);
    std::iota(unordered.begin(), unordered.end(), 0);

    auto ordered = unordered;
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(unordered), std::end(unordered), rng);
    BOOST_REQUIRE_MESSAGE(ordered != unordered, "ordered == unordered"
                                                    << " instead: "
                                                    << "ordered != unordered");
    my::vector::sort_qsort<uint64_t>(unordered);

    BOOST_REQUIRE_EQUAL_COLLECTIONS(unordered.begin(), unordered.end(), ordered.begin(), ordered.end());
}

BOOST_AUTO_TEST_CASE(test_vector_sort_qsort_2)
{
    std::vector<uint64_t> unordered(200);
    std::iota(unordered.begin(), unordered.end(), 0);

    auto ordered = unordered;
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(unordered), std::end(unordered), rng);
    BOOST_REQUIRE_MESSAGE(ordered != unordered, "ordered == unordered"
                                                    << " instead: "
                                                    << "ordered != unordered");
    my::vector::sort_qsort<uint64_t>(unordered);

    BOOST_REQUIRE_EQUAL_COLLECTIONS(unordered.begin(), unordered.end(), ordered.begin(), ordered.end());
}

BOOST_AUTO_TEST_CASE(test_vector_sort_stable_sort_1)
{
    std::vector<uint64_t> unordered(50);
    std::iota(unordered.begin(), unordered.end(), 0);

    auto ordered = unordered;
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(unordered), std::end(unordered), rng);
    BOOST_REQUIRE_MESSAGE(ordered != unordered, "ordered == unordered"
                                                    << " instead: "
                                                    << "ordered != unordered");
    my::vector::sort_stable_sort<uint64_t>(unordered);

    BOOST_REQUIRE_EQUAL_COLLECTIONS(unordered.begin(), unordered.end(), ordered.begin(), ordered.end());
}

BOOST_AUTO_TEST_CASE(test_vector_sort_stable_sort_2)
{
    std::vector<uint64_t> unordered(200);
    std::iota(unordered.begin(), unordered.end(), 0);

    auto ordered = unordered;
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(unordered), std::end(unordered), rng);
    BOOST_REQUIRE_MESSAGE(ordered != unordered, "ordered == unordered"
                                                    << " instead: "
                                                    << "ordered != unordered");
    my::vector::sort_stable_sort<uint64_t>(unordered);

    BOOST_REQUIRE_EQUAL_COLLECTIONS(unordered.begin(), unordered.end(), ordered.begin(), ordered.end());
}

BOOST_AUTO_TEST_CASE(test_vector_sort_bubble_1)
{
    std::vector<uint64_t> unordered(50);
    std::iota(unordered.begin(), unordered.end(), 0);

    auto ordered = unordered;
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(unordered), std::end(unordered), rng);
    BOOST_REQUIRE_MESSAGE(ordered != unordered, "ordered == unordered"
                                                    << " instead: "
                                                    << "ordered != unordered");
    my::vector::sort_bubble<uint64_t>(unordered);

    BOOST_REQUIRE_EQUAL_COLLECTIONS(unordered.begin(), unordered.end(), ordered.begin(), ordered.end());
}

BOOST_AUTO_TEST_CASE(test_vector_sort_bubble_2)
{
    std::vector<uint64_t> unordered(200);
    std::iota(unordered.begin(), unordered.end(), 0);

    auto ordered = unordered;
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(unordered), std::end(unordered), rng);
    BOOST_REQUIRE_MESSAGE(ordered != unordered, "ordered == unordered"
                                                    << " instead: "
                                                    << "ordered != unordered");
    my::vector::sort_bubble<uint64_t>(unordered);

    BOOST_REQUIRE_EQUAL_COLLECTIONS(unordered.begin(), unordered.end(), ordered.begin(), ordered.end());
}

BOOST_AUTO_TEST_CASE(test_vector_sort_gnome_1)
{
    std::vector<uint64_t> unordered(50);
    std::iota(unordered.begin(), unordered.end(), 0);

    auto ordered = unordered;
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(unordered), std::end(unordered), rng);
    BOOST_REQUIRE_MESSAGE(ordered != unordered, "ordered == unordered"
                                                    << " instead: "
                                                    << "ordered != unordered");
    my::vector::sort_gnome<uint64_t>(unordered);

    BOOST_REQUIRE_EQUAL_COLLECTIONS(unordered.begin(), unordered.end(), ordered.begin(), ordered.end());
}

BOOST_AUTO_TEST_CASE(test_vector_sort_gnome_2)
{
    std::vector<uint64_t> unordered(200);
    std::iota(unordered.begin(), unordered.end(), 0);

    auto ordered = unordered;
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(unordered), std::end(unordered), rng);
    BOOST_REQUIRE_MESSAGE(ordered != unordered, "ordered == unordered"
                                                    << " instead: "
                                                    << "ordered != unordered");
    my::vector::sort_gnome<uint64_t>(unordered);

    BOOST_REQUIRE_EQUAL_COLLECTIONS(unordered.begin(), unordered.end(), ordered.begin(), ordered.end());
}

BOOST_AUTO_TEST_CASE(test_vector_sort_insertion_1)
{
    std::vector<uint64_t> unordered(50);
    std::iota(unordered.begin(), unordered.end(), 0);

    auto ordered = unordered;
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(unordered), std::end(unordered), rng);
    BOOST_REQUIRE_MESSAGE(ordered != unordered, "ordered == unordered"
                                                    << " instead: "
                                                    << "ordered != unordered");
    my::vector::sort_insertion<uint64_t>(unordered);

    BOOST_REQUIRE_EQUAL_COLLECTIONS(unordered.begin(), unordered.end(), ordered.begin(), ordered.end());
}

BOOST_AUTO_TEST_CASE(test_vector_sort_insertion_2)
{
    std::vector<uint64_t> unordered(200);
    std::iota(unordered.begin(), unordered.end(), 0);

    auto ordered = unordered;
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(unordered), std::end(unordered), rng);
    BOOST_REQUIRE_MESSAGE(ordered != unordered, "ordered == unordered"
                                                    << " instead: "
                                                    << "ordered != unordered");
    my::vector::sort_insertion<uint64_t>(unordered);

    BOOST_REQUIRE_EQUAL_COLLECTIONS(unordered.begin(), unordered.end(), ordered.begin(), ordered.end());
}

BOOST_AUTO_TEST_CASE(test_vector_sort_shell_1)
{
    std::vector<uint64_t> unordered(50);
    std::iota(unordered.begin(), unordered.end(), 0);

    auto ordered = unordered;
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(unordered), std::end(unordered), rng);
    BOOST_REQUIRE_MESSAGE(ordered != unordered, "ordered == unordered"
                                                    << " instead: "
                                                    << "ordered != unordered");
    my::vector::sort_shell<uint64_t>(unordered);

    BOOST_REQUIRE_EQUAL_COLLECTIONS(unordered.begin(), unordered.end(), ordered.begin(), ordered.end());
}

BOOST_AUTO_TEST_CASE(test_vector_sort_shell_2)
{
    std::vector<uint64_t> unordered(200);
    std::iota(unordered.begin(), unordered.end(), 0);

    auto ordered = unordered;
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(unordered), std::end(unordered), rng);
    BOOST_REQUIRE_MESSAGE(ordered != unordered, "ordered == unordered"
                                                    << " instead: "
                                                    << "ordered != unordered");
    my::vector::sort_shell<uint64_t>(unordered);

    BOOST_REQUIRE_EQUAL_COLLECTIONS(unordered.begin(), unordered.end(), ordered.begin(), ordered.end());
}
