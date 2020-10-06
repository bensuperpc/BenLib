#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE cylinder

#include <boost/test/unit_test.hpp>
//#include <boost/multiprecision/cpp_int.hpp>
// boost::multiprecision::uint1024_t i = 0;

#include "../../../math/cylinder.hpp"

BOOST_AUTO_TEST_CASE(test_cylinder_volume_1)
{
    BOOST_CHECK_MESSAGE(my::math::cylinder::cylinderVolume<uint64_t>(1000, 5) == static_cast<uint64_t>(15707963),
        my::math::cylinder::cylinderVolume<uint64_t>(1000, 5) << " instead: " << static_cast<uint64_t>(15707963));

    BOOST_CHECK_MESSAGE(my::math::cylinder::cylinderVolume<uint64_t>(151, 10) == static_cast<uint64_t>(716314),
        my::math::cylinder::cylinderVolume<uint64_t>(151, 10) << " instead: " << static_cast<uint64_t>(716314));
    BOOST_CHECK_MESSAGE(my::math::cylinder::cylinderVolume<uint64_t>(0, 0) == static_cast<uint64_t>(0), my::math::cylinder::cylinderVolume<uint64_t>(0, 0)
                                                                                                            << " instead: " << static_cast<uint64_t>(0));
}

BOOST_AUTO_TEST_CASE(test_cylinder_volume_2)
{
    BOOST_CHECK_MESSAGE(my::math::cylinder::cylinderVolume<int>(32, 3) == static_cast<int>(9650), my::math::cylinder::cylinderVolume<int>(32, 3)
                                                                                                      << " instead: " << static_cast<int>(9650));
    BOOST_CHECK_MESSAGE(my::math::cylinder::cylinderVolume<int>(151, 2) == static_cast<int>(143262), my::math::cylinder::cylinderVolume<int>(151, 2)
                                                                                                         << " instead: " << static_cast<int>(143262));
    BOOST_CHECK_MESSAGE(my::math::cylinder::cylinderVolume<int>(0, 0) == static_cast<int>(0), my::math::cylinder::cylinderVolume<int>(0, 0)
                                                                                                  << " instead: " << static_cast<int>(0));
}

BOOST_AUTO_TEST_CASE(test_cylinder_surface_1)
{
    BOOST_CHECK_MESSAGE(my::math::cylinder::cylinderSurface<uint64_t>(10000, 9) == static_cast<uint64_t>(628884017),
        my::math::cylinder::cylinderSurface<uint64_t>(10000, 9) << " instead: " << static_cast<uint64_t>(628884017));
    BOOST_CHECK_MESSAGE(my::math::cylinder::cylinderSurface<uint64_t>(151, 6) == static_cast<uint64_t>(148955),
        my::math::cylinder::cylinderSurface<uint64_t>(151, 6) << " instead: " << static_cast<uint64_t>(148955));
    BOOST_CHECK_MESSAGE(my::math::cylinder::cylinderSurface<uint64_t>(0, 0) == static_cast<uint64_t>(0), my::math::cylinder::cylinderSurface<uint64_t>(0, 0)
                                                                                                             << " instead: " << static_cast<uint64_t>(0));
}

BOOST_AUTO_TEST_CASE(test_cylinder_surface_2)
{
    BOOST_CHECK_MESSAGE(my::math::cylinder::cylinderSurface<int>(6598, 4) == static_cast<int>(273695526), my::math::cylinder::cylinderSurface<int>(6598, 4)
                                                                                                              << " instead: " << static_cast<int>(273695526));
    BOOST_CHECK_MESSAGE(my::math::cylinder::cylinderSurface<int>(151, 6) == static_cast<int>(148955), my::math::cylinder::cylinderSurface<int>(151, 6)
                                                                                                          << " instead: " << static_cast<int>(148955));
    BOOST_CHECK_MESSAGE(my::math::cylinder::cylinderSurface<int>(0, 0) == static_cast<int>(0), my::math::cylinder::cylinderSurface<int>(0, 0)
                                                                                                   << " instead: " << static_cast<int>(0));
}
