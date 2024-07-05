/**
 * @file power.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2021-04-01
 *
 * MIT License
 *
 */

#ifndef BENLIB_MATH_POWER_HPP_
#define BENLIB_MATH_POWER_HPP_
namespace benlib {
namespace math {
/**
 * @brief
 *
 * @tparam T
 * @param nbr
 * @param pow
 * @return T
 */
template <typename T>
auto power(const T& nb, const long int& p) noexcept -> T {
    if (p < 0)
        return (0);
    if (p != 0)
        return (nb * power(nb, p - 1));
    else
        return 1;
}

/**
 * @brief
 *
 * @tparam T
 * @param nbr
 * @return true
 * @return false
 */
template <typename T>
auto isPowerOfTwo(const T& x) noexcept -> bool {
    return x && (!(x & (x - 1)));
}
}  // namespace math
}  // namespace benlib
#endif
