/*
** BENSUPERPC PROJECT, 2020
** Math
** File description:
** square_root.hpp
*/

#include "square_root.hpp"

// THANK https://stackoverflow.com/a/59248244/10152334
template <typename T, char iterations> T math::square_root::invsqrt(T x)
{
    static_assert(std::is_floating_point<T>::value, "T must be floating point");
    static_assert(iterations == 1 or iterations == 2, "itarations must equal 1 or 2");
    typedef typename std::conditional<sizeof(T) == 8, std::int64_t, std::int32_t>::type Tint;
    T y = x;
    T x2 = y * 0.5;
    Tint i = *(Tint *)&y;
    i = (sizeof(T) == 8 ? MagicNBR_64 : MagicNBR_32) - (i >> 1);
    y = *(T *)&i;
    y = y * (1.5 - (x2 * y * y));
    if (iterations == 2)
        y = y * (1.5 - (x2 * y * y));
    return y;
}