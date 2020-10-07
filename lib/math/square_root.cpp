/*
** BENSUPERPC PROJECT, 2020
** Math
** File description:
** square_root.hpp
*/

#include "square_root.hpp"
float math::square_root::invsqrt(float x)
{
    float xhalf = 0.5f * x;
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
    int i = *(int *)&x;
    i = MagicNBR_32 - (i >> 1);
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
    x = *(float *)&i;
    x = x * (1.5f - xhalf * x * x);
    // x  = x * (1.5 - ( xhalf * x * x ));   // 2nd iteration, this can be removed
    return x;
}

double math::square_root::invsqrt(double x)
{
    double y = x;
    double x2 = y * 0.5;
    std::int64_t i = *(std::int64_t *)&y;
    i = MagicNBR_64 - (i >> 1);
    y = *(double *)&i;
    y = y * (1.5 - (x2 * y * y));
    // y  = y * ( 1.5 - ( x2 * y * y ) );   // 2nd iteration, this can be removed
    return y;
}

// THANK https://stackoverflow.com/a/59248244/10152334
template <typename T, char iterations = 2> T math::square_root::invsqrt(T x)
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

template float math::square_root::invsqrt<float>(float x);
template double math::square_root::invsqrt<double>(double x);

float math::square_root::sqrt(float x)
{
    return std::sqrt(x);
}
double math::square_root::sqrt(double x)
{
    return std::sqrt(x);
}