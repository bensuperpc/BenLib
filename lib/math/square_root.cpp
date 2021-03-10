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

float math::square_root::sqrt(float x)
{
    return std::sqrt(x);
}
double math::square_root::sqrt(double x)
{
    return std::sqrt(x);
}