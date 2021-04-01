/**
 * @file float.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */
#include "float.hpp"

bool my::math::fp::are_aqual(double &x, double &y)
{
    double maxXYOne = std::max({1.0, std::fabs(x), std::fabs(y)});
    return std::fabs(x - y) <= std::numeric_limits<double>::epsilon() * maxXYOne;
}