/*
** BENSUPERPC PROJECT, 2020
** Math
** File description:
** float.hpp
*/

#include "float.hpp"

bool my::math::fp::are_aqual(double &x, double &y)
{
    double maxXYOne = std::max( { 1.0, std::fabs(x) , std::fabs(y) } ) ;
    return std::fabs(x - y) <= std::numeric_limits<double>::epsilon()*maxXYOne;
}