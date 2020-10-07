/*
** BENSUPERPC PROJECT, 2020
** Math
** File description:
** cylinder.cpp
*/

#ifndef CYLINDER_HPP_
#define CYLINDER_HPP_

#include "constant.hpp"

#if __cplusplus == 202002L
#    include <numbers>
#endif

namespace my
{
namespace math
{

namespace cylinder
{
template <typename T> T cylinderSurface(const T &, const T &);
template <typename T> T cylinderVolume(const T &, const T &);
} // namespace cylinder
} // namespace math
} // namespace my
#include "cylinder_imp.hpp"

#endif
