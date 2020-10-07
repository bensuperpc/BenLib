/*
** BENSUPERPC PROJECT, 2020
** RPG
** File description:
** getGravitationalAttraction.cpp
*/

#ifndef GETGRAVITATIONATTRACTION_HPP
#define GETGRAVITATIONATTRACTION_HPP

#include "constant.hpp"
namespace my
{
namespace math
{
namespace ga
{
template <typename T> T getGravitationalAttraction(const T &m1, const T &m2, const T &d);
} // namespace ga
} // namespace math
} // namespace my
#include "getGravitationalAttraction_imp.hpp"

#endif
