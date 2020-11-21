/*
** BENSUPERPC PROJECT, 2020
** Math
** File description:
** float.hpp
*/

#include "random.hpp"

double my::math::rand::rand(double fMin, double fMax)
{
    std::uniform_real_distribution<double> dist(fMin, fMax);

    std::mt19937 rng;
    rng.seed(std::random_device {}());
    return dist(rng);
}