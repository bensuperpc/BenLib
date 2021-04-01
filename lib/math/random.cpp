/**
 * @file random.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

#include "random.hpp"

double my::math::rand::rand(double fMin, double fMax)
{
    std::uniform_real_distribution<double> dist(fMin, fMax);

    std::mt19937 rng;
    rng.seed(std::random_device {}());
    return dist(rng);
}