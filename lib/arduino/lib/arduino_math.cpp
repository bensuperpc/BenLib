/**
 * @file arduino_math.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-02
 * 
 * MIT License
 * 
 */

#include "arduino_math.hpp"

long map(long x, long in_min, long in_max, long out_min, long out_max)
{
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

int map(const int x, const int in_min, const int in_max, const int out_min, const int out_max)
{
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

int constrain(const int x, const int a, const int b)
{
    if ((x <= b && x >= a) || (x >= b && x <= a)) {
        return x;
    }
    if (x < a) {
        return a;
    }
    if (x > a) {
        return b;
    }

    return 0;
}