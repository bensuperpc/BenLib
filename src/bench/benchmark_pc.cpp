/*
** BENSUPERPC PROJECT, 2020
** Texture
** File description:
** benchmark_pc.cpp
*/
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <time.h>
#include <vector>
#include "../../lib/time/chrono/chrono.hpp"
#include "../../lib/vector/vector.hpp"

#define NBRS 10000000

template <typename Type> Type divide(Type a, Type b)
{
    double da = (double)a;
    double db = (double)b;
    double q = da / db;
    return (Type)q;
}

template <typename Type> void my_test(const char *name)
{
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};
    std::vector<Type> &&v = std::vector<Type>(NBRS, (Type)5);
    std::vector<Type> &&t = std::vector<Type>(NBRS, (Type)7);
    my::vector::rnd_fill<Type>(t);
    my::vector::rnd_fill<Type>(v);
    std::shuffle(begin(t), end(t), mersenne_engine);
    std::shuffle(begin(v), end(v), mersenne_engine);
    auto &&t1 = my::chrono::now();
    for (size_t i = 0; i < NBRS; ++i) {
        v[i] += t[i];
    }
    auto &&t2 = my::chrono::now();
    std::cout << name << " add: " << (((double)NBRS / my::chrono::duration(t1, t2).count()))/ 1000000.0f << " MIPS" << std::endl;

    t1 = my::chrono::now();
    for (size_t i = 0; i < NBRS; ++i) {
        v[i] -= t[i];
    }
    t2 = my::chrono::now();
    // typeid(Type).name()
    std::cout << name << " sub: " << ((double)NBRS / my::chrono::duration(t1, t2).count())/ 1000000.0f << " MIPS" << std::endl;

    t1 = my::chrono::now();
    for (size_t i = 0; i < NBRS; ++i) {
        v[i] *= t[i];
    }
    t2 = my::chrono::now();
    std::cout << name << " mul: " << ((double)NBRS / my::chrono::duration(t1, t2).count())/ 1000000.0f << " MIPS" << std::endl;

    t1 = my::chrono::now();
    for (size_t i = 0; i < NBRS; ++i) {
        v[i] /= t[i];
    }
    t2 = my::chrono::now();
    std::cout << name << " div: " << ((double)NBRS / my::chrono::duration(t1, t2).count())/ 1000000.0f << " MIPS" << std::endl;
    if constexpr (std::is_integral<Type>::value) {
        t1 = my::chrono::now();
        for (size_t i = 0; i < NBRS; ++i) {
            v[i] %= t[i];
        }
        t2 = my::chrono::now();
        std::cout << name << " mod: " << ((double)NBRS / my::chrono::duration(t1, t2).count())/ 1000000.0f << " MIPS" << std::endl;
    }
}

int main()
{
    my_test<      int8_t>("     int8_t");
    my_test<     int16_t>("    int16_t");
    my_test<     int32_t>("    int32_t");
    my_test<     int64_t>("    int64_t");
    my_test<float>("      float");
    my_test<double>("     double");
    my_test<long double>("long double");
    return 0;
}
