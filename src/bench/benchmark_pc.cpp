
//////////////////////////////////////////////////////////////
//   ____                                                   //
//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
//                             |_|             |_|          //
//////////////////////////////////////////////////////////////
//                                                          //
//  BenLib, 2020                                            //
//  Created: 30, October, 2020                              //
//  Modified: 4, November, 2020                             //
//  file: benchmark_pc.cpp                                  //
//  Benchmark CPU with Optimization                         //
//  Source: -                                               //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <type_traits>
#include <vector>
#include "time/chrono/chrono.hpp"
#include "vector/vector.hpp"

#define NBRS 33554432 // 16777216

// Optimize devide for interger
template <typename Type> Type divide(Type a, Type b)
{
    double da = (double)a;
    double db = (double)b;
    double q = da / db;
    return (Type)q;
}

template <typename Type> inline void add(Type &t1, const Type &t2)
{
    /*
    if constexpr (std::integral<T>::value) { // is number
        return value;
    else
        return value.length();*/

    // t1 += t2;
    for (size_t i = 0; i < NBRS; ++i) {

        t1[i] += t2[i];
    }
}

template <typename Type> void /*__attribute__((optimize("O3")))*/ my_test(const char *name)
{
    // Create vector
    std::vector<Type> &&v = std::vector<Type>(NBRS, (Type)5);
    std::vector<Type> &&t = std::vector<Type>(NBRS, (Type)7);

    // Fill vector with random value
    my::vector::rnd_fill<Type>(t);
    my::vector::rnd_fill<Type>(v);

    // random place
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};
    std::shuffle(begin(t), end(t), mersenne_engine);
    std::shuffle(begin(v), end(v), mersenne_engine);
    auto &&t1 = my::chrono::now();
    add<std::vector<Type>>(v, t);
    auto &&t2 = my::chrono::now();
    std::cout << name << " add: " << (((double)NBRS / my::chrono::duration(t1, t2).count())) / 1000000000.0f << " GigaOps" << std::endl;

    t1 = my::chrono::now();
    for (size_t i = 0; i < NBRS; ++i) {
        v[i] -= t[i];
    }
    t2 = my::chrono::now();

    // typeid(Type).name()
    std::cout << name << " sub: " << ((double)NBRS / my::chrono::duration(t1, t2).count()) / 1000000000.0f << " GigaOps" << std::endl;

    t1 = my::chrono::now();
    for (size_t i = 0; i < NBRS; ++i) {
        v[i] *= t[i];
    }
    t2 = my::chrono::now();
    std::cout << name << " mul: " << ((double)NBRS / my::chrono::duration(t1, t2).count()) / 1000000000.0f << " GigaOps" << std::endl;

    t1 = my::chrono::now();
    for (size_t i = 0; i < NBRS; ++i) {
        v[i] /= t[i];
    }
    t2 = my::chrono::now();
    std::cout << name << " div: " << ((double)NBRS / my::chrono::duration(t1, t2).count()) / 1000000000.0f << " GigaOps" << std::endl;
    if constexpr (std::is_integral<Type>::value) {
        t1 = my::chrono::now();
        for (size_t i = 0; i < NBRS; ++i) {
            v[i] %= t[i];
        }
        t2 = my::chrono::now();
        std::cout << name << " mod: " << ((double)NBRS / my::chrono::duration(t1, t2).count()) / 1000000000.0f << " GigaOps" << std::endl;
    }
}
int main()
{
    my_test<int8_t>("     int8_t");
    my_test<int16_t>("    int16_t");
    my_test<int32_t>("    int32_t");
    my_test<int64_t>("    int64_t");
    my_test<float>("      float");
    my_test<double>("     double");
    my_test<long double>("long double");
    return EXIT_SUCCESS;
}
