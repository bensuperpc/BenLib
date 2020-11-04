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
//  Created: 6, October, 2020                               //
//  Modified: 4, November, 2020                             //
//  file: bench_isprime.cpp                                 //
//  Benchmark CPU with Optimization                         //
//  Source: -                                               //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <thread>
#include <vector>
#include "../../lib/math/prime.hpp"
#include "../../lib/time/chrono/chrono.hpp"
#include "../../lib/vector/vector.hpp"

#if __cplusplus >= 201703L
#    include "../../lib/thread/Pool.hpp"
#else
#    include "../../lib/thread/ThreadPool.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>

struct Processor
{
    // long double operator()(bool(*elem_fn)(const long long int &), long long int prime_nbr)
    long double operator()(std::function<bool(const long long int &)> elem_fn, long long int prime_nbr)
    {
        auto &&t1 = my::chrono::now();
        if ((elem_fn)(prime_nbr) != true) {
            std::cout << "ERROR, is prime NBR: " << prime_nbr << std::endl;
        }
        auto &&t2 = my::chrono::now();
        return my::chrono::duration(t1, t2).count();
        // std::this_thread::sleep_for(std::my::math::seconds(1));
    }
};

// https://bigprimes.org/
// https://stackoverflow.com/questions/41749792/can-i-have-a-stdvector-of-template-function-pointers

int main()
{
    // std::vector<std::vector<std::future<std::pair<std::string, long double>>>> results {};
    std::vector<std::pair<std::string, std::vector<std::future<long double>>>> results {};

#if __cplusplus >= 201703L
    thread::Pool thread_pool(std::thread::hardware_concurrency() / 2);
#else
    ThreadPool thread_pool(std::thread::hardware_concurrency() / 2);
#endif
    const std::vector<long long int> prime_nbrs
        = {7, 29, 151, 4219, 40829, 251857, 4000037, 40000003, 400000009, 400000009, 4000000007, 40000000003, 400000000019, 999999999989, 4312512644831,
            67280421310721, 369822932657561, 1323784290040759, 67428322156073819, 979025471535264563, 3815136756226794067};
    const std::vector<std::pair<const std::string, bool (*)(const long long int &)>> pointer_map {{"isPrime_opti_1", &my::math::prime::isPrime_opti_1},
        {"isPrime_opti_2", &my::math::prime::isPrime_opti_2}, {"isPrime_opti_3", &my::math::prime::isPrime_opti_3},
        {"isPrime_opti_4", &my::math::prime::isPrime_opti_4},
        {"isPrime_opti_5", &my::math::prime::isPrime_opti_5}, //{"isPrime_opti_5T", &my::math::prime::isPrime_opti_5<long long int>},
        {"isPrime_opti_6", &my::math::prime::isPrime_opti_6}, {"isPrime_opti_8(3&5)", &my::math::prime::isPrime_opti_8}};

    // Generate poolthreading
    results.reserve(pointer_map.size());
    for (auto &elem_fn : pointer_map) {
        results.emplace_back(std::pair<std::string, std::vector<std::future<long double>>>());
        results.back().first = elem_fn.first;
        results.back().second.reserve(prime_nbrs.size());
        for (auto &prime_nbr : prime_nbrs) {
            results.back().second.emplace_back(thread_pool.enqueue(Processor(), elem_fn.second, prime_nbr));
        }
    }

    // Get result
    std::vector<std::pair<std::string, std::vector<long double>>> time {};
    time.reserve(results.size());
    for (auto &y : results) {
        time.emplace_back(std::pair<std::string, std::vector<long double>>());
        time.back().second.reserve(y.second.size());
        time.back().first = y.first;
        for (auto &x : y.second) {
            time.back().second.emplace_back(x.get());
        }
    }

    std::ifstream in("in.txt");
    std::cin.tie(0);
    auto cinbuf = std::cin.rdbuf(in.rdbuf()); // save and redirect

    std::ofstream out("log_bench_is_prime.csv");
    std::ios_base::sync_with_stdio(false);

    auto coutbuf = std::cout.rdbuf(out.rdbuf()); // save and redirect

    // Header
    std::cout << "number,";
    for (size_t x = 0; x < time.size(); x++) {
        std::cout << time[x].first;
        if (x < time.size() - 1) {
            std::cout << ",";
        }
    }
    std::cout << std::setprecision(10) << std::fixed;
    std::cout << std::endl;

    for (size_t x = 0; x < time.back().second.size(); x++) {
        std::cout << x << ",";
        for (size_t y = 0; y < time.size(); y++) {
            std::cout << time[y].second[x];
            if (y < time.size() - 1) {
                std::cout << ",";
            }
        }
        std::cout << std::endl;
    }

    std::cin.rdbuf(cinbuf);
    std::cout.rdbuf(coutbuf);
}
/*
int main()
{

    ThreadPool pool(4);
    std::vector< std::future<int> > results;

    for(int i = 0; i < 8; ++i) {
        results.emplace_back(
            pool.enqueue([i] {
                std::cout << "hello " << i << std::endl;
                std::this_thread::sleep_for(std::my::math::seconds(1));
                std::cout << "world " << i << std::endl;
                return i*i;
            })
        );
    }

    for(auto && result: results)
        std::cout << result.get() << ' ';
    std::cout << std::endl;

    return 0;
}
*/
