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
//  Modified: 7, October, 2020                              //
//  file: bench_digits_count.cpp                            //
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
#include "math/count_digits_imp.hpp"
#include "time/chrono/chrono.hpp"
#include "vector/vector.hpp"

#if __cplusplus >= 201703L
#    include "thread/Pool.hpp"
#else
#    include "thread/ThreadPool.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>

// https://stackoverflow.com/questions/2219829/how-to-prevent-gcc-optimizing-some-statements-in-c
struct Processor
{
    long double __attribute__((optimize("O2"))) operator()(std::function<uint64_t(uint64_t)> elem_fn, uint64_t prime_nbr)
    {
        auto nbrs __attribute__((unused)) = (elem_fn)(prime_nbr);
        auto &&t1 = my::chrono::now();
        for (uint64_t i = 0; i < 50000; i++) {
            nbrs = (elem_fn)(prime_nbr);
        }
        auto &&t2 = my::chrono::now();
        return my::chrono::duration(t1, t2).count() / 50000.0;
        // std::this_thread::sleep_for(std::my::chrono::seconds(1));
    }
};

// https://bigprimes.org/
// https://stackoverflow.com/questions/41749792/can-i-have-a-stdvector-of-template-function-pointers

int main(int argc, char *argv[], char *envp[])
{
    // std::vector<std::vector<std::future<std::pair<std::string, long double>>>> results {};
    std::vector<std::pair<std::string, std::vector<std::future<long double>>>> results {};

#if __cplusplus >= 201703L
    thread::Pool thread_pool(std::thread::hardware_concurrency() / 2);
#else
    ThreadPool thread_pool(std::thread::hardware_concurrency() / 2);
#endif
    std::vector<uint64_t> prime_nbrs = std::vector<uint64_t>(1000);

    my::vector::fill_rowull(prime_nbrs);
    std::sort(prime_nbrs.begin(), prime_nbrs.end());

    const std::vector<std::pair<const std::string, uint64_t (*)(uint64_t)>> pointer_map {{"count_digits_1", &my::math::count_digits::count_digits_1<uint64_t>},
        {"count_digits_2", &my::math::count_digits::count_digits_2<uint64_t>}, {"count_digits_3", &my::math::count_digits::count_digits_3<uint64_t>},
        {"count_digits_4", &my::math::count_digits::count_digits_4<uint64_t>}};

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

    std::ofstream out("log_bench_digits_count.csv");
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
    return EXIT_SUCCESS;
}