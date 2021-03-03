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
//  file: bench_find_max.cpp                                //
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
#include "time/chrono/chrono.hpp"
#include "vector/vector.hpp"
#include "vector/vector_avx.hpp"

#if __cplusplus >= 201703L
#    include "thread/Pool.hpp"
#else
#    include "thread/ThreadPool.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>

struct Processor
{
    long double operator()(std::function<int(const int32_t *, size_t)> elem_fn, const int *array, const size_t &array_size, const int &nrbs)
    {
        int result __attribute__((unused)) = 0;

        // Prepare CPU
        for (auto k = 0; k < nrbs / 10; k++) {
            result = (elem_fn)(array, array_size);
        }

        // Real Bench
        auto &&t1 = my::chrono::now();
        for (auto k = 0; k < nrbs; k++) {
            result = (elem_fn)(array, array_size);
        }
        auto &&t2 = my::chrono::now();
        return my::chrono::duration(t1, t2).count();
    }
};

int main(int argc, char *argv[], char *envp[])
{
    std::vector<std::pair<std::string, std::vector<std::future<long double>>>> results2 {};
#if __cplusplus >= 201703L
    thread::Pool thread_pool(1); // Use 1 thread to improve results stability (avoid some spikes on graph)
#else
    ThreadPool thread_pool(1); // Use 1 thread to improve results stability (avoid some spikes on graph)
#endif
    const size_t N = 20;
    const size_t S = 0;
    const int nofTestCases = 100000;

    int **nx = new int *[N];
    std::cout << std::setprecision(10) << std::fixed;
    for (size_t x = 0; x < N; ++x) {
        nx[x] = new int[(int)std::pow(2, x + S)];
        std::cout << (int)std::pow(2, x + S) << std::endl;
    }

    for (size_t x = 0; x < N; ++x) {
        for (size_t y = 0; y < (size_t)std::pow(2, x); ++y) {
            nx[x][y] = rand() % 2000000000;
        }
    }
    std::vector<std::pair<const size_t, int *>> n_ptr = {};
    for (size_t x = 0; x < N; ++x) {
        n_ptr.emplace_back(std::make_pair((size_t)std::pow(2, x + S), nx[x]));
    }

    // Declare point to function
    const std::vector<std::pair<const std::string, std::function<int(const int32_t *, size_t)>>> pointer_fn_map
    {
        {"find_max_normal", &my::vector_avx::find_max_normal}, {"find_max_sse", &my::vector_avx::find_max_sse},
        {
            "find_max_avx", &my::vector_avx::find_max_avx
        }
#ifdef __AVX512F__
#    if (__AVX512F__)
        ,
        {
            "find_max_avx512", &my::vector_avx::find_max_avx512
        }
    };
#    endif
#else
    };
#endif
    // int n1[n * 2] __attribute__((aligned(32)));
    results2.reserve(pointer_fn_map.size());

    for (auto &elem_fn : pointer_fn_map) {
        results2.emplace_back(std::pair<std::string, std::vector<std::future<long double>>>());
        results2.back().first = elem_fn.first;
        results2.back().second.reserve(n_ptr.size());
        for (auto &table : n_ptr) {
            results2.back().second.emplace_back(thread_pool.enqueue(Processor(), elem_fn.second, table.second, table.first, nofTestCases));
        }
    }

    std::ios_base::sync_with_stdio(false);
    std::cout << "Pool threading: OK" << std::endl;
    // std::this_thread::sleep_for(std::my::math::milliseconds(20000));
    // Get result
    std::cout << std::setprecision(3) << std::fixed;
    size_t count = 0;
    const size_t xElem = 1;
    std::vector<std::pair<std::string, std::vector<long double>>> time {};
    time.reserve(results2.size());
    for (auto &y : results2) {
        time.emplace_back(std::pair<std::string, std::vector<long double>>());
        time.back().second.reserve(y.second.size());
        time.back().first = y.first;
        for (auto &x : y.second) {
            time.back().second.emplace_back(x.get());
            count++;
            if (count % xElem == 0) {
                std::cout << double(count) / double(n_ptr.size() * pointer_fn_map.size()) * 100.0f << " %" << std::endl;
            }
        }
    }
    std::cout << "Get data: OK" << std::endl;

    std::ifstream in("in.txt");
    std::cin.tie(0);
    auto cinbuf = std::cin.rdbuf(in.rdbuf()); // save and redirect

    std::ofstream out("log_bench_find_max.csv");
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
        std::cout << n_ptr[x].first << ",";
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
    for (size_t i = 0; i < N; ++i) {
        delete nx[i];
    }
    delete nx;
    return EXIT_SUCCESS;
}