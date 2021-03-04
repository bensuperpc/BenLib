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
//  Created: 4, March, 2021                                 //
//  Modified: 4, March, 2021                                //
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
#include "math/prime.hpp"
#include "time/chrono/chrono.hpp"
#include "vector/vector.hpp"

#include "crypto/crypto_CRC32.hpp"


#if __cplusplus >= 201703L
#    include "thread/Pool.hpp"
#else
#    include "thread/ThreadPool.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <string.h>  // <cstring> en C++
#include <omp.h>


struct Task
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


int main(int argc, char *argv[], char *envp[])
{
    // std::vector<std::vector<std::future<std::pair<std::string, long double>>>> results {};
    std::vector<std::pair<std::string, std::vector<std::future<long double>>>> results {};

#if __cplusplus >= 201703L
    thread::Pool thread_pool(std::thread::hardware_concurrency() / 2);
#else
    ThreadPool thread_pool(std::thread::hardware_concurrency() / 2);
#endif

    const size_t N = 30; // Nbrs tests
    const size_t P = 2; // Power
    const size_t S = 0;
    const unsigned char Min = 32; // ACSII = Space
    const unsigned char Max = 126; // ACSII = ~

    unsigned char **nx = new unsigned char *[N];

    #pragma omp parallel for num_threads(12)
    for (size_t x = 0; x < N; ++x) {
        nx[x] = new unsigned char[(std::size_t)std::pow(P, x + S)];
        //printf("Element %ld traité par le thread %d \n",x,omp_get_thread_num());
        #pragma omp critical
        std::cout << (size_t)std::pow(P, x + S) << " Elements" << std::endl;
        #pragma omp critical
        std::cout << ((std::pow(P, x + S)* 8) / (8 * 1024)) << " Ko" << "\n" << std::endl;
    }

    //#pragma omp parallel for collapse(2)
    //#pragma omp parallel for num_threads(12)
    //#pragma omp parallel for num_threads(2) collapse(2)
    //#pragma omp parallel for num_threads(4) schedule(dynamic, 100)
    //#pragma omp parallel for num_threads(12) private(x)

    //#pragma omp for
    //#pragma omp critical
    //#pragma omp parallel for num_threads(12) schedule(dynamic, 100)

    // https://stackoverflow.com/a/10625090/10152334
    unsigned seed;
    #pragma omp parallel private(seed)
    {
        seed = 25234 + 17 * omp_get_thread_num();
        for (size_t x = 0; x < N; ++x) {
            #pragma omp for
            for (size_t y = 0; y < (size_t)std::pow(P, x); ++y) {
                //nx[x][y] = (rand() % (Max - Min + 1)) + Min;
                nx[x][y] = (rand_r(&seed) % (Max - Min + 1)) + Min; // rand_r not truly rand but for this case is OK
            }
        }
    }

    // Free memory
    for(size_t i = 0 ; i < N ; i++ )
    {
        delete[] nx[i];
    }
    delete[] nx;    

/*
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
            results.back().second.emplace_back(thread_pool.enqueue(Task(), elem_fn.second, prime_nbr));
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
    */
    return EXIT_SUCCESS;
}