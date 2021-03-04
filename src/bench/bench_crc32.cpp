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
#include <random>
#include <thread>
#include <vector>
#include "crypto/crypto_CRC32.hpp"
#include "time/chrono/chrono.hpp"

#if __cplusplus >= 201703L
#    include "thread/Pool.hpp"
#else
#    include "thread/ThreadPool.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <string.h> // <cstring> in C++

#if defined(_OPENMP)
#    include <omp.h>
#endif

#define ToFile //Redirect cout/cin to file

struct Task
{
    long double operator()(std::function<uint32_t(const void *data, size_t length, uint32_t previousCrc32)> elem_fn, const unsigned char * data, const size_t length)
    {
        auto &&t1 = my::chrono::now();
        auto &&a __attribute__((unused)) = (elem_fn)(data, length, 0);
        auto &&t2 = my::chrono::now();
        return my::chrono::duration(t1, t2).count();
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
    const size_t P = 2;  // Power
    const unsigned char Min = 32;  // ACSII = Space
    const unsigned char Max = 126; // ACSII = ~

    unsigned char **nx = new unsigned char *[N];
#if defined(_OPENMP)
#    pragma omp parallel for num_threads(12)
    for (size_t x = 0; x < N; ++x) {
        nx[x] = new unsigned char[(std::size_t)std::pow(P, x )];
// printf("Element %ld traité par le thread %d \n",x,omp_get_thread_num());
#ifdef DNDEBUG
#    pragma omp critical
        std::cout << (size_t)std::pow(P, x) << " Elements" << std::endl;
#    pragma omp critical
        std::cout << ((std::pow(P, x ) * 8) / (8 * 1024)) << " Ko"
                  << "\n"
                  << std::endl;
#endif
    }
#else
    for (size_t x = 0; x < N; ++x) {
        nx[x] = new unsigned char[(std::size_t)std::pow(P, x )];
        // printf("Element %ld traité par le thread %d \n",x,omp_get_thread_num());
#ifdef DNDEBUG
        std::cout << (size_t)std::pow(P, x) << " Elements" << std::endl;
        std::cout << ((std::pow(P, x ) * 8) / (8 * 1024)) << " Ko"
                  << "\n"
                  << std::endl;
#endif
    }
#endif
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
#if defined(_OPENMP)
#    pragma omp parallel private(seed)
    {

        seed = 25234 + 17 * omp_get_thread_num();
        for (size_t x = 0; x < N; ++x) {
#    pragma omp for
            for (size_t y = 0; y < (size_t)std::pow(P, x); ++y) {
                // nx[x][y] = (rand() % (Max - Min + 1)) + Min;
                nx[x][y] = (rand_r(&seed) % (Max - Min + 1)) + Min; // rand_r not truly rand but for this case is OK
            }
        }
    }
#else
    seed = 25234 + 17 * 12;
    for (size_t x = 0; x < N; ++x) {
        for (size_t y = 0; y < (size_t)std::pow(P, x); ++y) {
            // nx[x][y] = (rand() % (Max - Min + 1)) + Min;
            nx[x][y] = (rand_r(&seed) % (Max - Min + 1)) + Min; // rand_r not truly rand but for this case is OK
        }
    }
#endif

    const std::vector<std::pair<const std::string, uint32_t (*)(const void *data, size_t length, uint32_t previousCrc32)>> pointer_map {{"CRC32_StackOverflow", &my::crypto::CRC32_StackOverflow},
        {"CRC32_1byte_tableless", &my::crypto::CRC32_1byte_tableless}, {"CRC32_1byte_tableless2", &my::crypto::CRC32_1byte_tableless2}, {"CRC32_bitwise", &my::crypto::CRC32_bitwise},
        {"CRC32_halfbyte", &my::crypto::CRC32_halfbyte}, {"CRC32_1byte", &my::crypto::CRC32_1byte}, {"CRC32_4bytes", &my::crypto::CRC32_4bytes}, {"CRC32_8bytes", &my::crypto::CRC32_8bytes},
        {"CRC32_4x8bytes", &my::crypto::CRC32_4x8bytes}, {"CRC32_16bytes", &my::crypto::CRC32_16bytes}, {"CRC32_Boost", &my::crypto::CRC32_Boost}};   

    // Generate poolthreading
    results.reserve(pointer_map.size());
    for (auto &elem_fn : pointer_map) {
        results.emplace_back(std::pair<std::string, std::vector<std::future<long double>>>());
        results.back().first = elem_fn.first;
        results.back().second.reserve(N);
        for (size_t i = 0; i < N; i++) {
            results.back().second.emplace_back(thread_pool.enqueue(Task(), elem_fn.second, nx[i], (size_t)std::pow(P, i)));
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
    
#ifdef ToFile
    std::ifstream in("in.txt");
    std::cin.tie(0);
    auto cinbuf = std::cin.rdbuf(in.rdbuf()); // save and redirect

    std::ofstream out("log_bench_crc.csv");
    std::ios_base::sync_with_stdio(false);

    auto coutbuf = std::cout.rdbuf(out.rdbuf()); // save and redirect
#endif

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

#ifdef ToFile
    std::cin.rdbuf(cinbuf);
    std::cout.rdbuf(coutbuf);
#endif

    // Free memory
    for (size_t i = 0; i < N; i++) {
        delete[] nx[i];
    }
    delete[] nx;
    return EXIT_SUCCESS;
}