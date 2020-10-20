#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <thread>
#include <vector>
#include "../../lib/time/chrono/chrono.hpp"
#include "../../lib/vector/vector.hpp"
#include "../../lib/vector/vector_avx.hpp"

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
    long double operator()(std::function<int(const int32_t *, size_t)> elem_fn, const int *array, const size_t &array_size, const int &nrbs)
    {
        int result = 0;

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

int main()
{
    std::vector<std::pair<std::string, std::vector<std::future<long double>>>> results2 {};
#if __cplusplus >= 201703L
    thread::Pool thread_pool(1);
#else
    ThreadPool thread_pool(1);
#endif
    const size_t n = 2; // 16384
    const int nofTestCases = 100000;
    int n1[n * 2] __attribute__((aligned(32)));
    int n2[n * 4] __attribute__((aligned(32)));
    int n3[n * 8] __attribute__((aligned(32)));
    int n4[n * 16] __attribute__((aligned(32)));
    int n5[n * 32] __attribute__((aligned(32)));
    int n6[n * 64] __attribute__((aligned(32)));
    int n7[n * 128] __attribute__((aligned(32)));
    int n8[n * 256] __attribute__((aligned(32)));
    int n9[n * 512] __attribute__((aligned(32)));
    int n10[n * 1024] __attribute__((aligned(32)));
    int n11[n * 2048] __attribute__((aligned(32)));
    int n12[n * 4096] __attribute__((aligned(32)));
    int n13[n * 8192] __attribute__((aligned(32)));
    int n14[n * 16384] __attribute__((aligned(32)));
    std::vector<std::pair<const size_t, int *>> n_ptr = {{n * 2, n1}, {n * 4, n2}, {n * 8, n3}, {n * 16, n4}, {n * 32, n5}, {n * 64, n6}, {n * 128, n7},
        {n * 256, n8}, {n * 512, n9}, {n * 1024, n10}, {n * 2048, n11}, {n * 4096, n12}, {n * 8192, n13}, {n * 16384, n14}};

    // Fill array with random values
    for (auto &n_ptr_s : n_ptr) {
        for (auto k = 0; k < n_ptr_s.first; k++) {
            n_ptr_s.second[k] = rand() % 10000;
        }
    }

    // Declare point to function
    const std::vector<std::pair<const std::string, std::function<int(const int32_t *, size_t)>>> pointer_fn_map {
        {"find_max_normal", &my::vector_avx::find_max_normal}, {"find_max_sse", &my::vector_avx::find_max_sse},
        {"find_max_avx", &my::vector_avx::find_max_avx}};

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
}