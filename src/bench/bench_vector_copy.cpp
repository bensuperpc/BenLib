/*
** BENSUPERPC PROJECT, 2020
** Texture
** File description:
** bench_vector_copy.cpp
*/
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
    long double operator()(std::function<void(std::vector<std::vector<uint64_t>> &, std::vector<std::vector<uint64_t>> &)> elem_fn, size_t size)
    {
        auto &&Mat1 = my::vector::generate_matrix<uint64_t>(size, size, 0);
        auto &&Mat2 = my::vector::generate_matrix<uint64_t>(size, size, 42);
        auto &&t1 = my::chrono::now();
        (elem_fn)(Mat1, Mat2);
        auto &&t2 = my::chrono::now();
        Mat1.clear();
        Mat1.shrink_to_fit();
        Mat2.clear();
        Mat2.shrink_to_fit();
        return my::chrono::duration(t1, t2).count();
    }
};

// https://bigprimes.org/
// https://stackoverflow.com/questions/41749792/can-i-have-a-stdvector-of-template-function-pointers

int main(int argc, char *argv[], char *envp[])
{
    std::vector<std::pair<std::string, std::vector<std::future<long double>>>> results2 {};

#if __cplusplus >= 201703L
    thread::Pool thread_pool2(1); // Use 1 thread to improve results stability (avoid some spikes on graph)
#else
    ThreadPool thread_pool2(1); // Use 1 thread to improve results stability (avoid some spikes on graph)
#endif

    std::vector<uint64_t> nbrs_nbrs(6000);
    std::iota(nbrs_nbrs.begin(), nbrs_nbrs.end(), 1);
    const std::vector<std::pair<const std::string, void (*)(std::vector<std::vector<uint64_t>> &, std::vector<std::vector<uint64_t>> &)>> pointer_fn_map {
        {"cache_friendly_loop_copy", &my::vector::cache_friendly_copy<uint64_t>}, {"cache_unfriendly_loop_copy", &my::vector::cache_unfriendly_copy<uint64_t>},
        {"std_assignment_copy", &my::vector::assignment_copy<uint64_t>}, {"std_copy", &my::vector::std_copy<uint64_t>},
        {"vector_assign_copy", &my::vector::vector_assign_copy<uint64_t>}};
    results2.reserve(pointer_fn_map.size());

    for (auto &elem_fn : pointer_fn_map) {
        results2.emplace_back(std::pair<std::string, std::vector<std::future<long double>>>());
        results2.back().first = elem_fn.first;
        results2.back().second.reserve(nbrs_nbrs.size());
        for (auto &prime_nbr : nbrs_nbrs) {
            results2.back().second.emplace_back(thread_pool2.enqueue(Processor(), elem_fn.second, prime_nbr));
        }
    }
    std::ios_base::sync_with_stdio(false);
    std::cout << "Pool threading: OK" << std::endl;
    // std::this_thread::sleep_for(std::my::math::milliseconds(20000));
    // Get result
    std::cout << std::setprecision(3) << std::fixed;
    size_t count = 0;
    const size_t xElem = 15;
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
                std::cout << double(count) / double(nbrs_nbrs.size() * pointer_fn_map.size()) * 100.0f << " %" << std::endl;
            }
        }
    }
    std::cout << "Get data: OK" << std::endl;

    std::ifstream in("in.txt");
    std::cin.tie(0);
    auto cinbuf = std::cin.rdbuf(in.rdbuf()); // save and redirect

    std::ofstream out("log_bench_vector_copy.csv");
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