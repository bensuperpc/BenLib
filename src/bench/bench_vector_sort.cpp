#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <thread>
#include <vector>
#include "../../math/prime.hpp"
#include "../../time/chrono/chrono.hpp"
#include "../../vector/vector.hpp"

#if __cplusplus >= 201703L
#    include "../../thread/Pool.hpp"
#else
#    include "../../thread/ThreadPool.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>

struct Processor
{
    long double operator()(std::function<void(std::vector<uint64_t> &)> elem_fn, size_t size)
    {
        // Generate vector with X elements
        auto &&Mat = std::vector<uint64_t>(size, 0);
        // Fill vector with RND nimbers
        my::vector::rnd_fill<uint64_t>(Mat);

        auto &&t1 = my::chrono::now();
        (elem_fn)(Mat);
        auto &&t2 = my::chrono::now();
        Mat.clear();
        Mat.shrink_to_fit();
        return my::chrono::duration(t1, t2).count();
    }
};

int main()
{
    std::vector<std::pair<std::string, std::vector<std::future<long double>>>> results2 {};
#if __cplusplus >= 201703L
    thread::Pool thread_pool2(1);
#else
    ThreadPool thread_pool2(1);
#endif

    std::vector<uint64_t> nbrs_nbrs(3000);
    std::iota(nbrs_nbrs.begin(), nbrs_nbrs.end(), 1);
    // CRASH {"sort_cocktail", &my::vector::sort_cocktail<uint64_t>}
    //{"sort_bucket", &my::vector::sort_bucket<uint64_t>}
    const std::vector<std::pair<const std::string, std::function<void(std::vector<uint64_t> &)>>> pointer_fn_map {
        {"sort_qsort", &my::vector::sort_qsort<uint64_t>}, {"sort_sort", &my::vector::sort_sort<uint64_t>},
        {"sort_stable_sort", &my::vector::sort_stable_sort<uint64_t>}, {"sort_bubble", &my::vector::sort_bubble<uint64_t>},
        {"sort_gnome", &my::vector::sort_gnome<uint64_t>}, {"sort_insertion", &my::vector::sort_insertion<uint64_t>},
        {"sort_shell", &my::vector::sort_shell<uint64_t>}, {"sort_bogo", &my::vector::sort_bogo<uint64_t>}};

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

    std::ofstream out("log_bench_vector_sort.csv");
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