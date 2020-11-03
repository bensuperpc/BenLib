#include <string>
#include <vector>
#include "../../lib/crypto/crypto.hpp"
#include "../../lib/filesystem/filesystem.hpp"
#include "../../lib/time/chrono/chrono.hpp"
#if __cplusplus >= 201703L
#    include "../../lib/thread/Pool.hpp"
#else
#    include "../../lib/thread/ThreadPool.h"
#endif

struct Processor
{
    double operator()(std::function<std::string(const std::string &)> elem_fn, const std::string &elem)
    {

        auto &&t1 = my::chrono::now();
        std::string && str = (elem_fn)(elem);
        auto &&t2 = my::chrono::now();
        return my::chrono::duration(t1, t2).count();
    }
};

// Source https://www.quora.com/How-can-I-get-the-MD5-or-SHA-hash-of-a-file-in-C

int main()
{
    std::vector<std::string> list_files = {};
    list_files.reserve(1000);

    my::filesystem::list_all_files(list_files, ".");

    std::ios_base::sync_with_stdio(false);

    std::vector<std::pair<std::string, std::vector<std::future<double>>>> results {};
#if __cplusplus >= 201703L
    thread::Pool thread_pool(12);
#else
    ThreadPool thread_pool(12);
#endif

    const std::vector<std::pair<const std::string, std::string (*)(const std::string &)>> pointer_map {{"get_md5hash", &my::crypto::get_md5hash},
        {"get_sha1hash", &my::crypto::get_sha1hash}, {"get_sha256hash", &my::crypto::get_sha256hash}, {"get_sha512hash", &my::crypto::get_sha512hash}};

    results.reserve(list_files.size());

    for (auto &elem_fn : pointer_map) {
        results.emplace_back(std::pair<std::string, std::vector<std::future<double>>>());
        results.back().first = elem_fn.first;
        results.back().second.reserve(list_files.size());
        for (auto &file : list_files) {
            results.back().second.emplace_back(thread_pool.enqueue(Processor(), elem_fn.second, file));
        }
    }

    size_t count = 0;
    const size_t xElem = 15;
    std::vector<std::pair<std::string, std::vector<double>>> time {};
    time.reserve(results.size());
    for (auto &y : results) {
        time.emplace_back(std::pair<std::string, std::vector<double>>());
        time.back().second.reserve(y.second.size());
        time.back().first = y.first;
        for (auto &x : y.second) {
            time.back().second.emplace_back(x.get());
            count++;
            if (count % xElem == 0) {
                std::cout << double(count) / double(list_files.size() * pointer_map.size()) * 100.0f << " %" << std::endl;
            }
        }
    }
    std::cout << "Get data: OK" << std::endl;

    std::ifstream in("in.txt");
    std::cin.tie(0);
    auto cinbuf = std::cin.rdbuf(in.rdbuf()); // save and redirect

    std::ofstream out("log_list_all_file_hash.csv");
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

    /*
    for (const auto &elem : list_files) {
        std::cout << elem << std::endl;
        std::cout << "MD5 : " << get_md5hash(elem) << std::endl;
        std::cout << "SHA1 : " << get_sha1hash(elem) << std::endl;
        std::cout << "SHA256 : " << get_sha256hash(elem) << std::endl;
        std::cout << "SHA512 : " << get_sha512hash(elem) << std::endl;
    }*/
}