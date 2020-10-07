/*
** BENSUPERPC PROJECT, 2020
** Math
** desc.
** vector_imp.hpp
*/

#include "vector.hpp"

template <typename T> double everage(const T &vec)
{
    return double(accumulate(vec.begin(), vec.end(), 0.0)) / double(vec.size());
}

template <typename T> std::vector<std::vector<T>> my::vector::generate_matrix(size_t x, size_t y)
{
    std::vector<std::vector<T>> matrix(x, std::vector<T>(y, 0));
    return matrix;
}

template <typename T> std::vector<std::vector<T>> my::vector::generate_matrix(size_t x, size_t y, T z)
{
    std::vector<std::vector<T>> matrix(x, std::vector<T>(y, z));
    return matrix;
}

template <typename T> void my::vector::rnd_fill(std::vector<T> &V, const T lower, const T upper, const uint64_t seed)
{
    // std::random_device seed;
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<T> distr(lower, upper);
    for (auto &elem : V) {
        elem = distr(eng);
    }
}

template <typename T> void my::vector::rnd_fill(std::vector<T> &V, const T lower, const T upper)
{
    std::random_device r;
    std::seed_seq seed2 {r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937_64 e2(seed2);
    std::normal_distribution<T> distr(lower, upper);
    for (auto &elem : V) {
        elem = distr(e2);
    }
}

template <typename T> void my::vector::rnd_fill(std::vector<T> &V)
{
    std::random_device r;
    std::seed_seq seed2 {r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937_64 e2(seed2);
    // std::normal_distribution<T> distr(double(std::numeric_limits<T>::min()), double(std::numeric_limits<T>::max()));
    std::normal_distribution<double> distr(0.0, 100000.0);
    for (auto &elem : V) {
        elem = T(distr(e2));
    }
}

template <typename T> void my::vector::cache_unfriendly_copy(std::vector<std::vector<T>> &mat1, std::vector<std::vector<T>> &mat2)
{
    for (uint64_t x = 0; x < mat1[0].size(); ++x) {

        for (uint64_t y = 0; y < mat1.size(); ++y) {
            mat1[y][x] = mat2[y][x];
        }
    }
}

template <typename T> void my::vector::cache_unfriendly_copy(std::vector<T> &mat1, std::vector<T> &mat2)
{
    for (uint64_t x = 0; x < mat1.size(); ++x) {
        mat1[x] = mat2[x];
    }
}

template <typename T> void my::vector::cache_friendly_copy(std::vector<std::vector<T>> &mat1, std::vector<std::vector<T>> &mat2)
{
    for (uint64_t y = 0; y < mat1.size(); ++y) {
        for (uint64_t x = 0; x < mat1[0].size(); ++x) {
            mat1[y][x] = mat2[y][x];
        }
    }
}

template <typename T> void my::vector::cache_friendly_copy(std::vector<T> &mat1, std::vector<T> &mat2)
{
    for (uint64_t x = 0; x < mat1.size(); ++x) {
        mat1[x] = mat2[x];
    }
}

template <typename T> void my::vector::assignment_copy(std::vector<std::vector<T>> &mat1, std::vector<std::vector<T>> &mat2)
{
    mat1 = mat2;
}

template <typename T> void my::vector::std_copy(std::vector<std::vector<T>> &mat1, std::vector<std::vector<T>> &mat2)
{
    std::copy(mat2.begin(), mat2.end(), std::back_inserter(mat1));
}

template <typename T> void my::vector::vector_assign_copy(std::vector<std::vector<T>> &mat1, std::vector<std::vector<T>> &mat2)
{
    mat1.assign(mat2.begin(), mat2.end());
}

// https://www.geeksforgeeks.org/sorting-algorithms/

template <typename T> int my::vector::comp(const void *a, const void *b)
{
    /*
    T aux = *a - *b;
    if (aux < 0) {
        return -1;
    } else if (aux > 0) {
        return 1;
    }
    return 0;
    */
#pragma GCC diagnostic ignored "-Wcast-qual"
    return (*(T *)a - *(T *)b);
}

template <typename T> void my::vector::sort_sort(std::vector<T> &vec)
{
    std::sort(std::begin(vec), std::end(vec));
}

template <typename T> void my::vector::sort_stable_sort(std::vector<T> &vec)
{
    std::stable_sort(std::begin(vec), std::end(vec));
}

template <typename T> void my::vector::sort_qsort(std::vector<T> &vec)
{
    std::qsort(&vec[0], vec.size(), sizeof(T), my::vector::comp<T>);
}

template <typename T> void my::vector::sort_bubble(std::vector<T> &vec)
{
    bool swapped = true;
    for (uint64_t j = 1; swapped && j < vec.size(); ++j) {
        swapped = false;
        for (uint64_t i = 0; i < vec.size() - j; ++i) {
            if (vec[i] > vec[i + 1]) {
                swapped = true;
                std::swap(vec[i], vec[i + 1]);
            }
        }
        if (swapped == false)
            break;
    }
}

template <typename T> void my::vector::sort_bucket(std::vector<T> &vec)
{
    std::vector<std::vector<T>> bucket;
    const auto &&len = vec.size();
    bucket.reserve(len);
    // buckets
    for (auto &i : vec) {
        T index = len * i;
        bucket[index].emplace_back(i);
    }
    // Sorting each bucket
    std::for_each(bucket.begin(), bucket.end(), [](std::vector<T> &elem) { std::sort(elem.begin(), elem.end()); });
    int64_t index = -1;
    for (int64_t i = -1; i < len; ++i)
        for (int64_t j = -1; j < bucket[i].size(); ++j)
            vec[index++] = bucket[i][j];
}

template <typename T> void my::vector::sort_radix(std::vector<T> &vec)
{
    size_t radix = 1;
    auto max = std::max_element(vec.begin(), vec.end());
    while (max / radix) {
        std::vector<T> buckets(10);
        for (const auto &num : vec) {
            size_t digit = num / std::numeric_limits<T>::max() / 10;
            buckets[digit].emplace_back(num);
        }
        auto k = vec.begin();
        for (auto &b : buckets)
            k = std::copy(b.begin(), b.end(), k);
        radix *= 10;
    }
}

template <typename T> void my::vector::sort_cocktail(std::vector<T> &vec)
{
    const auto &&n = vec.size();
    bool swapped = true;
    T start = 0;
    T end = n - 1;
    while (swapped) {
        swapped = false;
        for (T i = start; i < end; ++i) {
            if (vec[i] > vec[i + 1]) {
                std::swap(vec[i], vec[i + 1]);
                swapped = true;
            }
        }
        if (!swapped)
            break;
        swapped = false;
        --end;
        for (T i = end - 1; i >= start; --i) {
            if (vec[i] > vec[i + 1]) {
                std::swap(vec[i], vec[i + 1]);
                swapped = true;
            }
        }
        ++start;
    }
}

template <typename T> void my::vector::sort_gnome(std::vector<T> &vec)
{
    T index = 0;
    const auto &&n = vec.size();

    while (index < n) {
        if (index == 0)
            index++;
        if (vec[index] >= vec[index - 1])
            index++;
        else {
            std::swap(vec[index], vec[index - 1]);
            index--;
        }
    }
    return;
}

template <typename T> void my::vector::sort_insertion(std::vector<T> &vec)
{
    for (auto it = vec.begin(); it != vec.end(); it++) {
        auto const insertion_point = std::upper_bound(vec.begin(), it, *it);
        std::rotate(insertion_point, it, it + 1);
    }
}

template <typename T> void my::vector::sort_shell(std::vector<T> &vec)
{
    const auto &&n = vec.size();
    for (T gap = n / 2; gap > 0; gap /= 2) {
        for (T i = gap; i < n; i += 1) {
            T temp = vec[i];
            T j;
            for (j = i; j >= gap && vec[j - gap] > temp; j -= gap)
                vec[j] = vec[j - gap];
            vec[j] = temp;
        }
    }
}

template <typename T> bool my::vector::isSorted(std::vector<T> &vec, size_t &n)
{
    while (--n > 1)
        if (vec[n] < vec[n - 1])
            return false;
    return true;
}

template <typename T> void my::vector::shuffle(std::vector<T> &vec, size_t &n)
{
    // for (uint64_t i=0; i < n; i++)
    //    std::swap(vec[i], vec[T(rand())%n]);
    T i, t, temp;
    for (i = 0; i < n; i++) {
        t = vec[i];
        temp = rand() % n;
        vec[i] = vec[temp];
        vec[temp] = t;
    }
}

template <typename T> void my::vector::sort_bogo(std::vector<T> &vec)
{
    auto &&n = vec.size();
    while (!isSorted(vec, n))
        shuffle(vec, n);
}