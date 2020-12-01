/*
** BENSUPERPC PROJECT, 2020
** Math
** Source: https://stackoverflow.com/questions/178265/what-is-the-most-hard-to-understand-piece-of-c-code-you-know https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
** vector.hpp
*/

#ifndef VECTOR_HPP_
#    define VECTOR_HPP_

#    include <algorithm>
#    include <iomanip>
#    include <iostream>
#    include <random>
#    include <vector>

namespace my
{
namespace vector
{
std::vector<std::vector<int>> generate_matrix(size_t, size_t);
std::vector<std::vector<int>> generate_matrix(size_t, size_t, int);
void print_2d(std::vector<std::vector<int>> &);
void fill_row(std::vector<int> &);
void fill_rowull(std::vector<uint64_t> &);

void fill_matrix_1(std::vector<std::vector<int>> &);
void fill_matrix_2(std::vector<std::vector<int>> &);
void fill_matrix_2(std::vector<std::vector<int>> &, int, int);

template <typename T> double everage(const T &vec);
template <typename T> void rnd_fill(std::vector<T> &, const T, const T, const uint64_t);
template <typename T> void rnd_fill(std::vector<T> &, const T, const T);
template <typename T> void rnd_fill(std::vector<T> &);

template <typename T> std::vector<std::vector<T>> generate_matrix(size_t x, size_t y, T z);
template <typename T> std::vector<std::vector<T>> generate_matrix(size_t x, size_t y);

// Bench part
template <typename T> void cache_unfriendly_copy(std::vector<std::vector<T>> &, std::vector<std::vector<T>> &);
template <typename T> void cache_unfriendly_copy(std::vector<T> &, std::vector<T> &);
template <typename T> void cache_friendly_copy(std::vector<T> &, std::vector<T> &);
template <typename T> void cache_friendly_copy(std::vector<std::vector<T>> &, std::vector<std::vector<T>> &);

template <typename T> void assignment_copy(std::vector<std::vector<T>> &, std::vector<std::vector<T>> &);
template <typename T> void std_copy(std::vector<std::vector<T>> &, std::vector<std::vector<T>> &);
template <typename T> void vector_assign_copy(std::vector<std::vector<T>> &, std::vector<std::vector<T>> &);

template <typename T> int comp(const void *, const void *);
template <typename T> void sort_qsort(std::vector<T> &);
template <typename T> void sort_sort(std::vector<T> &);
template <typename T> void sort_stable_sort(std::vector<T> &);
template <typename T> void sort_bubble(std::vector<T> &);
template <typename T> void sort_bucket(std::vector<T> &);

template <typename T> void sort_radix(std::vector<T> &vec);

template <typename T> void sort_cocktail(std::vector<T> &);
template <typename T> void sort_gnome(std::vector<T> &vec);
template <typename T> void sort_insertion(std::vector<T> &);
template <typename T> void sort_shell(std::vector<T> &);

template <typename T> void sort_bogo(std::vector<T> &);
template <typename T> void shuffle(std::vector<T> &, size_t &);
template <typename T> bool isSorted(std::vector<T> &, size_t &);

#    include "vector_imp.hpp"

} // namespace vector
} // namespace my
#endif
// https://stackoverflow.com/questions/22312959/how-to-fill-a-vector-with-a-range
