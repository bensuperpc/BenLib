/**
 * @file string_lib.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

#include "string_lib.hpp"

std::vector<std::string> my::string::generateSequenceBySize(const std::size_t N)
{
    if (N == 1)
        return std::vector<std::string>();

    constexpr std::size_t base = alphabetSize;
    std::vector<std::string> seqs;
    //    seqs.reserve(pow(base, N)); // Ne jamais utiliser pow pour une puissance entière : ça utilise une décomposition numérique de e^ln(). C'est ultra lourd
    for (std::size_t i = 0; i < pow(base, N); i++) {
        std::size_t value = i;
        std::string tmp(N, 'A');
        for (std::size_t j = 0; j < N; j++) {
            tmp[N - 1 - j] = 'A' + value % base;
            value = value / base;
        }
        seqs.emplace_back(tmp);
    }
    return seqs;
}