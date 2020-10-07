/*
** BENSUPERPC PROJECT, 2020
** CPU
** File description:
** vector.cpp
*/

#include "vector.hpp"

void my::vector::fill_row(std::vector<int> &row)
{
    std::generate(row.begin(), row.end(), []() { return rand() % 100; });
}

void my::vector::fill_rowull(std::vector<uint64_t> &row)
{
    std::generate(row.begin(), row.end(), []() { return rand() % (18446744073709551615 - 0 + 1) + 0; });
}

void my::vector::fill_matrix_1(std::vector<std::vector<int>> &mat)
{
    std::for_each(mat.begin(), mat.end(), fill_row);
}

void my::vector::fill_matrix_2(std::vector<std::vector<int>> &Matrix)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
    // std::vector<std::vector<int>> Matrix(5, std::vector<int>(7, 0));

    for (auto it1 = Matrix.begin(); it1 != Matrix.end(); it1++) {
        for (auto it2 = it1->begin(); it2 != it1->end(); it2++) {
            *it2 = dis(gen);
        }
    }
}

void my::vector::fill_matrix_2(std::vector<std::vector<int>> &Matrix, int min, int max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min, max);
    // std::vector<std::vector<int>> Matrix(5, std::vector<int>(7, 0));

    for (auto it1 = Matrix.begin(); it1 != Matrix.end(); it1++) {
        for (auto it2 = it1->begin(); it2 != it1->end(); it2++) {
            *it2 = dis(gen);
        }
    }
}

void my::vector::print_2d(std::vector<std::vector<int>> &mat)
{
    for (const auto &y : mat) {
        for (const auto &x : y) {
            std::cout << std::setw(4) << x;
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}

std::vector<std::vector<int>> my::vector::generate_matrix(size_t x, size_t y)
{
    std::vector<std::vector<int>> matrix(x, std::vector<int>(y, 0));
    return matrix;
}

std::vector<std::vector<int>> my::vector::generate_matrix(size_t x, size_t y, int z)
{
    std::vector<std::vector<int>> matrix(x, std::vector<int>(y, z));
    return matrix;
}