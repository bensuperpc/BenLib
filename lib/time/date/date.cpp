/*
** BENSUPERPC PROJECT, 2020
** RPG
** File description:
** date.cpp
*/

#include "date.hpp"

std::string my::date::get_date()
{
    auto &&t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
    return oss.str();
}

std::string my::date::zellersAlgorithm(int &day, int &month, int &year)
{
    int h, q, m, k, j;
    if (month == 1) {
        month = 13;
        year--;
    }
    if (month == 2) {
        month = 14;
        year--;
    }
    q = day;
    m = month;
    k = year % 100;
    j = year / 100;
    h = q + 13 * (m + 1) / 5 + k + k / 4 + j / 4 + 5 * j;
    h = h % 7;
    return weekday[h];
}