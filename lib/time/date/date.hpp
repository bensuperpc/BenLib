/*
** BENSUPERPC PROJECT, 2020
** RPG
** File description:
** date.hpp
*/

#ifndef _DATE_HPP_
#define _DATE_HPP_

#include <ctime>
#include <string>
/* HEADER FOR DEBIAN */
#include <iomanip>
#include <sstream>
namespace my
{
namespace date
{
const std::string weekday[7] = {"Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"};
std::string zellersAlgorithm(int &, int &, int &);
std::string get_date();
} // namespace date
} // namespace my
// THANKS https://stackoverflow.com/a/16358111/10152334
// https://stackoverflow.com/questions/15127615/determining-day-of-the-week-using-zellers-congruence

#endif
