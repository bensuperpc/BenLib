/**
 * @file date.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.1.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
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
/**
 * @namespace my::date
 * @brief the date namespace, Get week day with date
 */
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
