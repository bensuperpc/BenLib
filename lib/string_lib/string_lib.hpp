/**
 * @file string_lib.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */
//////////////////////////////////////////////////////////////
//   ____                                                   //
//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
//                             |_|             |_|          //
//////////////////////////////////////////////////////////////
//                                                          //
//  BenLib, 2021                                            //
//  Created: 02, March, 2021                                //
//  Modified: 02, March, 2021                               //
//  file: string.hpp                                        //
//  Crypto                                                  //
//  Source:                                                 //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#ifndef STRING_LIB_HPP_
#define STRING_LIB_HPP_

#include <algorithm> // for std::find
#include <boost/crc.hpp>
#include <cmath> // pow
#include <cstring>
#include <iomanip> // setw
#include <string>
#include <string_view> // string_view
#include <tuple>
#include <utility> // std::make_pair
#include <vector>

/**
 * @brief 
 * 
 */
#define alphabetMax "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

/**
 * @brief 
 * 
 */
#define alphabetMin "abcdefghijklmnopqrstuvwxyz"

namespace my
{
namespace string
{
/**
 * @brief 
 * 
 */
constexpr std::uint32_t alphabetSize {26};

/**
 * @brief 
 * 
 * @tparam T 
 * @param n 
 * @return std::string 
 */
template <class T> std::string findString(T n);

/**
 * @brief 
 * 
 * @tparam T 
 * @param n 
 * @param array 
 */
template <class T> void findStringInv(T n, char *array);

/**
 * @brief 
 * 
 * @tparam T 
 * @param n 
 * @param array 
 */
template <class T> void findString(T n, char *array);

/**
 * @brief 
 * 
 * @param N 
 * @return std::vector<std::string> 
 */
std::vector<std::string> generateSequenceBySize(const std::size_t N);
} // namespace string
} // namespace my
#endif
