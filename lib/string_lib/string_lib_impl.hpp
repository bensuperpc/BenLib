/**
 * @file string_lib_impl.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2021-03-03
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
//  Created: 03, March, 2021                                //
//  Modified: 03, March, 2021                               //
//  file: string.hpp                                        //
//  Crypto                                                  //
//  Source:                                                 //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#ifndef STRING_LIB_IMPL_HPP_
#define STRING_LIB_IMPL_HPP_

#include "string_lib.hpp"

/**
 * @brief 
 * 
 * @tparam T 
 * @param n 
 * @return std::string 
 */
template <class T> std::string my::string::findString(T n)
{
#ifdef DNDEBUG
    assert(n > 0); // Test forbiden value
#endif
    const std::string alpha = alphabetMax;
    std::string ans;

    if (n <= alphabetSize) {
        ans = alpha[n - 1];
        return ans;
    }
    while (n > 0) {
        ans += alpha[(--n) % alphabetSize];
        n /= alphabetSize;
    }
    std::reverse(ans.begin(), ans.end());
    return ans;
}

/**
 * @brief 
 * 
 * @tparam T 
 * @param n 
 * @param array 
 */
template <class T> void my::string::findStringInv(T n, char *array)
{
#ifdef DNDEBUG
    assert(n > 0); // Test forbiden value
#endif
    constexpr std::uint32_t stringSizeAlphabet {alphabetSize + 1};
    constexpr std::array<char, stringSizeAlphabet> alpha {alphabetMax};
    if (n < stringSizeAlphabet) {
        array[0] = alpha[n - 1];
        return;
    }
    T i = 0;
    while (n > 0) {
        array[i] = alpha[(--n) % alphabetSize];
        n /= alphabetSize;
        ++i;
    }
}

/**
 * @brief 
 * 
 * @tparam T 
 * @param n 
 * @param array 
 */
template <class T> void my::string::findString(T n, char *array)
{
#ifdef DNDEBUG
    assert(n > 0); // Test forbiden value
#endif
    constexpr std::uint32_t stringSizeAlphabet {alphabetSize + 1};
    constexpr std::array<char, stringSizeAlphabet> alpha {alphabetMax};
    if (n < stringSizeAlphabet) {
        array[0] = alpha[n - 1];
        return;
    }
    T i = 0;
    while (n > 0) {
        array[i] = alpha[(--n) % alphabetSize];
        n /= alphabetSize;
        ++i;
    }
    std::reverse(array, array + strlen(array));
}

#endif
