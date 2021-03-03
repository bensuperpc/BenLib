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
//  file: string.cpp                                        //
//  Crypto                                                  //
//  Source:                                                 //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

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