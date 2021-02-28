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
//  Created: 26, February, 2021                             //
//  Modified: 26, February, 2021                            //
//  file: crypto.cpp                                        //
//  Crypto                                                  //
//  Source: http://stackoverflow.com/questions/8710719/generating-an-alphabetic-sequence-in-java                                                //
//          https://stackoverflow.com/a/19299611/10152334                                                //
//          https://gms.tf/stdfind-and-memchr-optimizations.html                                                //
//          https://medium.com/applied/applied-c-align-array-elements-32af40a768ee                                                //
//          https://create.stephan-brumme.com/crc32/                                                //
//          https://rosettacode.org/wiki/Generate_lower_case_ASCII_alphabet                                                //
//          https://web.archive.org/web/20090204140550/http://www.maxbot.com/gta/3wordcheatsdumpsorted.txt                                                //
//          https://www.codeproject.com/Articles/663443/Cplusplus-is-Fun-Optimal-Alphabetical-Order                                                //
//          https://cppsecrets.com/users/960210997110103971089710711510497116484964103109971051084699111109/Given-integer-n-find-the-nth-string-in-this-sequence-A-B-C-Z-AA-AB-AC-ZZ-AAA-AAB-AAZ-ABA-.php
//          https://www.careercup.com/question?id=14276663                                                //
//          https://stackoverflow.com/a/55074804/10152334                                                //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include <algorithm> // for std::find
#include <boost/crc.hpp>
#include <cmath> // pow
#include <cstring>
#include <iostream> // cout
#include <mutex>    // Mutex
#include <string>
#include <string_view> // string_view
#include <vector>
#include "thread/Pool.hpp" // Threadpool

#if defined(__GNUC__)
#    define CACHE_ALIGNED __attribute__((aligned(64))) // clang and GCC
#elif defined(_MSC_VER)
#    define CACHE_ALIGNED __declspec(align(64)) // MSVC
#endif

constexpr std::uint32_t alphabetSize {26};

const std::array<unsigned int, 87> cheat_list {0xDE4B237D, 0xB22A28D1, 0x5A783FAE, 0xEECCEA2B, 0x42AF1E28, 0x555FC201, 0x2A845345, 0xE1EF01EA, 0x771B83FC,
    0x5BF12848, 0x44453A17, 0xFCFF1D08, 0xB69E8532, 0x8B828076, 0xDD6ED9E9, 0xA290FD8C, 0x3484B5A7, 0x43DB914E, 0xDBC0DD65, 0xD08A30FE, 0x37BF1B4E, 0xB5D40866,
    0xE63B0D99, 0x675B8945, 0x4987D5EE, 0x2E8F84E8, 0x1A9AA3D6, 0xE842F3BC, 0x0D5C6A4E, 0x74D4FCB1, 0xB01D13B8, 0x66516EBC, 0x4B137E45, 0x78520E33, 0x3A577325,
    0xD4966D59, 0x5FD1B49D, 0xA7613F99, 0x1792D871, 0xCBC579DF, 0x4FEDCCFF, 0x44B34866, 0x2EF877DB, 0x2781E797, 0x2BC1A045, 0xB2AFE368, 0xFA8DD45B, 0x8DED75BD,
    0x1A5526BC, 0xA48A770B, 0xB07D3B32, 0x80C1E54B, 0x5DAD0087, 0x7F80B950, 0x6C0FA650, 0xF46F2FA4, 0x70164385, 0x885D0B50, 0x151BDCB3, 0xADFA640A, 0xE57F96CE,
    0x040CF761, 0xE1B33EB9, 0xFEDA77F7, 0x8CA870DD, 0x9A629401, 0xF53EF5A5, 0xF2AA0C1D, 0xF36345A8, 0x8990D5E1, 0xB7013B1B, 0xCAEC94EE, 0x31F0C3CC, 0xB3B3E72A,
    0xC25CDBFF, 0xD5CF4EFF, 0x680416B1, 0xCF5FDA18, 0xF01286E9, 0xA841CC0A, 0x31EA09CF, 0xE958788A, 0x02C83A7C, 0xE49C3ED4, 0x171BA8CC, 0x86988DAE, 0x2BDD2FA1};

std::mutex couter;

unsigned int GetCrc32(const std::string_view my_string);

unsigned int GetCrc32(const std::string_view my_string)
{
    boost::crc_32_type result;
    // ça c'est plus rapide que l'autre normalement pour le length. Ça donne le nombre d'élément, donc si tu as plusieurs éléments qui sont à '\0' forcément…
    result.process_bytes(my_string.data(), my_string.length());
    return result.checksum();
}

/*unsigned int GetCrc32(const char *const my_string)
{
    boost::crc_32_type result;
    result.process_bytes(my_string, strlen(my_string));
    return result.checksum();
}*/

std::vector<std::string> generateSequenceBySize(const std::size_t N);

std::vector<std::string> generateSequenceBySize(const std::size_t N)
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

template <class T> std::string findString(T n)
{
    const std::string alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
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

/*template <class T> void findString(T n, char *array)
{
    const char alpha[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    if (n <= alphabetSize) {
        array[0] = alpha[n - 1];
        return;
    }
    std::size_t i = 0;
    while (n > 0) {
        array[i] = alpha[(--n) % alphabetSize];
        n /= alphabetSize;
        ++i;
    }
    std::reverse(array, array + strlen(array));
}*/

template <class T> std::string findStringInv(T n)
{
    const std::string alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::string ans;
    if (n <= alphabetSize) {
        ans = alpha[n - 1];
        return ans;
    }
    while (n > 0) {
        ans += alpha[(--n) % alphabetSize];
        n /= alphabetSize;
    }
    return ans;
}

/**
 * \brief Que fait cette fonction ? oskour aled
 * \tparam T
 * \param n
 * \param array
 */
/*
template <class T> void findStringInv(T n, std::string &array)
{
    constexpr std::uint32_t stringSizeAlphabet {alphabetSize + 1};
    constexpr std::array<char, stringSizeAlphabet> alpha {"ABCDEFGHIJKLMNOPQRSTUVWXYZ"};
    if (n < stringSizeAlphabet) {
        array[0] = alpha[n - 1];
        return;
    }
    std::size_t i = 0;
    while (n > 0) {
        array[i] = alpha[(--n) % alphabetSize];
        n /= alphabetSize;
        ++i;
    }
}

*/

template <class T> void findStringInv(T n, std::string &array)
{
    constexpr std::uint32_t stringSizeAlphabet {alphabetSize + 1};
    constexpr std::array<char, stringSizeAlphabet> alpha {"ABCDEFGHIJKLMNOPQRSTUVWXYZ"};
    if (n < stringSizeAlphabet) {
        array[0] = alpha[n - 1];
        return;
    }
    std::size_t i = 0;
    while (n > 0) {
        array[i] = alpha[(--n) % alphabetSize];
        n /= alphabetSize;
        ++i;
    }
}

template <class T> void findStringInv(T n, char *array)
{
    constexpr std::uint32_t stringSizeAlphabet {alphabetSize + 1};
    constexpr std::array<char, stringSizeAlphabet> alpha {"ABCDEFGHIJKLMNOPQRSTUVWXYZ"};
    if (n < stringSizeAlphabet) {
        array[0] = alpha[n - 1];
        return;
    }
    std::size_t i = 0;
    while (n > 0) {
        array[i] = alpha[(--n) % alphabetSize];
        n /= alphabetSize;
        ++i;
    }
}

struct Processor
{
    size_t operator()(size_t x, size_t y)
    {
        char tmp[29];
        for (size_t i = x; i < y + x; i++) {
            findStringInv<size_t>(i, tmp);
            auto crc = ~(GetCrc32(tmp));
            if (std::find(std::begin(cheat_list), std::end(cheat_list), crc) != std::end(cheat_list)) {         
                std::reverse(tmp, tmp + strlen(tmp));
//#    ifdef DNDEBUG
                couter.lock();
                std::cout << tmp << ":0x" << std::hex << crc << std::endl;
                couter.unlock();
//#endif
            }
        }
        return 0;
    }
};

int main()
{
    std::ios_base::sync_with_stdio(false);
    std::vector<std::future<std::size_t>> results {};
    auto threads = std::thread::hardware_concurrency();
    thread::Pool thread_pool(threads);

    const size_t nbrcal = 308915776; // Nombre calcule à faire
    const size_t threadmult = 12; // Nombre de thread créé par thread CPU
    const size_t nbrthread = std::thread::hardware_concurrency() * threadmult; // Nombre de thread créé au total sur le threadpool

    const size_t nbrcalperthread = nbrcal/nbrthread; // Nombre de calcule par thread (1K mini à 1M max recommandé)
    
    results.reserve(nbrthread); //Réservation

    for (std::size_t i = 1; i < nbrthread; i++) {
        results.emplace_back(thread_pool.enqueue(Processor(), i * nbrcalperthread, nbrcalperthread));
    }

    for (auto &&result : results) {
        auto t = result.get();
        if (t == 1){
            std::cout << t << std::endl;
        }
    }
    /*
    std::string tmp(30, '\0');                    // déjà entièrement rempli de '\0'
    for (std::size_t i = 1; i < 308915776; i++) { // Quel est ce nombre énorme ? D'où il sort ?
        findStringInv<std::size_t>(i, tmp);
        const auto crc = ~(GetCrc32(tmp));
        if (std::find(cheat_list.cbegin(), cheat_list.cend(), crc) != cheat_list.cend()) {
            std::reverse(tmp.begin(), tmp.end());
            std::cout << tmp << ":0x" << std::hex << crc << '\n';  // std::endl flush le buffer, donc syscall à chaque tour de boucle : c'est plus lent.
        }
    }*/
    /*
        for (size_t i = 1; i < 308915776; i++) { // 208827064576
            // std::string tmp = findStringInv<size_t>(i);
            findStringInv<std::size_t>(i, tmp);
            auto crc = ~(GetCrc32(tmp));
            if (std::find(std::begin(cheat_list), std::end(cheat_list), crc) != std::end(cheat_list)) {
                std::reverse(tmp.begin(), tmp.end());
                std::cout << tmp << ":0x" << std::hex << crc << std::endl;
            }
        }*/
    return EXIT_SUCCESS;
}

// En C++ JAMAIS NULL, toujours nullptr. Comme malloc et free : ça n'existe plus en C++, faut pas manger :c
//    char *tmp = nullptr;
//    tmp = (char *)malloc((std::size_t)(29 + 1) * sizeof(char));
//    assert(tmp != nullptr);                       // assert valable en debug uniquement.
/*
  std::string tmp(30, '\0');                    // déjà entièrement rempli de '\0'
  for (std::size_t i = 1; i < 308915776; i++) { // Quel est ce nombre énorme ? D'où il sort ?
      // tmp[(std::size_t)(i / 26 + 1)] = '\0';
      //        tmp[29UL] = '\0';
      findStringInv<std::size_t>(i, tmp);
      const auto crc = ~(GetCrc32(tmp));
      if (std::find(cheat_list.cbegin(), cheat_list.cend(), crc) != cheat_list.cend()) {
          std::reverse(tmp.begin(), tmp.end());
          std::cout << tmp << ":0x" << std::hex << crc << '\n';
      }*/
/*if (std::find(std::begin(cheat_list), std::end(cheat_list), crc) != std::end(cheat_list)) {
    std::reverse(tmp, tmp + strlen(tmp));
    std::cout << tmp << ":0x" << std::hex << crc << '\n'; // std::endl flush le buffer, donc syscall à chaque tour de boucle : c'est plus lent.
}*/
/*}*/
//    free(tmp);

/*
char *tmp = NULL;
tmp = (char *)malloc((size_t)(29 + 1) * sizeof(char));
assert(tmp != NULL);
*/
/*
char tmp[29];
for (size_t i = 1; i < 308915776; i++) {
    //tmp[(size_t)(i / 26 + 1)] = '\0';
    //tmp[(size_t)(29)] = '\0';
    findStringInv<size_t>(i, tmp);
    auto crc = ~(GetCrc32(tmp));
    if (std::find(std::begin(cheat_list), std::end(cheat_list), crc) != std::end(cheat_list)) {
        std::reverse(tmp, tmp + strlen(tmp));
        std::cout << tmp << ":0x" << std::hex << crc << std::endl;
    }
    tmp[0] = '\0';
}*/
// free(tmp);

/*
std::string tmps = "";
for (std::size_t i = 1; i < 308915776; i++) {//208827064576
    tmps = findStringInv<std::size_t>(i);
    auto crc = ~(GetCrc32(tmps));
    if (std::find(std::begin(cheat_list), std::end(cheat_list), crc) != std::end(cheat_list)) {
        std::reverse(tmps.begin(), tmps.end());
        std::cout << tmps << ":0x" << std::hex << crc << std::endl;
    }
}
*/
