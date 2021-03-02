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
//          https://web.archive.org/web/20090204140550/http://www.maxbot.com/gta/3wordcheatsdumpsorted.txt                                                //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include <algorithm> // for std::find
#include <boost/crc.hpp>
#include <cmath> // pow
#include <cstring>
#include <iomanip>  // setw
#include <iostream> // cout
#include <mutex>    // Mutex
#include <string>
#include <string_view> // string_view
#include <tuple>
#include <utility> // std::make_pair
#include <vector>
#include "thread/Pool.hpp" // Threadpool

constexpr std::uint32_t alphabetSize {26};

const std::array<unsigned int, 87> cheat_list {0xDE4B237D, 0xB22A28D1, 0x5A783FAE, 0xEECCEA2B, 0x42AF1E28, 0x555FC201, 0x2A845345, 0xE1EF01EA, 0x771B83FC,
    0x5BF12848, 0x44453A17, 0xFCFF1D08, 0xB69E8532, 0x8B828076, 0xDD6ED9E9, 0xA290FD8C, 0x3484B5A7, 0x43DB914E, 0xDBC0DD65, 0xD08A30FE, 0x37BF1B4E, 0xB5D40866,
    0xE63B0D99, 0x675B8945, 0x4987D5EE, 0x2E8F84E8, 0x1A9AA3D6, 0xE842F3BC, 0x0D5C6A4E, 0x74D4FCB1, 0xB01D13B8, 0x66516EBC, 0x4B137E45, 0x78520E33, 0x3A577325,
    0xD4966D59, 0x5FD1B49D, 0xA7613F99, 0x1792D871, 0xCBC579DF, 0x4FEDCCFF, 0x44B34866, 0x2EF877DB, 0x2781E797, 0x2BC1A045, 0xB2AFE368, 0xFA8DD45B, 0x8DED75BD,
    0x1A5526BC, 0xA48A770B, 0xB07D3B32, 0x80C1E54B, 0x5DAD0087, 0x7F80B950, 0x6C0FA650, 0xF46F2FA4, 0x70164385, 0x885D0B50, 0x151BDCB3, 0xADFA640A, 0xE57F96CE,
    0x040CF761, 0xE1B33EB9, 0xFEDA77F7, 0x8CA870DD, 0x9A629401, 0xF53EF5A5, 0xF2AA0C1D, 0xF36345A8, 0x8990D5E1, 0xB7013B1B, 0xCAEC94EE, 0x31F0C3CC, 0xB3B3E72A,
    0xC25CDBFF, 0xD5CF4EFF, 0x680416B1, 0xCF5FDA18, 0xF01286E9, 0xA841CC0A, 0x31EA09CF, 0xE958788A, 0x02C83A7C, 0xE49C3ED4, 0x171BA8CC, 0x86988DAE, 0x2BDD2FA1};

const std::array<const std::string, 87> cheat_list_name {"Weapon Set 1", "Weapon Set 2", "Weapon Set 3", "Health, Armor, $250k, Repairs car",
    "Increase Wanted Level +2", "Clear Wanted Level", "Sunny Weather", "Very Sunny Weather", "Overcast Weather", "Rainy Weather", "Foggy Weather",
    "Faster Clock", "N°12", "N°13", "People attack each other with golf clubs", "Have a bounty on your head", "Everyone is armed", "Spawn Rhino",
    "Spawn Bloodring Banger", "Spawn Rancher", "Spawn Racecar", "Spawn Racecar", "Spawn Romero", "Spawn Stretch", "Spawn Trashmaster", "Spawn Caddy",
    "Blow Up All Cars", "Invisible car", "All green lights", "Aggressive Drivers", "Pink CArs", "Black Cars", "Fat Body", "Muscular Body", "Skinny Body",
    "People attack with Rocket Launchers", "N°41", "N°42", "Gangs Control the Streets", "N°44", "Slut Magnet", "N°46", "N°47", "Cars Fly", "N°49", "N°50",
    "Spawn Vortex Hovercraft", "Smash n' Boom", "N°53", "N°54", "N°55", "Orange Sky", "Thunderstorm", "Sandstorm", "N°59", "N°60", "Infinite Health",
    "Infinite Oxygen", "Have Parachute", "N°64", "Never Wanted", "N°66", "Mega Punch", "Never Get Hungry", "N°69", "N°70", "N°71", "N°72",
    "Full Weapon Aiming While Driving", "N°74", "Traffic is Country Vehicles", "Recruit Anyone (9mm)", "Get Born 2 Truck Outfit", "N°78", "N°79", "N°80",
    "L3 Bunny Hop", "N°82", "N°83", "N°84", "Spawn Quad", "Spawn Tanker Truck", "Spawn Dozer", "pawn Stunt Plane", "Spawn Monster"};

std::mutex mutex;

std::vector<std::tuple<std::size_t, std::string, unsigned int>> results = {};

unsigned int GetCrc32(const std::string_view my_string);

unsigned int GetCrc32(const std::string_view my_string)
{
    boost::crc_32_type result;
    // ça c'est plus rapide que l'autre normalement pour le length. Ça donne le nombre d'élément, donc si tu as plusieurs éléments qui sont à '\0' forcément…
    result.process_bytes(my_string.data(), my_string.length());
    return result.checksum();
}

/**
 * \brief Generate Alphabetic sequence from size_t value, A=1, Z=27, AA = 28, AB = 29
 * \tparam T
 * \param n
 * \param array
 */
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
        char tmp[29] = {0};
        ; // Temp array
        uint32_t crc = 0;
        ; // CRC value
        for (size_t i = x; i < y + x; i++) {
            findStringInv<size_t>(i, tmp); // Generate Alphabetic sequence from size_t value, A=1, Z=27, AA = 28, AB = 29
            crc = ~(GetCrc32(tmp));        // Get CRC32 and apply bitwise not, to convert CRC32 to JAMCRC
            if (std::find(std::begin(cheat_list), std::end(cheat_list), crc) != std::end(cheat_list)) { // If crc is present in Array
                std::reverse(tmp, tmp + strlen(tmp));                                                   // Invert char array
                mutex.lock();
#ifdef DNDEBUG
                std::cout << std::dec << i << ":" << std::string(tmp) << ":0x" << std::hex << crc << std::endl;
#endif
                results.emplace_back(std::make_tuple(i, std::string(tmp), crc)); // Save result: calculation position, Alphabetic sequence, CRC
                mutex.unlock();
            }
        }
        return 0;
    }
};

int main()
{
    std::ios_base::sync_with_stdio(false); // Improve std::cout and std::cin speed

    std::vector<std::future<std::size_t>> results_pool {};

    const size_t from_range = 1;        // Alphabetic sequence range min
    const size_t to_range = 8031810176; // Alphabetic sequence range max

#ifdef DNDEBUG
    assert(from_range < to_range);
    assert(from_range > 0);
#endif

    const size_t nbrcal = to_range - from_range + 1;                    // Number of calculations to do
    const std::size_t hardthread = std::thread::hardware_concurrency(); // Number of threads in the threadpool
    const std::size_t threadmult = 128;                                 // Thread Multiplier (So that each pool has multiple operations available)

    thread::Pool thread_pool(hardthread);

    const size_t nbrtask = hardthread * threadmult; // Total number of tasks created on the threadpool

    const size_t nbrcalperthread = nbrcal / nbrtask; // Number of calculations per task (100K mini to 100M max recommended)

    std::cout << "Threadpool with: " << hardthread << " threads" << std::endl;
    std::cout << "Thread multiplier: x" << threadmult << " (Nbr tasks per thread)" << std::endl;
    std::cout << "Threadpool with: " << nbrtask << " tasks" << std::endl;
    std::cout << "Number of calculations: " << nbrcal << std::endl;
    std::cout << "Number of calculations per tasks: " << nbrcalperthread << std::endl;
    std::cout << "" << std::endl;
    // Display Alphabetic sequence range
    char tmp1[29] = {0};
    char tmp2[29] = {0};
    findStringInv<size_t>(from_range, tmp1);
    findStringInv<size_t>(to_range, tmp2);
    std::cout << "From: " << tmp1 << " to: " << tmp2 << " Alphabetic sequence" << std::endl;
    std::cout << "" << std::endl;

    results_pool.reserve(nbrtask); // Vectors reservation

    for (std::size_t i = from_range; i < nbrtask; i++) {
        results_pool.emplace_back(thread_pool.enqueue(Processor(), i * nbrcalperthread, nbrcalperthread)); // Send work to be done to the threadpool
    }

    std::size_t count = 0;
    std::size_t t __attribute__((unused));

    for (auto &&result_pool : results_pool) {
        t = result_pool.get(); // Get result from threadpool

        // Calculate work %
        if (count % (std::size_t)(nbrtask / 50) == 0) {
            std::cout << double(count) / double(nbrtask) * 100.0f << " %" << std::endl;
        }
        count++;

        // Free results if is "full"
        /*
        if(results.length() >= 10000){
            mutex.lock();
            mutex.unlock();
        }*/
    }

    std::cout << "" << std::endl;
    std::cout << std::left << std::setw(13) << "Calc N°" << std::left << std::setw(12) << "Cheat Code" << std::left << std::setw(16) << "CRC32/JAMCRC"
              << "Cheat code name" << std::endl;

    for (auto &&result : results) {
        std::cout << std::left << std::setw(14) << std::dec << std::get<0>(result) << std::left << std::setw(12) << std::get<1>(result) << "0x" << std::hex
                  << std::left << std::setw(12) << std::get<2>(result);
        auto it = std::find(cheat_list.begin(), cheat_list.end(), std::get<2>(result));
        if (it != cheat_list.end()) {
            std::cout << cheat_list_name[it - cheat_list.begin()] << std::endl;
        }
    }
    return EXIT_SUCCESS;
}