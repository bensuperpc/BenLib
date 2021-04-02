/**
 * @file arduino_compatibility.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

/** @defgroup Arduino_software Arduino software
 *  @brief The main Arduino_software group who contain all software to Arduino card
 */
/** @defgroup Arduino_io Arduino io software
 *  @ingroup Arduino_software
 *  @brief io software
 *  @sa @link Arduino_software The first group Arduino_software@endlink
 */
/** @defgroup Arduino_math Arduino math software
 *  @ingroup Arduino_software
 *  @brief math software
 *  @sa @link Arduino_software The first group Arduino_software@endlink
 */
/** @defgroup Arduino_time Arduino time software
 *  @ingroup Arduino_software
 *  @brief time software
 *  @sa @link Arduino_software The first group Arduino_software@endlink
 */
/** @defgroup Arduino_is Arduino is software
 *  @ingroup Arduino_software
 *  @brief is software
 *  @sa @link Arduino_software The first group Arduino_software@endlink
 */

#ifndef ARDUINO_COMPATIBILITY_HPP
#define ARDUINO_COMPATIBILITY_HPP

#include <chrono>
#include <iostream>
//#include "arduino/lib/arduino_string.hpp"
#include "arduino_io.hpp"
#include "arduino_is.hpp"
#include "arduino_math.hpp"
#include "arduino_serial.hpp"
#include "arduino_string.hpp"
#include "arduino_time.hpp"

extern unsigned int __bss_end;
extern unsigned int __heap_start;

#define WDTO_15MS 1

using byte = char;

using String = my::String;

///using String = std::string;

/**
 * @brief Init Serial
 * 
 */
Serial_arduino Serial;

void wdt_enable(const int value);
void wdt_enable(const int value)
{
}
struct EEPROM
{
    void write(const int adress, const int valuie);
    int read(const int adress);
    void clear(const int adress);
};
#endif