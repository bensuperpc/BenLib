/**
 * @file reboot_software.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-02
 * 
 * MIT License
 * 
 */

#ifndef REBOOT_SOFTWARE_H
#define REBOOT_SOFTWARE_H

/// If use GCC-AVR for ARDUINO
#if defined(__AVR_ATmega168__) || defined(__AVR_ATmega328P__) || defined(__AVR_ATmega2560__) || defined(__AVR_ATmega1280__) || defined(__AVR_ATmega32U4__)     \
    || defined(__SAM3X8E__)
#    include <avr/wdt.h>
#endif

#include "arduino_compatibility.hpp"

/**
 * @brief 
 * 
 */
void reboot_software(void);

#endif
