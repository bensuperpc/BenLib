/**
 * @file reboot_software.h
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief Servolent 2016 and Jeu de réflexe 2017
 *        BAC project 2016-2017
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */


#ifndef REBOOT_SOFTWARE_H
#    define REBOOT_SOFTWARE_H

/// If use GCC-AVR for ARDUINO
#if defined(__AVR_ATmega168__) || defined(__AVR_ATmega328P__) || defined(__AVR_ATmega2560__) || defined(__AVR_ATmega1280__) || \
    defined(__AVR_ATmega32U4__) || defined(__SAM3X8E__)
#include <avr/wdt.h>
#endif 

#include "arduino_compatibility.hpp"

/**
 * @brief 
 * 
 */
void reboot_software(void);

#endif
