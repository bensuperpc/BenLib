/**
 * @file blink_number.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-02
 * 
 * MIT License
 * 
 */

#ifndef BLINK_NUMBER_H
#define BLINK_NUMBER_H

#include "arduino_compatibility.hpp"

/**
 * @brief 
 *
 * @ingroup Arduino_io
 *  
 * @param targetPin 
 * @param numBlinks 
 * @param blinkRate 
 */
void blink_led(byte targetPin, int numBlinks, int blinkRate);

#endif
