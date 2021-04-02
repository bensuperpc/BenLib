/**
 * @file arduino_time.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

#ifndef ARDUINO_TIME_HPP
#define ARDUINO_TIME_HPP

#include <chrono>
#include <iostream>
#include <thread>

std::chrono::steady_clock::time_point time_start_since_launch;

/**
 * @brief 
 *
 * @ingroup Arduino_time
 * 
 */
void start_program();

/**
 * @brief 
 *
 * @ingroup Arduino_time
 * 
 * @return unsigned int 
 */
unsigned int millis();

/**
 * @brief 
 *
 * @ingroup Arduino_time
 * 
 * @param value 
 */
void delay(const unsigned int value);

/**
 * @brief 
 *
 * @ingroup Arduino_time
 *  
 * @param value 
 */
void delay(const int value);

/**
 * @brief 
 *
 * @ingroup Arduino_time
 * 
 * @param value 
 */
void delayMicroseconds(const unsigned int value);

#endif
