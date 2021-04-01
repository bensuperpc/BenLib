/**
 * @file arduino_time.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

#include "arduino_time.hpp"

unsigned int millis()
{
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - time_start_since_launch).count();
}

void delay(const unsigned int value)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(value));
}

void delay(const int value)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(value));
}

void delayMicroseconds(const unsigned int value)
{
    std::this_thread::sleep_for(std::chrono::microseconds(value));
}