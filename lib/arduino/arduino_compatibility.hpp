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

#ifndef ARDUINO_COMPATIBILITY_HPP
#define ARDUINO_COMPATIBILITY_HPP

#include <chrono>
#include <iostream>

#include "arduino_serial.hpp"
#include "arduino_time.hpp"
#include "arduino_io.hpp"

extern unsigned int __bss_end;
extern unsigned int __heap_start;

#define WDTO_15MS 1

using byte = char;
//#define digitalWrite(pin, value) digitalWrite_standard(pin, value)
static int pin1 = 0;

/**
 * @brief Init Serial
 * 
 */
Serial_arduino Serial;



void wdt_enable(const int value);
void wdt_enable(const int value)
{
}

long map(long x, long in_min, long in_max, long out_min, long out_max);
long map(long x, long in_min, long in_max, long out_min, long out_max) 
{
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

int map(int x, int in_min, int in_max, int out_min, int out_max);
int map(int x, int in_min, int in_max, int out_min, int out_max) 
{
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}


struct EEPROM
{
    void write(const int adress, const int valuie);
    int read(const int adress);
    void clear(const int adress);
};
#endif