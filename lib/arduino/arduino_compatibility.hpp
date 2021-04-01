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

extern unsigned int __bss_end;
extern unsigned int __heap_start;

#define HIGH 1
#define LOW 0

#define INPUT 0
#define OUTPUT 1

#define WDTO_15MS 1

using byte = char;
//#define digitalWrite(pin, value) digitalWrite_standard(pin, value)
static int pin1 = 0;


/**
 * @brief Init Serial
 * 
 */
Serial_arduino Serial;


#define A0 0
#define A1 1
#define A3 3

void pinMode(const int pin_value, const int inout);
void pinMode(const int pin_value, const int inout)
{
}

void digitalWrite(const int pin, const int value);
void digitalWrite(const int pin, const int value)
{
    std::cout << "Pin N°" << pin << " Value: " << value << "\n";
}

int digitalRead(const int pin);
int digitalRead(const int pin)
{
    return 0;
}

void analogWrite(const int pin, const int value);
void analogWrite(const int pin, const int value)
{
    std::cout << "Pin N°" << pin << " Value: " << value << "\n";
}

int analogRead(const int pin);
int analogRead(const int pin)
{
    return 0;
}

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