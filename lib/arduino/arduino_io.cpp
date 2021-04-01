/**
 * @file arduino_io.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-02
 * 
 * MIT License
 * 
 */

#include "arduino_io.hpp"


void pinMode(const int pin_value, const int inout)
{
    if (inout == OUTPUT) {
        std::cout << "Pin N°" << pin_value << " OUTPUT" << std::endl;
    } else {
        std::cout << "Pin N°" << pin_value << " INPUT" << std::endl;
    }
}


void digitalWrite(const int pin, const int value)
{
    std::cout << "Pin N°" << pin << " Value: " << value << "\n";
}


int digitalRead(const int pin)
{
    return 0;
}


void analogWrite(const int pin, const int value)
{
    std::cout << "Pin N°" << pin << " Value: " << value << "\n";
}


int analogRead(const int pin)
{
    return 0;
}