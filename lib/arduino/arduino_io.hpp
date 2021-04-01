/**
 * @file arduino_iohpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-02
 * 
 * MIT License
 * 
 */

#ifndef ARDUINO_IO_HPP
#define ARDUINO_IO_HPP

#define A0 0
#define A1 1
#define A3 3

#define HIGH 1
#define LOW 0

#define INPUT 0
#define OUTPUT 1

void pinMode(const int pin_value, const int inout);

void digitalWrite(const int pin, const int value);

void analogWrite(const int pin, const int value);

int digitalRead(const int pin);

int analogRead(const int pin);

#endif
