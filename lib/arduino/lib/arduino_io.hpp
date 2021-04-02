/**
 * @file arduino_io.hpp
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

/**
 * @brief 
 *
 * @ingroup Arduino_io
 * 
 * @param pin_value 
 * @param inout 
 */
void pinMode(const int pin_value, const int inout);

/**
 * @brief 
 *
 * @ingroup Arduino_io
 * 
 * @param pin 
 * @param value 
 */
void digitalWrite(const int pin, const int value);

/**
 * @brief 
 *
 * @ingroup Arduino_io
 *  
 * @param pin 
 * @param value 
 */
void analogWrite(const int pin, const int value);

/**
 * @brief 
 *
 * @ingroup Arduino_io
 * 
 * @param pin 
 * @return int 
 */
int digitalRead(const int pin);

/**
 * @brief 
 *
 * @ingroup Arduino_io
 * 
 * @param pin 
 * @return int 
 */
int analogRead(const int pin);

#endif
