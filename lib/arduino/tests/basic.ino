/**
 * @file basic.ino
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

//#include "arduino_compatibility.hpp"

/// Use C++ function but later replace it with arduino functions
void setup()
{
    pinMode(1, OUTPUT);
}

void loop()
{
    digitalWrite(1, HIGH);
    delay(500);
    digitalWrite(1, LOW);
    delay(500);
}