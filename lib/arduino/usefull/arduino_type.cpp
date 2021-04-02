/**
 * @file arduino_type.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-02
 * 
 * MIT License
 * 
 */

#include "arduino_type.hpp"

int arduino_type()
{
#if defined(__AVR_ATmega168__) || defined(__AVR_ATmega328P__)
    // Serial.println("Regular Arduino");
    return 1;
#elif defined(__AVR_ATmega2560__) || defined(__AVR_ATmega1280__)
    // Serial.println("Arduino Mega");
    return 2;
#elif defined(__AVR_ATmega32U4__)
    // Serial.println("Arduino Leonardo");
    return 3;
#elif defined(__SAM3X8E__)
    // Serial.println("Arduino Due");
    return 4;
#else
    // Serial.println("Unknown");
    return 0;
#endif
}

void arduino_type_Serial()
{
#if defined(__AVR_ATmega168__) || defined(__AVR_ATmega328P__)
    Serial.println("Regular Arduino");
#elif defined(__AVR_ATmega2560__) || defined(__AVR_ATmega1280__)
    Serial.println("Arduino Mega");
#elif defined(__AVR_ATmega32U4__)
    Serial.println("Arduino Leonardo");
#elif defined(__SAM3X8E__)
    Serial.println("Arduino Due");
#else
    Serial.println("Unknown");
#endif
}
