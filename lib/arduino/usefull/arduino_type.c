/*
** Bensuperpc, 2016-2019
** Servolent 2016 and Jeu de réflexe 2017
** BAC project 2016-2017
** File description:
** >arduino_type.c
*/

#include "arduino_type.h"

int arduino_type() {
#if defined(__AVR_ATmega168__) || defined(__AVR_ATmega328P__)
  //Serial.println("Regular Arduino");
return 1;
#elif defined(__AVR_ATmega2560__) || defined(__AVR_ATmega1280__)
  //Serial.println("Arduino Mega");
return 2;
#elif defined(__AVR_ATmega32U4__)
  //Serial.println("Arduino Leonardo");
return 3;
#elif defined(__SAM3X8E__)
  //Serial.println("Arduino Due");
return 4;
#else
  //Serial.println("Unknown");
return 0;  
#endif
}

void arduino_type_serial() {
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
