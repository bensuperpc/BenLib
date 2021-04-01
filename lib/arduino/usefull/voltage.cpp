/*
** Bensuperpc, 2016-2019
** Servolent 2016 and Jeu de réflexe 2017
** BAC project 2016-2017
** Source: https://www.codeproject.com/Tips/987180/Arduino-Tips-Tricks
** >voltage.c
*/

#include "voltage.hpp"

int GetVccMiliVolts()
{
#if defined(ARDUINO_ARCH_AVR)
    const long int scaleConst = 1156.300 * 1000;
// Read 1.1V reference against Avcc
#    if defined(__AVR_ATmega32U4__) || defined(__AVR_ATmega1280__) || defined(__AVR_ATmega2560__)
    ADMUX = _BV(REFS0) | _BV(MUX4) | _BV(MUX3) | _BV(MUX2) | _BV(MUX1);
#    elif defined(__AVR_ATtiny24__) || defined(__AVR_ATtiny44__) || defined(__AVR_ATtiny84__)
    ADMUX = _BV(MUX5) | _BV(MUX0);
#    elif defined(__AVR_ATtiny25__) || defined(__AVR_ATtiny45__) || defined(__AVR_ATtiny85__)
    ADMUX = _BV(MUX3) | _BV(MUX2);
#    else
    ADMUX = _BV(REFS0) | _BV(MUX3) | _BV(MUX2) | _BV(MUX1);
#    endif
    delay(2);            // Wait for Vref to settle
    ADCSRA |= _BV(ADSC); // Start conversion
    while (bit_is_set(ADCSRA, ADSC))
        ;                // measuring
    uint8_t low = ADCL;  // must read ADCL first - it then locks ADCH
    uint8_t high = ADCH; // unlocks both
    long int result = (high << 8) | low;
    result = scaleConst / result;
    // Calculate Vcc (in mV); 1125300 = 1.1*1023*1000
    return (int)result; // Vcc in millivolts
#else
    return -1;
#endif
}