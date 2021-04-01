/**
 * @file blink_number.c
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief Servolent 2016 and Jeu de réflexe 2017
 *        BAC project 2016-2017
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

#include "blink_number.hpp"

void blink_led(byte pin_address, int num_blinks, int blink_delay)
{
    for (int i = 0; i < num_blinks; i++) {
        digitalWrite(pin_address, HIGH);
        delay(blink_delay);
        digitalWrite(pin_address, LOW);
        delay(blink_delay);
    }
}
