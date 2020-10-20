/*
** Bensuperpc, 2016-2019
** Servolent 2016 and Jeu de réflexe 2017
** BAC project 2016-2017
** File description:
** >reboot_software.c
*/

#include "blink_number.h"

void blink_led(byte pin_address, int num_blinks, int blink_delay) {
  for (int i=0; i < num_blinks; i++) {
    digitalWrite(pin_address, HIGH);   
    delay(blink_delay);                
    digitalWrite(pin_address, LOW);   
    delay(blink_delay);
  }
}
