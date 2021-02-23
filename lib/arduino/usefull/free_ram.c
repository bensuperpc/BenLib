/*
** Bensuperpc, 2016-2019
** Servolent 2016 and Jeu de réflexe 2017
** BAC project 2016-2017
** File description:
** >free_ram.c
*/

#include "free_ram.h"

int freeRam()
{
    extern int __heap_start, *__brkval;
    int v;
    return (int)&v - (__brkval == 0 ? (int)&__heap_start : (int)__brkval);
}
