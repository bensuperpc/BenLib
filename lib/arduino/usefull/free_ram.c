/**
 * @file free_ram.c
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief Servolent 2016 and Jeu de réflexe 2017
 *        BAC project 2016-2017
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */


#include "free_ram.h"

int freeRam()
{
    extern int __heap_start, *__brkval;
    int v;
    return (int)&v - (__brkval == 0 ? (int)&__heap_start : (int)__brkval);
}
