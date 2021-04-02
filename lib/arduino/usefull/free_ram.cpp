/**
 * @file free_ram.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

#include "free_ram.hpp"

unsigned int freeRam()
{
    //extern unsigned int __heap_start, *__brkval;
    unsigned int __heap_start, *__brkval;
    unsigned int v;
    return &v - (__brkval == 0 ? &__heap_start : __brkval);
}
