/**
 * @file reboot_software.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-02
 * 
 * MIT License
 * 
 */

#include "reboot_software.hpp"

void reboot_software(void)
{
    wdt_enable(WDTO_15MS);
}
