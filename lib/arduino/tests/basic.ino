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

/// Use C++ function but later replace it with arduino functions
#include <iostream>
void setup()
{
    std::cout << "OK" << std::endl;
}

void loop()
{
    std::cout << "Loop" << std::endl;
    return;
}