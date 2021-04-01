/**
 * @file arduino_serial.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

#include "arduino_serial.hpp"

void Serial_arduino::println(const std::string &str)
{
    std::cout << str << "\n";
}

void Serial_arduino::println(const std::string &str, const CODAGE codage)
{
    if (codage == HEX) {
        std::cout << std::hex << str << "\n";
    } else if (codage == DEC) {
        std::cout << std::dec << str << "\n";
    } else if (codage == OCT) {
        std::cout << std::oct << str << "\n";
    } else if (codage == BIN) {
        for (std::size_t i = 0; i < str.size(); ++i) {
            std::cout << std::bitset<8>(str.c_str()[i]);
        }
        std::cout << std::endl;
    } else if (codage == BYTE) {
        //std::cout << std::byte << str << "\n";
    } else {
    }
}

void Serial_arduino::println(const int i)
{
    std::cout << i << "\n";
}

void Serial_arduino::writeChar(const char c)
{
    std::cout << c;
}

int Serial_arduino::openDevice(const std::string &str, const int baud)
{
    return 0;
}

void Serial_arduino::begin(const int baud)
{
    std::cout << "Start with:" << baud << "Baud/s" << "\n";
}

Serial_arduino::Serial_arduino()
{

}

Serial_arduino::~Serial_arduino()
{

}