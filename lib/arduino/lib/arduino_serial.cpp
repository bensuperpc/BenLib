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
#include "arduino/lib/arduino_string.hpp"

void Serial_arduino::println()
{
    std::cout << "\n";
}

void Serial_arduino::println(const std::string &str)
{
    std::cout << str << "\n";
}

void Serial_arduino::println(my::String &str)
{
    std::cout << str.get_data() << "\n";
}

void Serial_arduino::println(const int i)
{
    std::cout << i << "\n";
}

void Serial_arduino::print(const int i)
{
    std::cout << i;
}

void Serial_arduino::print(const std::string &str)
{
    std::cout << str;
}

void Serial_arduino::print(my::String &str)
{
    std::cout << str.get_data();
}

void Serial_arduino::println(const std::string &str, const CODAGE codage)
{
    println<std::string>(str, codage);
}

void Serial_arduino::println(const int i, const CODAGE codage)
{
    println<int>(i, codage);
}

template <typename T> void Serial_arduino::println(const T &value, const CODAGE codage)
{
    if (codage == HEX) {
        std::cout << std::hex << value << "\n";
    } else if (codage == DEC) {
        std::cout << std::dec << value << "\n";
    } else if (codage == OCT) {
        std::cout << std::oct << value << "\n";
    } else if (codage == BIN) {
        if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
            std::cout << std::bitset<8>((uint64_t)value) << "\n";
        } else {
            for (std::size_t i = 0; i < value.size(); ++i) {
                std::cout << std::bitset<8>((uint64_t)value.c_str()[i]);
            }
            std::cout << "\n";
        }
    } else if (codage == BYTE) {
        //std::cout << std::byte << str << "\n";
    } else {
    }
}

void Serial_arduino::println(const my::String &value, const CODAGE codage)
{
    if (codage == HEX) {
        std::cout << std::hex << value << "\n";
    } else if (codage == DEC) {
        std::cout << std::dec << value << "\n";
    } else if (codage == OCT) {
        std::cout << std::oct << value << "\n";
    } else if (codage == BIN) {
        for (std::size_t i = 0; i < value.get_data().size(); ++i) {
            std::cout << std::bitset<8>((uint64_t)value.get_data().c_str()[i]);
        }
        std::cout << "\n";
    } else if (codage == BYTE) {
        //std::cout << std::byte << str << "\n";
    } else {
    }
}

void Serial_arduino::print(const std::string &str, const CODAGE codage)
{
    print<std::string>(str, codage);
}

void Serial_arduino::print(const int i, const CODAGE codage)
{
    print<int>(i, codage);
}

template <typename T> void Serial_arduino::print(const T &value, const CODAGE codage)
{
    if (codage == HEX) {
        std::cout << std::hex << value;
    } else if (codage == DEC) {
        std::cout << std::dec << value;
    } else if (codage == OCT) {
        std::cout << std::oct << value;
    } else if (codage == BIN) {
        if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
            std::cout << std::bitset<8>((uint64_t)value);
        } else {
            for (std::size_t i = 0; i < value.size(); ++i) {
                std::cout << std::bitset<8>((uint64_t)value.c_str()[i]);
            }
        }
    } else if (codage == BYTE) {
        //std::cout << std::byte << str << "\n";
    } else {
    }
}

void Serial_arduino::print(const my::String &value, const CODAGE codage)
{
    if (codage == HEX) {
        std::cout << std::hex << value;
    } else if (codage == DEC) {
        std::cout << std::dec << value;
    } else if (codage == OCT) {
        std::cout << std::oct << value;
    } else if (codage == BIN) {
        for (std::size_t i = 0; i < value.get_data().size(); ++i) {
            std::cout << std::bitset<8>((uint64_t)value.get_data().c_str()[i]);
        }
    } else if (codage == BYTE) {
        //std::cout << std::byte << str << "\n";
    } else {
    }
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
    std::cout << "Start with:" << baud << " Baud/s"
              << "\n";
    this->was_launch = false;
}

void Serial_arduino::write(const int w)
{
    std::cout << (char)w;
}

void Serial_arduino::write(const char w)
{
    std::cout << (char)w;
}

int Serial_arduino::read()
{
    return 0;
}

bool Serial_arduino::available()
{
    return true;
}

bool Serial_arduino::operator!()
{
    return this->was_launch;
}

Serial_arduino::Serial_arduino()
{
}

Serial_arduino::~Serial_arduino()
{
}