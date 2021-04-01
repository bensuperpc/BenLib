/**
 * @file arduino_serial.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */
#ifndef ARDUINO_SERIAL_HPP
#define ARDUINO_SERIAL_HPP
#include <bitset>
#include <iostream>
#include <string>

/**
 * @brief enum : For println codage
 * @enum enum
 */
enum CODAGE
{
    BYTE,
    DEC,
    HEX,
    OCT,
    BIN
};

/**
 * @class Serial
 * @brief Need to simulate Serial on arduino
 */
class Serial {
  public:
    /**
        * @brief Write char[] or string on serial port
        * 
        * @param str 
        */
    static void println(const std::string &str);

    /**
         * @brief Write char[] or string on serial port with specific codage
         * 
         * @param str 
         * @param codage 
         */
    static void println(const std::string &str, const CODAGE codage);

    /**
         * @brief 
         * 
         * @param str 
         * @param baud 
         * @return int 
         */
    static int openDevice(const std::string &str, const int baud);

    /**
         * @brief 
         * 
         * @param baud 
         */
    static void begin(const int baud);

    /**
         * @brief 
         * 
         * @param c 
         */
    static void writeChar(const char c);

    /**
         * @brief Write char on serial port
         * 
         */
    static void closeDevice();

    /**
         * @brief 
         * 
         * @return true 
         * @return false 
         */
    static bool available();
};

#endif
