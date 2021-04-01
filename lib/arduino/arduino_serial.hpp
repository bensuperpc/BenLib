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
 * @class Serial_arduino
 * @brief Need to simulate Serial_arduino on arduino
 */
class Serial_arduino {
  public:
    /**
        * @brief Write char[] or string on serial port
        * 
        * @param str 
        */
    void println(const std::string &str);

    /**
     * @brief 
     * 
     * @param i 
     */
    void println(const int i);

    /**
         * @brief Write char[] or string on serial port with specific codage
         * 
         * @param str 
         * @param codage 
         */
    void println(const std::string &str, const CODAGE codage);

    /**
         * @brief 
         * 
         * @param str 
         * @param baud 
         * @return int 
         */
    int openDevice(const std::string &str, const int baud);

    /**
         * @brief 
         * 
         * @param baud 
         */
    void begin(const int baud);

    /**
         * @brief 
         * 
         * @param c 
         */
    void writeChar(const char c);

    /**
         * @brief Write char on serial port
         * 
         */
    void closeDevice();

    /**
         * @brief 
         * 
         * @return true 
         * @return false 
         */
    bool available();
    Serial_arduino();
    ~Serial_arduino();
    private:
};

#endif
