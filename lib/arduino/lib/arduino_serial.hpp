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
#include <type_traits>
#include <typeinfo>
#include "arduino_string.hpp"
#include "arduino_type.hpp"

/**
 * @class Serial_arduino
 * @brief Need to simulate Serial_arduino on arduino
 */
class Serial_arduino {
  public:
    /**
     * @brief 
     * 
     */
    void println();

    /**
     * @brief 
     * 
     * @param str 
     */
    void println(const std::string &str);

    /**
     * @brief 
     * 
     * @param str 
     */
    void println(my::String &str);

    /**
     * @brief 
     * 
     * @param str 
     */
    void print(const std::string &str);

    /**
     * @brief 
     * 
     * @param str 
     */
    void print(my::String &str);

    /**
     * @brief 
     * 
     * @param i 
     */
    void println(const int i);

    /**
     * @brief 
     * 
     * @param i 
     */
    void print(const int i);

    /**
     * @brief 
     * 
     * @param str 
     * @param codage 
     */
    void println(const std::string &str, const CODAGE codage);

    /**
     * @brief 
     * 
     * @param str 
     * @param codage 
     */
    void println(const my::String &str, const CODAGE codage);

    /**
     * @brief 
     * 
     * @tparam T 
     * @param value 
     * @param codage 
     */
    template <typename T> void println(const T &value, const CODAGE codage);

    /**
     * @brief 
     * 
     * @tparam T 
     * @param value 
     * @param codage 
     */
    template <typename T> void print(const T &value, const CODAGE codage);

    /**
     * @brief 
     * 
     * @param value 
     * @param codage 
     */
    void print(const my::String &value, const CODAGE codage);

    /**
     * @brief 
     * 
     * @param str 
     * @param codage 
     */
    void print(const std::string &str, const CODAGE codage);

    /**
     * @brief 
     * 
     * @param str 
     * @param codage 
     */
    void print(const int i, const CODAGE codage);

    /**
     * @brief 
     * 
     * @param i 
     * @param codage 
     */
    void println(const int i, const CODAGE codage);

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
     * @brief 
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

    void write(const int w);
    void write(const char w);

    int read();

    /**
     * @brief 
     * 
     * @return true 
     * @return false 
     */
    bool operator!();

    Serial_arduino();
    ~Serial_arduino();

  private:
    bool was_launch = true;
};

#endif
