/**
 * @file arduino_string.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-02
 * 
 * MIT License
 * 
 */

#ifndef ARDUINO_STRING_HPP
#define ARDUINO_STRING_HPP

#include <array>
#include <iostream>
#include <string>

/**
 * @brief 
 * @namespace my
 */
namespace my
{
/**
 * @class String
 * 
 * @brief 
 * 
 */
class String : public std::string {
  public:
    /**
     * @brief 
     *
     * @ingroup Arduino_string
     *
     * @param i 
     */
    void concat(const int &i);

    /**
     * @brief 
     *
     * @ingroup Arduino_string
     * 
     * @param i 
     */
    void concat(const unsigned int &i);

    /**
     * @brief 
     *
     * @ingroup Arduino_string
     * 
     * @param rhs 
     * @return String 
     */
    String operator+(const int &rhs);

    /**
     * @brief 
     *
     * @ingroup Arduino_string
     * 
     * @param rhs 
     * @return String 
     */
    String operator+(const unsigned int &rhs);

    /**
     * @brief 
     *
     * @ingroup Arduino_string
     * 
     * @param rhs 
     * @return String 
     */
    String operator+(const char &rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return String& 
     */
    String &operator=(const std::string &rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     */
    void operator+=(const unsigned int &rhs);

    /**
     * @brief 
     *
     * @ingroup Arduino_string
     * 
     * @param rhs 
     */
    void operator+=(const std::string &rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     */
    void operator+=(const my::String &rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return my::String& 
     */
    my::String &operator+=(const int &rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return my::String 
     */
    my::String operator+=(const unsigned int &rhs) const;

    /**
     * @brief 
     *
     * @ingroup Arduino_string
     * 
     * @param rhs 
     * @return String 
     */
    String to_string(const int &rhs);

    /**
     * @brief 
     *
     * @ingroup Arduino_string
     * 
     * @param rhs 
     * @return String 
     */
    String to_string(const unsigned int &rhs);

    /**
     * @brief Construct a new String object
     *
     * @ingroup Arduino_string
     * 
     */
    String();

    /**
     * @brief Construct a new String object
     *
     * @ingroup Arduino_string
     * 
     * @param str 
     */
    String(const std::string &str);

    /**
     * @brief Destroy the String object
     * 
     */
    ~String();

    /**
     * @brief Get the data object
     *
     * @ingroup Arduino_string
     * 
     * @return std::string 
     */
    std::string get_data() const;

    /**
     * @brief Set the data object
     * 
     * @param str 
     */
    void set_data(const std::string &str);

  private:
    /**
     * @brief 
     *
     * @ingroup Arduino_string
     * 
     */
    std::string data;
};
} // namespace my
#endif