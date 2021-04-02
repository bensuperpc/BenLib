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

#include <algorithm>
#include <array>
#include <cctype>
#include <iostream>
#include <sstream>
#include <string>
#include "arduino_type.hpp"

/**
 * @brief 
 * @namespace my
 */
namespace my
{
/**
 * @class String
 * 
 * @brief String
 * 
 */
class String : public std::string {
  public:
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
    String(const my::String &str);

    /**
     * @brief Construct a new String object
     *
     * @ingroup Arduino_string
     * 
     * @param str 
     */
    String(const std::string &str);

    /**
     * @brief Construct a new String object
     *
     * @ingroup Arduino_string
     *  
     * @param str 
     */
    String(const char *str);

    /**
     * @brief Construct a new String object
     *
     * @ingroup Arduino_string
     *  
     * @param str 
     */
    String(const char str);

    /**
     * @brief Construct a new String object
     *
     * @ingroup Arduino_string
     *  
     * @param str 
     */
    String(const char str, CODAGE);

    /**
     * @brief Construct a new String object
     *
     * @ingroup Arduino_string
     *  
     * @param nbr 
     * @param i 
     */
    String(const float nbr, const int i);

    /**
     * @brief Construct a new String object
     *
     * @ingroup Arduino_string
     *  
     * @param i 
     */
    String(const int i);

    /**
     * @brief Construct a new String object
     *
     * @ingroup Arduino_string
     *  
     * @param i 
     */
    String(const int i, CODAGE);

    /**
     * @brief Construct a new String object
     *
     * @ingroup Arduino_string
     *  
     * @param i 
     */
    String(const unsigned int i);

    /**
     * @brief Construct a new String object
     *
     * @ingroup Arduino_string
     * 
     * @param i 
     */
    String(const unsigned int i, CODAGE);

    /**
     * @brief Destroy the String object
     *
     * @ingroup Arduino_string
     * 
     * @ingroup Arduino_string
     *  
     */
    ~String();

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
     * @param i 
     */
    void concat(const unsigned int &i);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return my::String& 
     */
    my::String &operator+(const int &rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return my::String& 
     */
    my::String &operator+(const unsigned int &rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return my::String& 
     */
    my::String &operator+(const char &rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return my::String& 
     */
    my::String &operator+(const std::string &rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return my::String& 
     */
    my::String &operator+(const char *rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return my::String& 
     */
    my::String &operator+(const my::String &rhs);

    /**
     * @brief 
     *
     * @ingroup Arduino_string
     * 
     * @param rhs 
     * @return String& 
     */
    my::String &operator=(const std::string &rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return my::String& 
     */
    my::String &operator=(const char *rhs);

    /**
     * @brief 
     *
     * @ingroup Arduino_string
     * 
     * @param rhs 
     * @return my::String& 
     */
    my::String &operator+=(const unsigned int &rhs);

    /**
     * @brief 
     *
     * @ingroup Arduino_string
     *  
     * @param rhs 
     * @return my::String& 
     */
    my::String &operator+=(const std::string &rhs);

    /**
     * @brief 
     *
     * @ingroup Arduino_string
     *  
     * @param rhs 
     * @return my::String& 
     */
    my::String &operator+=(const my::String &rhs);

    /**
     * @brief 
     *
     * @ingroup Arduino_string
     *   
     * @param rhs 
     * @return my::String& 
     */
    my::String &operator+=(const char *rhs);

    /**
     * @brief 
     *
     * @ingroup Arduino_string
     *  
     * @param rhs 
     * @return my::String& 
     */
    my::String &operator+=(const int &rhs);

    /**
     * @brief 
     *
     * @ingroup Arduino_string
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
     * @ingroup Arduino_string
     *  
     * @param str 
     */
    void set_data(const std::string &str);

    /**
     * @brief 
     *
     * @ingroup Arduino_string
     *   
     */
    void toUpperCase();

    /**
     * @brief 
     *
     * @ingroup Arduino_string
     *   
     */
    void toLowerCase();

    /**
     * @brief 
     *
     * @ingroup Arduino_string
     *   
     * @param os 
     * @return std::ostream& 
     */
    std::ostream &operator<<(std::ostream &os);

    /**
     * @brief 
     *
     * @ingroup Arduino_string
     *   
     * @param os 
     * @return std::ostream& 
     */
    std::ostream &operator>>(std::ostream &os);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return true 
     * @return false 
     */
    bool equals(const my::String &rhs);
    /// https://stackoverflow.com/a/12568475/10152334

    /**
     * @brief 
     * 
     * @param rhs 
     * @return true 
     * @return false 
     */
    bool equalsIgnoreCase(const my::String &rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return int 
     */
    int compareTo(const my::String &rhs);

    /**
     * @brief Convert valide string to int
     * 
     * @return int 
     */
    int toInt();

    /**
     * @brief 
     * 
     * @param rhs 
     * @return true 
     * @return false 
     */
    bool operator==(const my::String &rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return true 
     * @return false 
     */
    bool operator==(const char *rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return true 
     * @return false 
     */
    bool operator>=(const my::String &rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return true 
     * @return false 
     */
    bool operator<=(const my::String &rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return true 
     * @return false 
     */
    bool operator!=(const my::String &rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return true 
     * @return false 
     */
    bool operator>=(const char *rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return true 
     * @return false 
     */
    bool operator<=(const char *rhs);

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