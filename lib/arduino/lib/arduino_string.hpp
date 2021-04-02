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
#include <string>
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
    void concat(const int &i);
    void concat(const unsigned int &i);

    String operator+(const int &rhs);
    String operator+(const unsigned int &rhs);
    String operator+(const char &rhs);
    String operator=(const std::string &rhs);
    void operator+=(const int &rhs);
    void operator+=(const std::string &rhs);

    String to_string(const int &rhs);
    String to_string(const unsigned int &rhs);

    String();
    String(const std::string &str);

    ~String();
    std::string get_data();

  private:
    std::string data;
};
} // namespace my
#endif