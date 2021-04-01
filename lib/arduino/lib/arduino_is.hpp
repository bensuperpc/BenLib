/**
 * @file arduino_is.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 *
 * @ingroup Arduino_is
 *  @version 1.0.0
 * @date 2021-04-02
 * 
 * MIT License
 * 
 */

#ifndef ARDUINO_IS_HPP
#define ARDUINO_IS_HPP

/**
 * @brief 
 *
 * @ingroup Arduino_is
 * 
 * @param c 
 * @return true 
 * @return false 
 */
bool isSpace(const char c);

/**
 * @brief 
 *
 * @ingroup Arduino_is
 *  
 * @param c 
 * @return true 
 * @return false 
 */
bool isAlphaNumeric(const char c);

/**
 * @brief 
 *
 * @ingroup Arduino_is
 *  
 * @param c 
 * @return true 
 * @return false 
 */
bool isLowerCase(const char c);

/**
 * @brief 
 *
 * @ingroup Arduino_is
 *  
 * @param c 
 * @return true 
 * @return false 
 */
bool isUpperCase(const char c);

/**
 * @brief 
 *
 * @ingroup Arduino_is
 *  
 * @param c 
 * @return true 
 * @return false 
 */
bool isDigit(const char c);

/**
 * @brief 
 *
 * @ingroup Arduino_is
 *  
 * @param c 
 * @return true 
 * @return false 
 */
bool isControl(const char c);

/**
 * @brief 
 *
 * @ingroup Arduino_is
 *  
 * @param c 
 * @return true 
 * @return false 
 */
bool isWhitespace(const char c);

/**
 * @brief 
 *
 * @ingroup Arduino_is
 *  
 * @param c 
 * @return true 
 * @return false 
 */
bool isGraph(const char c);

/**
 * @brief 
 *
 * @ingroup Arduino_is
 *  
 * @param c 
 * @return true 
 * @return false 
 */
bool isAscii(const char c);

/**
 * @brief 
 *
 * @ingroup Arduino_is
 *  
 * @param c 
 * @return true 
 * @return false 
 */
bool isPunct(const char c);

/**
 * @brief 
 *
 * @ingroup Arduino_is
 *  
 * @param c 
 * @return true 
 * @return false 
 */
bool isPrintable(const char c);

/**
 * @brief 
 *
 * @ingroup Arduino_is
 *  
 * @param c 
 * @return true 
 * @return false 
 */
bool isHexadecimalDigit(const char c);

/**
 * @brief 
 *
 * @ingroup Arduino_is
 *  
 * @param c 
 * @return true 
 * @return false 
 */
bool isAlpha(const char c);

#endif
