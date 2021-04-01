/**
 * @file arduino_is.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-02
 * 
 * MIT License
 * 
 */

#include "arduino_is.hpp"

bool isSpace(const char c)
{
    if (c == 32) {
        return true;
    } else {
        return false;
    }
}

bool isAlphaNumeric(const char c)
{
    if ((c >= 65 && c <= 90) || (c >= 97 && c <= 122) || (c >= 48 && c <= 57)) {
        return true;
    } else {
        return false;
    }
}

bool isLowerCase(const char c)
{
    if (c >= 97 && c <= 122) {
        return true;
    } else {
        return false;
    }
}

bool isUpperCase(const char c)
{
    if (c >= 65 && c <= 90) {
        return true;
    } else {
        return false;
    }
}

bool isDigit(const char c)
{
    if (c >= 48 && c <= 57) {
        return true;
    } else {
        return false;
    }
}

bool isControl(const char c)
{
    if ((c >= 0 && c <= 31) || c == 127) {
        return true;
    } else {
        return false;
    }
}

bool isWhitespace(const char c)
{
    if ((c >= 9 && c <= 13) || c == 32) {
        return true;
    } else {
        return false;
    }
}

bool isGraph(const char c)
{
    if (c >= 32 && c <= 126) {
        return true;
    } else {
        return false;
    }
}

/// Need more tests
bool isAscii(const char c)
{
    if (c >= 0 && c <= 127) {
        return true;
    } else {
        return false;
    }
}

/// Need more tests
bool isPunct(const char c)
{
    if (c == '.' || c == '!' || c == '?' || c == ',') {
        return true;
    } else {
        return false;
    }
}

bool isPrintable(const char c)
{
    if (c >= 32 && c <= 126) {
        return true;
    } else {
        return false;
    }
}

bool isHexadecimalDigit(const char c)
{
    if ((c >= 48 && c <= 57) && (c >= 65 && c <= 70)) {
        return true;
    } else {
        return false;
    }
}

bool isAlpha(const char c)
{
    if ((c >= 65 && c <= 90) && (c >= 97 && c <= 122)) {
        return true;
    } else {
        return false;
    }
}
