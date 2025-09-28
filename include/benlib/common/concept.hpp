/**
 * @file concept.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2024-07-05
 *
 * MIT License
 *
 */

#ifndef CONCEPT_HPP_
#define CONCEPT_HPP_

#if __cplusplus >= 202002L

#include <concepts>

// Likw int, float etc...
template <typename Type>
concept ArithmeticType = std::is_arithmetic<Type>::value;

// Like int, long, short etc...
template <typename Type>
concept IntegerType = std::is_integral<Type>::value;

// Like float, double etc...
template <typename Type>
concept FloatingPointType = std::is_floating_point<Type>::value;

// Like int, long, short etc...
template <typename Type>
concept SignedType = std::is_signed<Type>::value;

// Like unsigned int, unsigned long, unsigned short etc...
template <typename Type>
concept UnsignedType = std::is_unsigned<Type>::value;

// Like bool
template <typename Type>
concept BooleanType = std::is_same<Type, bool>::value;

#else

#error "This header requires C++20 or later"

#endif

#endif