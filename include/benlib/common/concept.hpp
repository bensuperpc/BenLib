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

#if __cplusplus == 202002L

#include <concepts>

template <typename Type>
concept ArithmeticType = std::is_arithmetic<Type>::value;

template <typename Type>
concept IntegerType = std::is_integral<Type>::value;

template <typename Type>
concept FloatingPointType = std::is_floating_point<Type>::value;

template <typename Type>
concept SignedType = std::is_signed<Type>::value;

template <typename Type>
concept UnsignedType = std::is_unsigned<Type>::value;

#endif

#endif