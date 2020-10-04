/*
** BENSUPERPC PROJECT, 2020
** Vector
** Source:  https://stackoverflow.com/questions/178265/what-is-the-most-hard-to-understand-piece-of-c-code-you-know
**          https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
**          https://stackoverflow.com/questions/51274287/computing-8-horizontal-sums-of-eight-avx-single-precision-floating-point-vectors
**          https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_add_sd&expand=154
**          https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
**          https://godbolt.org/z/OLgHUs
** vector.hpp
*/

#ifndef VECTOR_AVX_HPP_
#define VECTOR_AVX_HPP_
#if (__AVX2__ || __AVX__ || __SSE3__)
#include <immintrin.h>

//__m256 m1 = _mm256_setr_ps(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0);

namespace my
{
namespace vector_avx
{
#        if (__AVX2__ || __AVX__)
inline __m256 HorizontalSums(__m256 &, __m256 &, __m256 &, __m256 &, __m256 &, __m256 &, __m256 &, __m256 &);
inline __m256 HorizontalSums_less_p5_pressure(__m256 &, __m256 &, __m256 &, __m256 &, __m256 &, __m256 &, __m256 &, __m256 &);
inline __m256 sort_of_alternative_hadd_ps(__m256 &, __m256 &);

inline double hsum_double_avx(__m256d &);
inline float hsum_float_avx(__m256 &);

inline void multiply_and_add(__m256 &, __m256 &, __m256 &, __m256 &);
#        endif

inline float extract_float(const __m128 &, const int i);

inline float *__m128_to_float(const __m128 &);
#        if (__AVX2__ || __AVX__)
inline float *__m256_to_float(const __m256 &);
#        endif

inline double *__m128_to_double(const __m128d &);
#        if (__AVX2__ || __AVX__)
inline double *__m256_to_double(const __m256d &);
#        endif

inline int *__m128_to_integer(const __m128i &);
#        if (__AVX2__ || __AVX__)
inline int *__m256_to_integer(const __m256i &);
#        endif

inline __m128i double_to_uint64(__m128d &);
inline __m128i double_to_int64(__m128d &);
inline __m128d uint64_to_double(__m128i &);
inline __m128d int64_to_double(__m128i &);
inline __m128d uint64_to_double_full(__m128i &);
inline __m128d int64_to_double_full(__m128i &);
#        if (__AVX2__ || __AVX__)
inline __m256d int64_to_double_fast_precise(const __m256i &);

inline __m256d uint64_to_double_full_range(const __m256i &);
inline __m256d int64_to_double_fast_precise_no_FM(const __m256i &);
inline __m256d int64_to_double_based_on_cvtsi2sd(const __m256i &);
inline __m256d int64_to_double_full_range(const __m256i &);
#        endif

} // namespace vector_avx
} // namespace my
#    endif
#endif
//_mm256_add_ps/_mm256_add_pd
//_mm256_add_epi8/16/32/64 / _mm256_add_epu8/16/32/64
//_mm256_mul_ps/_mm256_mul_pd
//_mm256_div_ps/_mm256_div_pd
//_mm256_sub_ps/_mm256_sub_pd
//__m512