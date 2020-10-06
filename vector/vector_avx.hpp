/*
** BENSUPERPC PROJECT, 2020
** Vector
** Source:  https://stackoverflow.com/questions/178265/what-is-the-most-hard-to-understand-piece-of-c-code-you-know
**          https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
**          https://stackoverflow.com/questions/51274287/computing-8-horizontal-sums-of-eight-avx-single-precision-floating-point-vectors
**          https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_add_sd&expand=154
**          https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
**          https://godbolt.org/z/OLgHUs
**          https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=3327,4174,4174,3576,5194&techs=SSE,SSE2,SSE3,SSSE3,SSE4_1,SSE4_2&text=Shuffle%25252525252525252520
**          https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=3327,4174,4174,5153,5144,4174,4179&text=_mm256_permute4x64_epi64&techs=AVX,AVX2,FMA
**          https://www.programmersought.com/article/85712182324/
**          https://db.in.tum.de/~finis/x86-intrin-cheatsheet-v2.1.pdf
**          https://stackoverflow.com/questions/56033329/sse-shuffle-permutevar-4x32-integers
** vector.hpp
*/

#ifndef VECTOR_AVX_HPP_
#define VECTOR_AVX_HPP_
#if (__AVX2__ || __AVX__ || __SSE3__)
#include <immintrin.h>
#include <emmintrin.h>
#include <mmintrin.h>

#define _MM_SHUFFLE8(fp7, fp6, fp5, fp4, fp3, fp2, fp1, fp0)\
(((fp7) << 21) | ((fp6) << 18) | ((fp5) << 15) | ((fp4) << 12)) | \
(((fp3) << 9) | ((fp2) << 6) | ((fp1) << 3) | ((fp0)))


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
#if (__AVX2__ || __AVX__)
inline __m256d int64_to_double_fast_precise(const __m256i &);

inline __m256d uint64_to_double_full_range(const __m256i &);
inline __m256d int64_to_double_fast_precise_no_FM(const __m256i &);
inline __m256d int64_to_double_based_on_cvtsi2sd(const __m256i &);
inline __m256d int64_to_double_full_range(const __m256i &);
#        endif

__m128i _mm_shuffle_epi16(__m128i &, int &);
__m128i vperm(__m128i &, __m128i &);

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