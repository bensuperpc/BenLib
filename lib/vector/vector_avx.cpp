/*
** BENSUPERPC PROJECT, 2020
** Vector
** Source:  https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_max_epi&expand=3327,3579,3597
**          https://stackoverflow.com/questions/178265/what-is-the-most-hard-to-understand-piece-of-c-code-you-know
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
**          https://stackoverflow.com/questions/9795529/how-to-find-the-horizontal-maximum-in-a-256-bit-avx-vector
** vector.cpp
*/

#include "vector_avx.hpp"

#if (__AVX2__ || __AVX__ || __SSE3__)
//__SSE__ __SSE2__
#    pragma GCC push_options
#    pragma GCC optimize("-O2")

#    if (__AVX2__ || __AVX__)
__m256 my::vector_avx::HorizontalSums(__m256 &v0, __m256 &v1, __m256 &v2, __m256 &v3, __m256 &v4, __m256 &v5, __m256 &v6, __m256 &v7)
{
    const __m256 &&s01 = _mm256_hadd_ps(v0, v1);
    const __m256 &&s23 = _mm256_hadd_ps(v2, v3);
    const __m256 &&s45 = _mm256_hadd_ps(v4, v5);
    const __m256 &&s67 = _mm256_hadd_ps(v6, v7);
    const __m256 &&s0123 = _mm256_hadd_ps(s01, s23);
    const __m256 &&s4556 = _mm256_hadd_ps(s45, s67);

    // inter-lane shuffle
    v0 = _mm256_blend_ps(s0123, s4556, 0xF0);
    v1 = _mm256_permute2f128_ps(s0123, s4556, 0x21);

    return _mm256_add_ps(v0, v1);
}

inline __m256 my::vector_avx::sort_of_alternative_hadd_ps(__m256 &x, __m256 &y)
{
    __m256 y_hi_x_lo = _mm256_blend_ps(x, y, 0b11001100);   /* y7 y6 x5 x4 y3 y2 x1 x0 */
    __m256 y_lo_x_hi = _mm256_shuffle_ps(x, y, 0b01001110); /* y5 y4 x7 x6 y1 y0 x3 x2 */
    return _mm256_add_ps(y_hi_x_lo, y_lo_x_hi);
}

__m256 my::vector_avx::HorizontalSums_less_p5_pressure(__m256 &v0, __m256 &v1, __m256 &v2, __m256 &v3, __m256 &v4, __m256 &v5, __m256 &v6, __m256 &v7)
{
    __m256 &&s01 = sort_of_alternative_hadd_ps(v0, v1);
    __m256 &&s23 = sort_of_alternative_hadd_ps(v2, v3);
    __m256 &&s45 = sort_of_alternative_hadd_ps(v4, v5);
    __m256 &&s67 = sort_of_alternative_hadd_ps(v6, v7);
    __m256 &&s0123 = _mm256_hadd_ps(s01, s23);
    __m256 &&s4556 = _mm256_hadd_ps(s45, s67);

    v0 = _mm256_blend_ps(s0123, s4556, 0xF0);
    v1 = _mm256_permute2f128_ps(s0123, s4556, 0x21);
    return _mm256_add_ps(v0, v1);
}
#    endif

#    if (__AVX2__ || __AVX__)
inline double my::vector_avx::hsum_double_avx(__m256d &v)
{
    __m128d &&vlow = _mm256_castpd256_pd128(v);
    __m128d &&vhigh = _mm256_extractf128_pd(v, 1);
    vlow = _mm_add_pd(vlow, vhigh);
    __m128d &&high64 = _mm_unpackhi_pd(vlow, vlow);
    return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
}

inline float my::vector_avx::hsum_float_avx(__m256 &v)
{
    const __m128 &&x128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    const __m128 &&x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 &&x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

inline void my::vector_avx::multiply_and_add(__m256 &a, __m256 &b, __m256 &c, __m256 &d)
{
    d = _mm256_fmadd_ps(a, b, c);
}
#    endif

inline float my::vector_avx::extract_float(const __m128 &v, const int i)
{
    float x = 0.0;
    _MM_EXTRACT_FLOAT(x, v, i);
    return x;
}

inline float *my::vector_avx::__m128_to_float(const __m128 &v)
{
    return (float *)&v;
}

#    if (__AVX2__ || __AVX__)
inline float *my::vector_avx::__m256_to_float(const __m256 &v)
{
    return (float *)&v;
}
#    endif

inline double *my::vector_avx::__m128_to_double(const __m128d &v)
{
    return (double *)&v;
}

#    if (__AVX2__ || __AVX__)
inline double *my::vector_avx::__m256_to_double(const __m256d &v)
{
    return (double *)&v;
}
#    endif

inline int *my::vector_avx::__m128_to_integer(const __m128i &v)
{
    return (int *)&v;
}

#    if (__AVX2__ || __AVX__)
inline int *my::vector_avx::__m256_to_integer(const __m256i &v)
{
    return (int *)&v;
}
#    endif

inline __m128i my::vector_avx::double_to_uint64(__m128d &x)
{
    x = _mm_add_pd(x, _mm_set1_pd(0x0010000000000000));
    return _mm_xor_si128(_mm_castpd_si128(x), _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)));
}

inline __m128i my::vector_avx::double_to_int64(__m128d &x)
{
    x = _mm_add_pd(x, _mm_set1_pd(0x0018000000000000));
    return _mm_sub_epi64(_mm_castpd_si128(x), _mm_castpd_si128(_mm_set1_pd(0x0018000000000000)));
}

inline __m128d my::vector_avx::uint64_to_double(__m128i &x)
{
    x = _mm_or_si128(x, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)));
    return _mm_sub_pd(_mm_castsi128_pd(x), _mm_set1_pd(0x0010000000000000));
}

inline __m128d my::vector_avx::int64_to_double(__m128i &x)
{
    x = _mm_add_epi64(x, _mm_castpd_si128(_mm_set1_pd(0x0018000000000000)));
    return _mm_sub_pd(_mm_castsi128_pd(x), _mm_set1_pd(0x0018000000000000));
}

inline __m128d my::vector_avx::uint64_to_double_full(__m128i &x)
{
    __m128i xH = _mm_srli_epi64(x, 32);
    xH = _mm_or_si128(xH, _mm_castpd_si128(_mm_set1_pd(19342813113834066795298816.)));
    __m128i xL = _mm_blend_epi16(x, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)), 0xcc);
    __m128d f = _mm_sub_pd(_mm_castsi128_pd(xH), _mm_set1_pd(19342813118337666422669312.));
    return _mm_add_pd(f, _mm_castsi128_pd(xL));
}

inline __m128d my::vector_avx::int64_to_double_full(__m128i &x)
{
    __m128i xH = _mm_srai_epi32(x, 16);
    xH = _mm_blend_epi16(xH, _mm_setzero_si128(), 0x33);
    xH = _mm_add_epi64(xH, _mm_castpd_si128(_mm_set1_pd(442721857769029238784.)));
    __m128i xL = _mm_blend_epi16(x, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)), 0x88);
    __m128d f = _mm_sub_pd(_mm_castsi128_pd(xH), _mm_set1_pd(442726361368656609280.));
    return _mm_add_pd(f, _mm_castsi128_pd(xL));
}

#    if (__AVX2__ || __AVX__)
//__attribute__((optimize("no-fast-math")))
inline __m256d my::vector_avx::int64_to_double_fast_precise_no_FM(const __m256i &v)
/* Optimized full range int64_t to double conversion           */
/* Emulate _mm256_cvtepi64_pd()                                */
{
    __m256i magic_i_lo = _mm256_set1_epi64x(0x4330000000000000);   /* 2^52               encoded as floating-point  */
    __m256i magic_i_hi32 = _mm256_set1_epi64x(0x4530000080000000); /* 2^84 + 2^63        encoded as floating-point  */
    __m256i magic_i_all = _mm256_set1_epi64x(0x4530000080100000);  /* 2^84 + 2^63 + 2^52 encoded as floating-point  */
    __m256d magic_d_all = _mm256_castsi256_pd(magic_i_all);

    __m256i v_lo = _mm256_blend_epi32(magic_i_lo, v, 0b01010101); /* Blend the 32 lowest significant bits of v with magic_int_lo */
    __m256i v_hi = _mm256_srli_epi64(v, 32); /* Extract the 32 most significant bits of v                                                                     */
    v_hi = _mm256_xor_si256(v_hi, magic_i_hi32);                              /* Flip the msb of v_hi and blend with 0x45300000                              */
    __m256d v_hi_dbl = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all); /* Compute in double precision: */
    __m256d result
        = _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo)); /* (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !! */
    return result; /* With gcc use -O3, then -fno-associative-math is default. Do not use -Ofast, which enables -fassociative-math! */
                   /* With icc use -fp-model precise                                                                                */
}

inline __m256d my::vector_avx::int64_to_double_fast_precise(const __m256i &v)
{
    __m256i magic_i_lo = _mm256_set1_epi64x(0x4330000000000000);   /* 2^52        encoded as floating-point  */
    __m256i magic_i_hi32 = _mm256_set1_epi64x(0x4530000000000000); /* 2^84        encoded as floating-point  */
    __m256i magic_i_all = _mm256_set1_epi64x(0x4530000000100000);  /* 2^84 + 2^52 encoded as floating-point  */
    __m256d magic_d_all = _mm256_castsi256_pd(magic_i_all);

    __m256i v_lo = _mm256_blend_epi32(magic_i_lo, v, 0b01010101); /* Blend the 32 lowest significant bits of v with magic_int_lo */
    __m256i v_hi = _mm256_srli_epi64(v, 32); /* Extract the 32 most significant bits of v                                                                     */
    v_hi = _mm256_xor_si256(v_hi, magic_i_hi32);                              /* Blend v_hi with 0x45300000                              */
    __m256d v_hi_dbl = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all); /* Compute in double precision: */
    __m256d result
        = _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo)); /* (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !! */
    return result;
}

inline __m256d my::vector_avx::uint64_to_double256(__m256i &x)
{ /*  Mysticial's fast uint64_to_double. Works for inputs in the range: [0, 2^52)     */
    x = _mm256_or_si256(x, _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)));
    return _mm256_sub_pd(_mm256_castsi256_pd(x), _mm256_set1_pd(0x0010000000000000));
}

inline __m256d my::vector_avx::int64_to_double256(__m256i &x)
{ /*  Mysticial's fast int64_to_double. Works for inputs in the range: (-2^51, 2^51)  */
    x = _mm256_add_epi64(x, _mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000)));
    return _mm256_sub_pd(_mm256_castsi256_pd(x), _mm256_set1_pd(0x0018000000000000));
}

inline __m256d my::vector_avx::int64_to_double_full_range(const __m256i &v)
{
    __m256i msk_lo = _mm256_set1_epi64x(0xFFFFFFFF);
    __m256d cnst2_32_dbl = _mm256_set1_pd(4294967296.0); /* 2^32                                                                    */

    __m256i v_lo = _mm256_and_si256(v, msk_lo);          /* extract the 32 lowest significant bits of v                             */
    __m256i v_hi = _mm256_srli_epi64(v, 32);             /* 32 most significant bits of v. srai_epi64 doesn't exist                 */
    __m256i v_sign = _mm256_srai_epi32(v, 32);           /* broadcast sign bit to the 32 most significant bits                      */
    v_hi = _mm256_blend_epi32(v_hi, v_sign, 0b10101010); /* restore the correct sign of v_hi                                        */
    __m256d v_lo_dbl = int64_to_double256(v_lo);         /* v_lo is within specified range of int64_to_double                       */
    __m256d v_hi_dbl = int64_to_double256(v_hi);         /* v_hi is within specified range of int64_to_double                       */
    v_hi_dbl = _mm256_mul_pd(cnst2_32_dbl, v_hi_dbl);    /* _mm256_mul_pd and _mm256_add_pd may compile to a single fma instruction */
    return _mm256_add_pd(v_hi_dbl, v_lo_dbl);            /* rounding occurs if the integer doesn't exist as a double                */
}

inline __m256d my::vector_avx::int64_to_double_based_on_cvtsi2sd(const __m256i &v)
{
    __m128d zero = _mm_setzero_pd(); /* to avoid uninitialized variables in_mm_cvtsi64_sd                       */
    __m128i v_lo = _mm256_castsi256_si128(v);
    __m128i v_hi = _mm256_extracti128_si256(v, 1);
    __m128d v_0 = _mm_cvtsi64_sd(zero, _mm_cvtsi128_si64(v_lo));
    __m128d v_2 = _mm_cvtsi64_sd(zero, _mm_cvtsi128_si64(v_hi));
    __m128d v_1 = _mm_cvtsi64_sd(zero, _mm_extract_epi64(v_lo, 1));
    __m128d v_3 = _mm_cvtsi64_sd(zero, _mm_extract_epi64(v_hi, 1));
    __m128d v_01 = _mm_unpacklo_pd(v_0, v_1);
    __m128d v_23 = _mm_unpacklo_pd(v_2, v_3);
    __m256d v_dbl = _mm256_castpd128_pd256(v_01);
    v_dbl = _mm256_insertf128_pd(v_dbl, v_23, 1);
    return v_dbl;
}

inline __m256d my::vector_avx::uint64_to_double_full_range(const __m256i &v)
{
    __m256i msk_lo = _mm256_set1_epi64x(0xFFFFFFFF);
    __m256d cnst2_32_dbl = _mm256_set1_pd(4294967296.0); /* 2^32                                                                    */

    __m256i v_lo = _mm256_and_si256(v, msk_lo);   /* extract the 32 lowest significant bits of v                             */
    __m256i v_hi = _mm256_srli_epi64(v, 32);      /* 32 most significant bits of v                                           */
    __m256d v_lo_dbl = uint64_to_double256(v_lo); /* v_lo is within specified range of uint64_to_double                      */
    __m256d v_hi_dbl = uint64_to_double256(v_hi); /* v_hi is within specified range of uint64_to_double                      */
    v_hi_dbl = _mm256_mul_pd(cnst2_32_dbl, v_hi_dbl);
    return _mm256_add_pd(v_hi_dbl, v_lo_dbl); /* rounding may occur for inputs >2^52                                     */
}
#    endif

__m128i my::vector_avx::_mm_shuffle_epi16(__m128i &_A, int _Imm)
{
    // _MM_SHUFFLE8(0, 1, 2, 3, 4, 5, 6, 7)
    _Imm &= 0xffffff;
    char m01 = (_Imm >> 0) & 0x7, m03 = (_Imm >> 3) & 0x7;
    char m05 = (_Imm >> 6) & 0x7, m07 = (_Imm >> 9) & 0x7;
    char m09 = (_Imm >> 12) & 0x7, m11 = (_Imm >> 15) & 0x7;
    char m13 = (_Imm >> 18) & 0x7, m15 = (_Imm >> 21) & 0x7;
    m01 <<= 1;
    m03 <<= 1;
    m05 <<= 1;
    m07 <<= 1;
    m09 <<= 1;
    m11 <<= 1;
    m13 <<= 1;
    m15 <<= 1;
    char m00 = m01 + 1, m02 = m03 + 1, m04 = m05 + 1, m06 = m07 + 1;
    char m08 = m09 + 1, m10 = m11 + 1, m12 = m13 + 1, m14 = m15 + 1;

    //__m128i vMask = _mm_set_epi8(m00, m01, m02, m03, m04, m05, m06, m07,
    //  m08, m09, m10, m11, m12, m13, m14, m15);
    //__m128i vMask = _mm_set_epi8(m14, m15, m12, m13, m10, m11, m08, m09,
    //    m06, m07, m04, m05, m02, m03, m00, m01);
    __m128i vMask = _mm_set_epi8(m00, m01, m02, m03, m04, m05, m06, m07, m08, m09, m10, m11, m12, m13, m14, m15);
    return _mm_shuffle_epi8(_A, vMask);
}
__m128i my::vector_avx::vperm(__m128i &a, __m128i &idx)
{
    //__m128i idx = _mm_set_epi32(0, 1, 2, 3)
    idx = _mm_and_si128(idx, _mm_set1_epi32(0x00000003));
    idx = _mm_mullo_epi32(idx, _mm_set1_epi32(0x04040404));
    idx = _mm_or_si128(idx, _mm_set1_epi32(0x03020100));
    return _mm_shuffle_epi8(a, idx);
}

#    if (__AVX2__ || __AVX__)
/*
int my::vector_avx::find_max_avx(const int32_t *array, size_t n)
{
    __m256i vresult = _mm256_set1_epi32(0);
    __m256i v;

    // Find max value in array 8 by 8
    for (size_t k = 0; k < n; k += 8) {
        v = _mm256_load_si256((__m256i *)&array[k]);
        vresult = _mm256_max_epi32(vresult, v);
    }

    v = _mm256_permute2x128_si256(vresult, vresult, 1);
    vresult = _mm256_max_epi32(vresult, v);
    v = _mm256_permute4x64_epi64(vresult, 1);
    vresult = _mm256_max_epi32(vresult, v);
    v = _mm256_shuffle_epi32(vresult, 1);
    vresult = _mm256_max_epi32(vresult, v);
    __m128i vres128 = _mm256_extracti128_si256(vresult, 0);
    return _mm_extract_epi32(vres128, 0);
}*/

int my::vector_avx::find_max_avx(const int32_t *array, size_t n)
{
    __m256i vresult = _mm256_set1_epi32(0);
    __m256i v;

    // Find max value in array 8 by 8
    for (size_t k = 0; k < n; k += 8) {
        v = _mm256_load_si256((__m256i *)&array[k]);
        vresult = _mm256_max_epi32(vresult, v);
    }

    return horizontal_max_Vec8i(vresult);
}

#    endif

#    if (__AVX2__ || __AVX__)
__m256i my::vector_avx::_mm256_div_epi16(const __m256i &va, const int b)
{
    __m256i &&vb = _mm256_set1_epi16(32768 / b);
    return _mm256_mulhrs_epi16(va, vb);
}
#    endif
/*
int my::vector_avx::horizontal_max_Vec4i(__m128i &x)
{
    int result[4] __attribute__((aligned(16))) = {0};
    _mm_store_si128((__m128i *)result, x);
    return std::max(std::max(std::max(result[0], result[1]), result[2]), result[3]);
}
*/
int my::vector_avx::horizontal_max_Vec8i(__m256i &x)
{
    int result[8] __attribute__((aligned(32))) = {0};
    _mm256_store_si256((__m256i *)result, x);
    return std::max(
        std::max(std::max(std::max(std::max(std::max(std::max(result[0], result[1]), result[2]), result[3]), result[4]), result[5]), result[6]), result[7]);
}


int my::vector_avx::horizontal_max_Vec4i(__m128i &x) {
    __m128i max1 = _mm_shuffle_epi32(x, _MM_SHUFFLE(0,0,3,2));
    __m128i max2 = _mm_max_epi32(x,max1);
    __m128i max3 = _mm_shuffle_epi32(max2, _MM_SHUFFLE(0,0,0,1));
    __m128i max4 = _mm_max_epi32(max2,max3);
    return _mm_cvtsi128_si32(max4);
}
/*
int my::vector_avx::find_max_sse(const int32_t *array, size_t n)
{
    __m128i vresult = _mm_set1_epi32(0);
    __m128i v;
    // Find max value in array 4 by 4
    for (size_t k = 0; k < n; k += 4) {
        v = _mm_load_si128((__m128i *)&array[k]);
        vresult = _mm_max_epi32(vresult, v);
    }

    return horizontal_max_Vec4i(vresult);
}
*/

int my::vector_avx::find_max_sse(const int32_t *array, size_t n)
{
    __m128i vresult = _mm_set1_epi32(0);
    __m128i v;
    // Find max value in array 4 by 4
    for (size_t k = 0; k < n; k += 4) {
        v = _mm_load_si128((__m128i *)&array[k]);
        vresult = _mm_max_epi32(vresult, v);
    }


    v = _mm_shuffle_epi32(vresult, 1);
    vresult = _mm_max_epi32(vresult, v);
    __m128i idx = _mm_set_epi32(0, 1, 2, 3);
    v = vperm(vresult, idx);

    vresult = _mm_max_epi32(vresult, v);
    v = _mm_shuffle_epi16(vresult, _MM_SHUFFLE8(0, 1, 2, 3, 4, 5, 6, 7));
    vresult = _mm_max_epi32(vresult, v);

    __int64_t vres64 = _mm_extract_epi64(vresult, 0);

    __m64 v64 = _mm_set_pi64x(vres64);

    return _mm_extract_pi16(v64, 0);
}

int my::vector_avx::find_max_normal(const int32_t *array, size_t n)
{
    int max = 0;
    int tempMax;
    for (size_t k = 0; k < n; k++) {
        tempMax = array[k];
        if (max < tempMax) {
            max = tempMax;
        }
    }
    return max;
}

#    pragma GCC pop_options
#endif

//__AVX512CD__ __AVX512BW__ __AVX512DQ__ __AVX512VL__
#ifdef __AVX512F__
#    if (__AVX512F__)
int my::vector_avx::find_max_avx512(const int32_t *array, size_t n)
{
    __m512i vresult = _mm512_set1_epi32(0);
    __m512i v;

    // Find max value in array 16 by 16
    for (size_t k = 0; k < n; k += 16) {
        v = _mm512_load_epi32((__m512i *)&array[k]);
        vresult = _mm512_max_epi32(vresult, v);
    }
    /*
    v = _mm256_permute2x128_si256(vresult, vresult, 1);
    vresult = _mm512_max_epi32(vresult, v);
    v = _mm256_permute4x64_epi64(vresult, 1);
    vresult = _mm512_max_epi32(vresult, v);
    v = _mm256_shuffle_epi32(vresult, 1);
    vresult = _mm512_max_epi32(vresult, v);
    __m128i vres128 = _mm256_extracti128_si256(vresult, 0);
    return _mm_extract_epi32(vres128, 0);
    */
    return 0;
}
#    endif
#endif