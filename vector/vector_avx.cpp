/*
** BENSUPERPC PROJECT, 2020
** CPU
** File description:
** vector.cpp
*/

#include "vector_avx.hpp"

#if (__AVX2__ || __AVX__ || __SSE3__)
//__SSE__ __SSE2__

#    if (__AVX2__ || __AVX__)
#        pragma GCC push_options
#        pragma GCC optimize("-O1")
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
#        pragma GCC pop_options
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
__attribute__((optimize("no-fast-math"))) inline __m256d my::vector_avx::int64_to_double_fast_precise_no_FM(const __m256i &v)
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

inline __m256d uint64_to_double256(__m256i x)
{ /*  Mysticial's fast uint64_to_double. Works for inputs in the range: [0, 2^52)     */
    x = _mm256_or_si256(x, _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)));
    return _mm256_sub_pd(_mm256_castsi256_pd(x), _mm256_set1_pd(0x0010000000000000));
}

inline __m256d int64_to_double256(__m256i x)
{ /*  Mysticial's fast int64_to_double. Works for inputs in the range: (-2^51, 2^51)  */
    x = _mm256_add_epi64(x, _mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000)));
    return _mm256_sub_pd(_mm256_castsi256_pd(x), _mm256_set1_pd(0x0018000000000000));
}

inline __m256d int64_to_double_full_range(const __m256i &v)
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

inline __m256d int64_to_double_based_on_cvtsi2sd(const __m256i &v)
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

inline __m256d uint64_to_double_full_range(const __m256i &v)
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
#endif
#endif