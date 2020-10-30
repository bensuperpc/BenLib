//#include <boost/multiprecision/cpp_int.hpp>
#include <stdio.h>
#include <stdlib.h>
/*
#ifdef _WIN32
#include <sys/timeb.h>
#else
#include <sys/time.h>
#endif
*/
#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <time.h>
#include <vector>
#include "../../lib/time/chrono/chrono.hpp"
#include "../../lib/vector/vector.hpp"

#define NBRS 10000000

template <typename Type> Type divide(Type a, Type b)
{
    double da = (double)a;
    double db = (double)b;
    double q = da / db;
    return (Type)q;
}

/*
double
mygettime(void) {
# ifdef _WIN32
  struct _timeb tb;
  _ftime(&tb);
  return (double)tb.time + (0.001 * (double)tb.millitm);
# else
  struct timeval tv;
  if(gettimeofday(&tv, 0) < 0) {
    perror("oops");
  }
  return (double)tv.tv_sec + (0.000001 * (double)tv.tv_usec);
# endif
}
*/

template <typename Type> void my_test(const char *name)
{
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};
    std::vector<Type> &&v = std::vector<Type>(NBRS, (Type)5);
    std::vector<Type> &&t = std::vector<Type>(NBRS, (Type)7);
    my::vector::rnd_fill<Type>(t);
    my::vector::rnd_fill<Type>(v);
    std::shuffle(begin(t), end(t), mersenne_engine);
    std::shuffle(begin(v), end(v), mersenne_engine);
    auto &&t1 = my::chrono::now();
    for (size_t i = 0; i < NBRS; ++i) {
        v[i] += t[i];
    }
    auto &&t2 = my::chrono::now();
    std::cout << name << " add: " << (((double)NBRS / my::chrono::duration(t1, t2).count()) * 1.0f) / 1000000.0f << " MIPS" << std::endl;

    t1 = my::chrono::now();
    for (size_t i = 0; i < NBRS; ++i) {
        v[i] -= t[i];
    }
    t2 = my::chrono::now();
    // typeid(Type).name()
    std::cout << name << " sub: " << (((double)NBRS / my::chrono::duration(t1, t2).count()) * 1.0f) / 1000000.0f << " MIPS" << std::endl;

    t1 = my::chrono::now();
    for (size_t i = 0; i < NBRS; ++i) {
        v[i] *= t[i];
    }
    t2 = my::chrono::now();
    std::cout << name << " mul: " << (((double)NBRS / my::chrono::duration(t1, t2).count()) * 1.0f) / 1000000.0f << " MIPS" << std::endl;

    t1 = my::chrono::now();
    for (size_t i = 0; i < NBRS; ++i) {
        v[i] /= t[i];
    }
    t2 = my::chrono::now();
    std::cout << name << " div: " << (((double)NBRS / my::chrono::duration(t1, t2).count()) * 1.0f) / 1000000.0f << " MIPS" << std::endl;
    if constexpr (std::is_integral<Type>::value) {
        t1 = my::chrono::now();
        for (size_t i = 0; i < NBRS; ++i) {
            v[i] %= t[i];
        }
        t2 = my::chrono::now();
        std::cout << name << " mod: " << (((double)NBRS / my::chrono::duration(t1, t2).count()) * 1.0f) / 1000000.0f << " MIPS" << std::endl;
    }
}

int main()
{
    // using namespace boost::multiprecision;
    /*
    my_test<      short >("      short");
    my_test<        int >("        int");
    my_test<       long >("       long");
    my_test<  long long >("  long long");*/
    my_test<int8_t>("      int8_t");
    my_test<int16_t>("     int16_t");
    my_test<int32_t>("     int32_t");
    my_test<int64_t>("     int64_t");

    // my_test<int128_t>("    int128_t");
    // my_test<int256_t>("    int256_t");
    // my_test<int512_t>("    int512_t");

    my_test<float>("      float");
    my_test<double>("     double");
    my_test<long double>("long double");
    return 0;
}
