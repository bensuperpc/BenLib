/**
 * @file chrono.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

#ifndef _CHRONO_HPP_
#define _CHRONO_HPP_

#include <chrono>
#include <iostream>
#include <vector>

typedef std::chrono::high_resolution_clock Clock;

/**
 * @class my_chrono
 * @brief the my_chrono class, To calculate time between to line of code
 */
class my_chrono {
  public:
    // Fonctions
    void reset();
    void start();
    void stop();
    void add_step();
    long int elapsed_ns();
    long int elapsed_ms();
    std::vector<size_t> get_steps();

    // void generate_leafs();

    // Constructeurs
    my_chrono();

    // Destructeurs
    ~my_chrono();

  private:
    // Variables
    Clock::time_point start_time = Clock::now();
    Clock::time_point stop_time = Clock::now();

    // For vector step by step
    std::vector<Clock::time_point> step {};

  protected:
};

namespace my
{
namespace chrono
{
Clock::time_point now();
/**
 * @brief 
 * 
 * @param t1 
 * @param t2 
 * @return std::chrono::duration<long double> 
 */
std::chrono::duration<long double> duration(Clock::time_point &t1, Clock::time_point &t2);
} // namespace chrono
} // namespace my
#endif
