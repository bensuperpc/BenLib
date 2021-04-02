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
    /**
     * @brief 
     * 
     */
    void reset();

    /**
     * @brief 
     * 
     */
    void start();
    void stop();
    void add_step();
    long int elapsed_ns();
    long int elapsed_ms();
    std::vector<size_t> get_steps();

    // void generate_leafs();

    /**
     * @brief Construct a new my chrono object
     * 
     */
    my_chrono();

    /**
     * @brief Destroy the my chrono object
     * 
     */
    ~my_chrono();

  private:
    /**
     * @brief start time
     * 
     */
    Clock::time_point start_time = Clock::now();

    /**
     * @brief stop time
     * 
     */
    Clock::time_point stop_time = Clock::now();

    // For vector step by step
    std::vector<Clock::time_point> step {};

  protected:
};

namespace my
{
namespace chrono
{
/**
 * @brief Get current time
 * 
 * @return Clock::time_point 
 */
Clock::time_point now();
/**
 * @brief Calc duration between two times
 * 
 * @param t1 
 * @param t2 
 * @return std::chrono::duration<long double> 
 */
std::chrono::duration<long double> duration(Clock::time_point &t1, Clock::time_point &t2);
} // namespace chrono
} // namespace my
#endif
