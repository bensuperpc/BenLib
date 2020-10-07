/*
** BENSUPERPC PROJECT, 2020
** RPG
** File description:
** chrono.cpp
*/

#ifndef _CHRONO_HPP_
#define _CHRONO_HPP_

#include <chrono>
#include <iostream>
#include <vector>

typedef std::chrono::high_resolution_clock Clock;

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
std::chrono::duration<long double> duration(Clock::time_point &, Clock::time_point &);
} // namespace chrono
} // namespace my
#endif
