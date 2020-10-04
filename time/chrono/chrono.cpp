/*
** BENSUPERPC PROJECT, 2020
** RPG
** File description:
** chrono.cpp
*/

#include "chrono.hpp"

my_chrono::my_chrono()
{
}

long int my_chrono::elapsed_ns()
{
    return (stop_time - start_time).count();
}

long int my_chrono::elapsed_ms()
{
    return elapsed_ns() / 1000;
}

void my_chrono::reset()
{
    start_time = Clock::now();
}

void my_chrono::start()
{
    reset();
}

void my_chrono::stop()
{
    stop_time = Clock::now();
}

void my_chrono::add_step()
{

    step.emplace_back(Clock::now());
}

std::vector<size_t> my_chrono::get_steps()
{
    std::vector<size_t> converted;
    converted.reserve(step.size());
    for (size_t i = 0; i < step.size(); i++) {
        converted.emplace_back((stop_time - start_time).count());
    }
    return converted;
}

my_chrono::~my_chrono()
{
}

Clock::time_point my::chrono::now()
{
    return Clock::now();
}

std::chrono::duration<long double> my::chrono::duration(Clock::time_point &t1, Clock::time_point &t2)
{
    return std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
}
