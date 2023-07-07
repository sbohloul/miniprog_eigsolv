#include "timer.hpp"

Timer::Timer() : startTime(), stopTime(){};

void Timer::start()
{
    startTime = std::chrono::high_resolution_clock::now();
}

void Timer::stop()
{
    stopTime = std::chrono::high_resolution_clock::now();
}

double Timer::duration() const
{
    auto elapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(stopTime - startTime);

    return elapsedTime.count() * 1.0e-9;
}