#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include <chrono>

class Timer
{
public:
    Timer() : startTime(), stopTime(){};

    void start();
    void stop();
    double duration() const;

private:
    typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_t;
    time_t startTime;
    time_t stopTime;
};

void Timer::start() { startTime = std::chrono::high_resolution_clock::now(); }

void Timer::stop() { stopTime = std::chrono::high_resolution_clock::now(); }

double Timer::duration() const
{
    auto elapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(stopTime - startTime);
    return elapsedTime.count() * 1.0e-9;
}

#endif /*__TIMER_HPP__*/