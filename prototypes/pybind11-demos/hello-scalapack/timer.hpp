#ifndef _TIMER_HPP_
#define _TIMER_HPP_

#include <chrono>

class Timer
{
public:
    Timer();

    void start();
    void stop();
    double duration() const;

private:
    typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_t;
    time_t startTime;
    time_t stopTime;
};

#endif // _TIMER_HPP_