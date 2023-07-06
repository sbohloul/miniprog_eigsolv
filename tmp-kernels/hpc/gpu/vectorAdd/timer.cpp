#include <timer.h>

void cpu_timer_start(struct timespec *tstart)
{
    clock_gettime(CLOCK_MONOTONIC, tstart);
}

double cpu_timer_stop(struct timespec tstart)
{
    struct timespec tstop;
    clock_gettime(CLOCK_MONOTONIC, &tstop);
    double telps = (tstop.tv_sec - tstart.tv_sec) + (tstop.tv_nsec - tstart.tv_nsec) * 1.0e-9;

    return telps;
}