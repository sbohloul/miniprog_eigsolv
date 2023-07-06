#include <timer.h>

void cpu_timer_start(struct timespec *tstart_cpu)
{
    clock_gettime(CLOCK_MONOTONIC, tstart_cpu);
}
double cpu_timer_stop(struct timespec tstart_cpu)
{
    struct timespec tstop_cpu, telapsed_cpu;
    clock_gettime(CLOCK_MONOTONIC, &tstop_cpu);
    telapsed_cpu.tv_sec = tstop_cpu.tv_sec - tstart_cpu.tv_sec;
    telapsed_cpu.tv_nsec = tstop_cpu.tv_nsec - tstart_cpu.tv_nsec;
    double elapsed = telapsed_cpu.tv_sec + telapsed_cpu.tv_nsec * 1.0e-9;
    return elapsed;
}
