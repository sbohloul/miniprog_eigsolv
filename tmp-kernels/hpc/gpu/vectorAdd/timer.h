#ifndef _TIMER_H
#define _TIMER_H
#include <time.h>
void cpu_timer_start(struct timespec *tstart);
double cpu_timer_stop(struct timespec tstart);
#endif
