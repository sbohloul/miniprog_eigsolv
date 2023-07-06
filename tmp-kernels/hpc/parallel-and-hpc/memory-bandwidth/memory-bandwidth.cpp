#include <iostream>
#include <time.h>
#include <math.h>

using namespace std;

int main(int argc, char *argv[])
{
    int P = 27;
    int M, N;
    int p1, p2;

    struct timespec tstart_cpu;
    struct timespec tstop_cpu;
    struct timespec telapsed_cpu;

    long FLOPS = pow(2, P);
    cout << "flops,"
         << "p1,"
         << "p2,"
         << "elapsed time" << endl;

    clock_gettime(CLOCK_MONOTONIC, &tstart_cpu);
    for (int p = 0; p < P; p++)
    {
        int p1 = p;
        int p2 = P - p;
        int N = pow(2, p1);
        int M = pow(2, p2);
        double *a = (double *)malloc(N * sizeof(double));

        clock_gettime(CLOCK_MONOTONIC, &tstart_cpu);
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                a[n] = a[n] + 1;
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &tstop_cpu);

        telapsed_cpu.tv_sec = tstop_cpu.tv_sec - tstart_cpu.tv_sec;
        telapsed_cpu.tv_nsec = tstop_cpu.tv_nsec - tstart_cpu.tv_nsec;
        double elapsed = telapsed_cpu.tv_sec + telapsed_cpu.tv_nsec * 1.0e-9;
        cout << FLOPS << "," << p1 << "," << p2 << "," << elapsed << endl;

        free(a);
    }

    return 0;
}