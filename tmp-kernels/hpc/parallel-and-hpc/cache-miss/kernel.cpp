#include <iostream>
#include <stdlib.h>
#include <timer.h>
#include <likwid.h>

using namespace std;

int main(int argc, char *argv[])
{
    LIKWID_MARKER_INIT;
    LIKWID_MARKER_REGISTER("ARRAY1D");

    // initialize some variables
    int niter = 1000;
    int imax = 2002;
    int lmax = imax * imax * 10;

    cout << "imax = " << imax << endl;
    cout << "lmax = " << lmax << endl;
    cout << "niter = " << niter << endl;

    // initialize timing variables
    struct timespec tstart_cpu,
        tstop_cpu;
    double cpu_time = 0.0;

    // allocate and initialize the array
    double *x = (double *)malloc(imax * sizeof(double));
    for (int i = 0; i < imax; i++)
    {
        x[i] = 1.0;
    }

    // to flush the cache
    double *flush = (double *)malloc(lmax * sizeof(double));

    for (int iter = 0; iter < niter; iter++)
    {
        // flushing cache
        for (int l = 0; l < lmax; l++)
        {
            flush[l] = 2.0;
        }

        // start the timer and likwid
        cpu_timer_start(&tstart_cpu);
        LIKWID_MARKER_START("ARRAY1D");

        // computation
        for (int i = 1; i < imax - 1; i++)
        {
            x[i] = (x[i] + x[i - 1] + x[i + 1] - x[i]) / 3.0;
        }

        // stop the timer and collect
        LIKWID_MARKER_STOP("ARRAY1D");
        cpu_time += cpu_timer_stop(tstart_cpu);

        if (iter % 10 == 0)
        {
            cout << "iter = " << iter << endl;
        }
    }
    cout << "elapsed = " << cpu_time << endl;

    free(x);
    free(flush);
    LIKWID_MARKER_CLOSE;

    return 0;
}