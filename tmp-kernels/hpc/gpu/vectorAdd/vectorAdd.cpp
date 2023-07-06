#include <iostream>
#include <math.h>
#include <timer.h>

using namespace std;

void addVector(double *a, double *b, double *c, int numElements)
{
    for (int i = 0; i < numElements; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char *argv[])
{
    struct timespec tstart_cpu;

    int numElements = 1 << 20; // 1M elements
    size_t bytes = numElements * sizeof(double);

    double *a;
    double *b;
    double *c;

    // allocate host memory
    a = (double *)malloc(bytes);
    b = (double *)malloc(bytes);
    c = (double *)malloc(bytes);

    // init vectors
    for (int i = 0; i < numElements; i++)
    {
        a[i] = 1.0;
        b[i] = 2.0;
    }

    // launch kernel
    cpu_timer_start(&tstart_cpu);
    addVector(a, b, c, numElements);
    double telsp = cpu_timer_stop(tstart_cpu);
    cout << "telps: " << telsp << endl;

    // validate results
    double maxError = 0.0;
    for (int i = 0; i < numElements; i++)
    {
        maxError = fmax(maxError, fabs(c[i] - 3.0));
    }

    cout << "maxError: " << maxError << endl;

    // free host memory
    free(a);
    free(b);
    free(c);

    return 0;
}