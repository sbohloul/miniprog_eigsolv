#include <iostream>
#include <math.h>
#include <time.h>

using namespace std;

__global__ void addVector(double *a, double *b, double *c, int numElements)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < numElements)
    {
        c[tid] = a[tid] + b[tid];
    }
}

int main(int argc, char *argv[])
{
    int inputBlockSize = 256;
    if (argc >= 2)
    {
        inputBlockSize = stoi(argv[1]);
    }
    cout << "inputBlockSize: " << inputBlockSize << endl;

    // timing parameters
    struct timespec tstrt_cpu;
    struct timespec tstop_cpu;
    struct timespec telps_cpu;

    // vector parameters
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

    // allocate device memory
    double *d_a;
    double *d_b;
    double *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // copy data to device
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, bytes, cudaMemcpyHostToDevice);

    // launch device kernel
    int blockSize, numBlocks;
    blockSize = inputBlockSize;
    numBlocks = (numElements + blockSize - 1) / blockSize;

    clock_gettime(CLOCK_MONOTONIC, &tstrt_cpu);
    addVector<<<numBlocks, blockSize>>>(d_a, d_b, d_c, numElements);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &tstop_cpu);

    telps_cpu.tv_sec = tstop_cpu.tv_sec - tstrt_cpu.tv_sec;
    telps_cpu.tv_nsec = tstop_cpu.tv_nsec - tstrt_cpu.tv_nsec;
    double telps = telps_cpu.tv_sec + telps_cpu.tv_nsec * 1.0e-9;
    cout << "telsp: " << telps << endl;

    // copy data to host
    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

    // validate results
    double maxError = 0.0;
    for (int i = 0; i < numElements; i++)
    {
        maxError = fmax(maxError, fabs(c[i] - 3.0));
    }
    cout << "maxError: " << maxError << endl;

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // free host memory
    free(a);
    free(b);
    free(c);

    return 0;
}