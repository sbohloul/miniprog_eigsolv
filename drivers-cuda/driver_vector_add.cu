#include <iostream>
#include <matrix.hpp>

template <typename T>
__global__ void vectorAdd(const T *a, const T *b, T *c, int numElemetns)
{
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_idx < numElemetns)
    {
        c[thread_idx] = a[thread_idx] + b[thread_idx];
    }
}

int main(int argc, char **argv)
{

    std::cout << "=======================" << std::endl;
    std::cout << "Cuda vector add example" << std::endl;
    std::cout << "=======================" << std::endl;

    int numElements;
    int numThreadsInBlock;

    if (argc < 3)
    {
        numElements = 1000;
        numThreadsInBlock = 256;
    }
    else
    {
        numElements = atoi(argv[1]);
        numThreadsInBlock = atoi(argv[2]);
    }
    int numBlocks = (numElements + numThreadsInBlock - 1) / numThreadsInBlock;

    VectorV<double> h_a(numElements);
    VectorV<double> h_b(numElements);
    VectorV<double> h_c(numElements);

    for (int i = 0; i < numElements; i++)
    {
        h_a(i) = 1.0;
        h_b(i) = 2.0;
    }

    // host memory pointers

    // allocate device memory
    double *d_a;
    double *d_b;
    double *d_c;

    cudaMalloc((void **)&d_a, numElements * sizeof(double));
    cudaMalloc((void **)&d_b, numElements * sizeof(double));
    cudaMalloc((void **)&d_c, numElements * sizeof(double));

    // copy to device
    cudaMemcpy(d_a, h_a.data(), numElements * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), numElements * sizeof(double), cudaMemcpyHostToDevice);

    // call kernel

        vectorAdd<<<numBlocks, numThreadsInBlock>>>(d_a, d_b, d_c, numElements);
    cudaDeviceSynchronize();
    // copy to host
    cudaMemcpy(h_c.data(), d_c, numElements * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // wrap up
    double err = 0.0;
    for (size_t i = 0; i < numElements; i++)
    {
        err += h_a(i) + h_b(i) - h_c(i);
    }
    std::cout << "err " << err << std::endl;

    return 0;
}