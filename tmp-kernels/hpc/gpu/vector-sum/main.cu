#include <iostream>
#include <vector>
#include <timer.hpp>
#include <cassert>
#include <stdio.h>

#define imin(a, b) (a < b ? a : b)

const int NELEM = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (NELEM + threadsPerBlock - 1) / threadsPerBlock);

__global__ void gpu_vector_sum(double *psum, const double *v, const int n)
{

    // cache for block sum
    __shared__ double bsum[threadsPerBlock];

    // get global and local thread id
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_local = threadIdx.x;

    int stride = blockDim.x * gridDim.x;
    double tsum = 0.0;
    for (int i = tid; i < n; i += stride)
    {
        tsum += v[i];
    }

    bsum[tid_local] = tsum;

    // synch threads
    __syncthreads();

    // collect contribution from threads within the block
    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (tid_local < i)
        {
            bsum[tid_local] += bsum[tid_local + i];
        }
        __syncthreads();
        i /= 2;
    }

    // collect contribution from the blocks which is hold in master thread
    if (tid_local == 0)
    {
        psum[blockIdx.x] = bsum[0];
    }
}

int main()
{
    double init_val = 1.0;
    double result_ref = init_val;

    std::vector<double> v1(NELEM, init_val);
    std::vector<double> psum(blocksPerGrid, 0.0);

    // cpu
    double sum_cpu = 0.0;
    for (size_t i = 0; i < NELEM; i++)
    {
        sum_cpu += v1[i];
    }
    std::cout << "sum_cpu / N = " << sum_cpu / NELEM << std::endl;

    // gpu

    // allocate device mem
    double *d_v1;
    double *d_psum;

    cudaMalloc((void **)&d_v1, NELEM * sizeof(double));
    cudaMalloc((void **)&d_psum, blocksPerGrid * sizeof(double));

    // copy from host to device
    cudaMemcpy(d_v1, v1.data(), NELEM * sizeof(double), cudaMemcpyHostToDevice);

    // call kernel
    gpu_vector_sum<<<blocksPerGrid, threadsPerBlock>>>(d_psum, d_v1, NELEM);

    // copy from device to host
    cudaMemcpy(psum.data(), d_psum, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);

    double sum_gpu = 0.0;
    for (int i = 0; i < psum.size(); i++)
    {
        std::cout << "psum[" << i << "] = " << psum[i] << std::endl;
        sum_gpu += psum[i];
    }
    // print info
    std::cout << "sum_gpu / N = " << sum_gpu / NELEM << std::endl;

    // free mem
    cudaFree(d_v1);
    cudaFree(d_psum);
}