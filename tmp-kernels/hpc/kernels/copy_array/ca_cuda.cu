#include <iostream>
#include <vector>
#include <numeric>
#include <timer.hpp>

__global__ void vectorCopyKernel(const double *vin, double *vout, int nelem)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < nelem)
    {
        vout[tid] = vin[tid];
    }
}

constexpr int nelem{8 * 1000000};

int main()
{
    std::vector<double> host_vin(nelem, 1.0);
    std::vector<double> host_vout(nelem, 0.0);
    size_t size = nelem * sizeof(double);

    // allocate device memory
    double *device_vin;
    double *device_vout;
    cudaMalloc((void **)&device_vin, size);
    cudaMalloc((void **)&device_vout, size);

    // copy to device
    cudaMemcpy(device_vin, host_vin.data(), size, cudaMemcpyHostToDevice);

    // call kernel
    Timer t1;
    int nThreadsPerBlock = 256;
    int nBlocks = (nelem + nThreadsPerBlock - 1) / nThreadsPerBlock;
    t1.start();
    vectorCopyKernel<<<nBlocks, nThreadsPerBlock>>>(device_vin, device_vout, nelem);
    cudaDeviceSynchronize();
    t1.stop();

    Timer t2;
    nThreadsPerBlock = 32;
    nBlocks = (nelem + nThreadsPerBlock - 1) / nThreadsPerBlock;
    t2.start();
    vectorCopyKernel<<<nBlocks, nThreadsPerBlock>>>(device_vin, device_vout, nelem);
    cudaDeviceSynchronize();
    t2.stop();

    // copy to device
    cudaMemcpy(host_vout.data(), device_vout, size, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(device_vin);
    cudaFree(device_vout);

    // check
    std::cout << "vout[i] = ";
    std::cout << std::accumulate(host_vout.begin(), host_vout.end(), 0.0) / nelem << std::endl;

    // timing and membandwidth
    double bandwidth1 = (size * 2) / t1.duration() * 1e-9;
    double bandwidth2 = (size * 2) / t2.duration() * 1e-9;

    std::cout << "t1 = " << t1.duration() << std::endl;
    std::cout << "t2 = " << t2.duration() << std::endl;
    std::cout << "b1 = " << bandwidth1 << " GB/s" << std::endl;
    std::cout << "b2 = " << bandwidth2 << " GB/s" << std::endl;

    return 0;
}