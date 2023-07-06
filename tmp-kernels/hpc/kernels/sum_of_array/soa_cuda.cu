#include <iostream>
#include <vector>
#include <numeric>
#include <timer.hpp>

__global__ void sumVector(const double *v_in, double *v_out, int nelem)
{
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_idx < nelem)
    {
        v_out[thread_idx] = v_in[thread_idx] + 1.0;
    }
}

constexpr int nelem = 10000000;

int main()
{

    std::vector<double> h_v(nelem, 1.0);
    std::vector<double> h_w(nelem, 0.0);

    // allocate device memory
    double *d_v;
    double *d_w;
    cudaMalloc((void **)&d_v, nelem * sizeof(double));
    cudaMalloc((void **)&d_w, nelem * sizeof(double));

    // copy to device
    cudaMemcpy(d_v, h_v.data(), nelem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), nelem * sizeof(double), cudaMemcpyHostToDevice);

    // call kernel on device
    int nThreadsPerBlock = 256;
    int nBlocks = (nelem + nThreadsPerBlock - 1) / nThreadsPerBlock;

    sumVector<<<nBlocks, nThreadsPerBlock>>>(d_v, d_w, nelem);
    cudaDeviceSynchronize();

    // copy to host
    cudaMemcpy(h_w.data(), d_w, nelem * sizeof(double), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_v);
    cudaFree(d_w);

    // check
    double sum = std::accumulate(h_w.begin(), h_w.end(), 0.0);
    std::cout << "h_w[i] = " << sum / nelem << std::endl;

    return 0;
}