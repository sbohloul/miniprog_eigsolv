#include <iostream>
#include <vector>
#include <cassert>

#define imin(a, b) (a < b ? a : b)
const int N = 2 * 256;
const int numThreadsPerBlock = 256;
const int numBlocks = imin(32, (N + numThreadsPerBlock - 1) / numThreadsPerBlock);

void cpu_vector_dot(double &sum, const std::vector<double> &v1, const std::vector<double> &v2)
{
    for (size_t i = 0; i < v1.size(); i++)
    {
        sum += v1[i] * v2[i];
    }
}

__global__ void gpu_vector_dot(double *psum, double *v1, double *v2)
{
    __shared__ double bsum[numThreadsPerBlock];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_local = threadIdx.x;

    if (tid < 2)
    {
        printf("numBlocks = %d\n", gridDim.x);
    }

    double tsum = 0.0;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x)
    {
        tsum += v1[i] * v2[i];
    }

    if (tid_local == 0)
    {
        printf("tid_local = %d tsum = %f\n", tid_local, tsum);
    }

    bsum[tid_local] = tsum;

    if (tid_local == 0)
    {
        printf("tid_local = %d bsum = %f\n", tid_local, bsum[tid_local]);
    }

    __syncthreads();

    // reduction
    int i = blockDim.x / 2;
    if (tid_local == 0)
    {
        for (int j = 2 * i; j < blockDim.x; j++)
        {
            bsum[0] += bsum[j];
        }
    }
    __syncthreads();

    while (i != 0)
    {
        if (tid_local < i)
        {
            bsum[tid_local] += bsum[tid_local + i];
        }
        __syncthreads();
        i /= 2;
    }

    // update psum
    if (tid_local == 0)
    {
        psum[blockIdx.x] = bsum[0];
    }
}

int main(int argc, char **argv)
{

    double val1 = 1.0;
    double val2 = 2.0;
    double result_ref = val1 * val2;

    std::vector<double> v1(N, val1);
    std::vector<double> v2(N, val2);

    // ==========
    // cpu kernel
    // ==========
    double cpu_val = 0.0;
    cpu_vector_dot(cpu_val, v1, v2);
    std::cout << "cpu_val / N = " << cpu_val / N << std::endl;
    assert(cpu_val / N == result_ref);

    // ==========
    // gpu kernel
    // ==========
    std::vector<double> psum(numBlocks, 0.0);

    // allocate device mem
    double *d_v1;
    double *d_v2;
    double *d_psum;

    cudaMalloc((void **)&d_v1, N * sizeof(double));
    cudaMalloc((void **)&d_v2, N * sizeof(double));
    cudaMalloc((void **)&d_psum, N * sizeof(numBlocks));

    // copy to device
    cudaMemcpy(d_v1, v1.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2, v2.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // call kernel
    gpu_vector_dot<<<numBlocks, numThreadsPerBlock>>>(d_psum, d_v1, d_v2);

    // copy to host
    cudaMemcpy(psum.data(), d_psum, numBlocks * sizeof(double), cudaMemcpyDeviceToHost);

    // final reduction
    double gpu_val = 0.0;
    for (int i = 0; i < psum.size(); i++)
    {
        std::cout << "psum[" << i << "] = " << psum[i] << std::endl;
        gpu_val += psum[i];
    }
    std::cout << "gpu_val / N = " << gpu_val / N << std::endl;
    assert(gpu_val / N == result_ref);

    // free mem
    cudaFree(d_v1);
    cudaFree(d_v2);
}