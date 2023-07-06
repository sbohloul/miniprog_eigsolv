#include <stdio.h>
#include <iostream>

__global__ void print_gpu()
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("tid = %d\n", tid);
}

int main()
{
    print_gpu<<<2, 256>>>();
    cudaDeviceSynchronize();
}