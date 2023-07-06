#include <stdio.h>

__global__ void hello_gpu()
{
    int tid = threadIdx.x;
    printf("Hello World from %u on GPU!\n", tid);
}
int main()
{

    printf("Hello World from CPU!\n");
    hello_gpu<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}