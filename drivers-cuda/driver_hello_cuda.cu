#include <iostream>

__global__ void helloCuda()
{
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from threadIdx %d on GPU\n", thread_idx);
}

int main(int argc, char *argv[])
{
    std::cout << "========================" << std::endl;
    std::cout << "Cuda hello world example" << std::endl;
    std::cout << "========================" << std::endl;

    int numBlocks;
    int numThreadsInBlock;

    if (argc < 3)
    {
        std::cout << "Using default values" << std::endl;
        numBlocks = 1;
        numThreadsInBlock = 32;
    }
    else
    {
        numBlocks = std::atoi(argv[1]);
        numThreadsInBlock = std::atoi(argv[2]);
    }

    std::cout << "numBlocks " << numBlocks << std::endl;
    std::cout << "numThreadsInBlock " << numThreadsInBlock << std::endl;

    std::cout << "Hello from CPU!" << std::endl;

    helloCuda<<<numBlocks, numThreadsInBlock>>>();
    cudaDeviceSynchronize();

    return 0;
}