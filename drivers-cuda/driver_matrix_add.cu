#include <iostream>
#include <matrix.hpp>
#include <timer.hpp>

template <typename T>
__global__ void addMatrix(const T *a, const T *b, T *c, int rows, int cols)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < rows)
    {
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        if (j < cols)
        {
            int index = i * cols + j;
            c[index] = a[index] + b[index];
        }
    }
}

int main(int argc, char **argv)
{
    int blockSizeX;
    int blockSizeY;
    int rows;
    int cols;

    if (argc < 5)
    {
        rows = 10;
        cols = 10;
        blockSizeX = 16;
        blockSizeY = 16;
    }
    else
    {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
        blockSizeX = atoi(argv[3]);
        blockSizeY = atoi(argv[4]);
    }
    int numElements = rows * cols;
    // dim3 numBlocks = ((rows + blockSizeX - 1) / blockSizeX, (cols + blockSizeY - 1) / blockSizeY);
    dim3 numBlocks((rows + blockSizeX - 1) / blockSizeX, (cols + blockSizeY - 1) / blockSizeY);
    dim3 threadsPerBlock(blockSizeX, blockSizeY);

    std::cout << numBlocks.x << std::endl;
    std::cout << numBlocks.y << std::endl;
    std::cout << numBlocks.z << std::endl;

    MatrixP<double> h_a(rows, cols);
    MatrixP<double> h_b(rows, cols);
    MatrixP<double> h_c(rows, cols);

    for (int i = 0; i < numElements; i++)
    {
        h_a.data()[i] = static_cast<double>(i);
        h_b.data()[i] = static_cast<double>(i) / numElements;
    }

    // allocate device memory
    double *d_a;
    double *d_b;
    double *d_c;

    cudaMalloc((void **)&d_a, numElements * sizeof(double));
    cudaMalloc((void **)&d_b, numElements * sizeof(double));
    cudaMalloc((void **)&d_c, numElements * sizeof(double));

    // copy data to device
    cudaMemcpy(d_a, h_a.data(), numElements * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), numElements * sizeof(double), cudaMemcpyHostToDevice);

    // call kernel
    Timer timer;

    timer.start();

    addMatrix<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, rows, cols);
    cudaDeviceSynchronize();

    timer.stop();

    std::cout << "duration: " << timer.duration() << std::endl;

    // copy data to host
    cudaMemcpy(h_c.data(), d_c, numElements * sizeof(double), cudaMemcpyDeviceToHost);

    // release memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // wrap up
    if (numElements <= 100)
    {
        std::cout << h_a;
        std::cout << h_b;
        std::cout << h_c;
    }

    return 0;
}