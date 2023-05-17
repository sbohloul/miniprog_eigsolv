#include <iostream>
#include <vector>
#include <blas.h>
#include <parameters.h>

int main(int argc, char **argv)
{
    std::cout << "ddot for dot product of vectors x and y" << std::endl;

    // parameters
    const int n = VECTOR_SIZE;

    std::cout << "n = " << n << std::endl;

    // init vectors
    std::vector<double> x(n);
    std::vector<double> y(n);

    for (std::vector<double>::size_type i = 0; i < x.size(); i++)
    {
        x[i] = static_cast<double>(ONE_INT);
    }
    for (std::vector<double>::size_type i = 0; i < y.size(); i++)
    {
        y[i] = static_cast<double>(TWO_INT);
    }

    // print vector x
    std::cout << "x = ";
    for (const auto &element : x)
    {
        std::cout << element << " ";
    }
    std::cout << std::endl;

    // print vector y
    std::cout << "y = ";
    for (const auto &element : y)
    {
        std::cout << element << " ";
    }
    std::cout << std::endl;

    // call blas
    int incx = 1;
    int incy = 1;
    double d = cblas_ddot(n, &x[0], incx, &y[0], incy);

    // print <x|y>
    std::cout << "d = " << d << std::endl;

    return 0;
}