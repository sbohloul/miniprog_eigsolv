#include <iostream>
#include <vector>
#include <blas.h>
#include <parameters.h>

int main(int argc, char **argv)
{

    std::cout << "dcopy for copying x vector to y" << std::endl;

    // parameters
    const int n = VECTOR_SIZE;

    std::cout << "n = " << n << std::endl;

    // init vectors
    std::vector<double> x(n);
    std::vector<double> y(n);

    for (auto &element : x)
    {
        element = (double)(ONE_INT);
    }

    // copy x to y
    int incx = 1;
    int incy = 1;

    cblas_dcopy(n, &x[0], incx, &y[0], incy);

    // print x vectors
    std::vector<double>::iterator iter;

    std::cout << "x: ";
    for (iter = x.begin(); iter < x.end(); iter++)
    {
        std::cout << *iter << " ";
    }
    std::cout << std::endl;

    // print y vector
    std::cout << "y: ";
    for (iter = y.begin(); iter < y.end(); iter++)
    {
        std::cout << *iter << " ";
    }
    std::cout << std::endl;

    return 0;
}