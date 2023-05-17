#include <iostream>
#include <vector>
#include <blas.h>
#include <parameters.h>

int main(int argc, char **argv)
{
    std::cout << "dscal for scaling vector x by scalar s" << std::endl;

    // parameters
    const int n = VECTOR_SIZE;

    // int vector
    std::vector<double> x(n);

    for (std::vector<double>::size_type i = 0; i < x.size(); i++)
    {
        x[i] = static_cast<double>(ONE_INT);
    }

    // print vector x
    std::cout << "x: ";
    for (const auto &element : x)
    {
        std::cout << element << " ";
    }
    std::cout << std::endl;

    // call blas
    const int incx = 1;

    cblas_dscal(n, TWO_INT, &x[0], incx);

    // print scaled x
    std::cout << "x scaled: ";
    for (const auto &element : x)
    {
        std::cout << element << " ";
    }
    std::cout << std::endl;

    return 0;
}