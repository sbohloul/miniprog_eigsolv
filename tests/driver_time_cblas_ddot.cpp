#include <iostream>
#include "time_kernels.hpp"

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        throw std::runtime_error("wrong inputs: prog.x niter nelem");
    }

    int niter = atoi(argv[1]);
    int nelem = atoi(argv[2]);

    std::vector<double> x(nelem, 1.0);
    std::vector<double> y(nelem, 2.0);

    double t = time_cblas_ddot(niter, x, y);

    std::cout << "t = " << t << std::endl;
    return 0;
}