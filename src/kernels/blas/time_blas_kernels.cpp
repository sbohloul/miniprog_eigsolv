#include "time_blas_kernels.hpp"

double time_blas_ddot(int niter, std::vector<double> const &x, std::vector<double> const &y)
{
    assert(x.size() == y.size());

    const int nelem = x.size();
    const int incx = 1;
    const int incy = 1;

    Timer timer;
    double d = 0.0;

    timer.start();
    for (int iter = 0; iter < niter; iter++)
    {
        d = cblas_ddot(nelem, x.data(), incx, y.data(), incy);
    }
    timer.stop();

    double t_kernel = timer.duration() / niter;

    return t_kernel;
}