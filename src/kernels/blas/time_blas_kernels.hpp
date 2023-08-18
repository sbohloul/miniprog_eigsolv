#ifndef _TIME_BLAS_KERNELS_HPP_
#define _TIME_BLAS_KERNELS_HPP_

#include <vector>
#include "blas.h"
#include "timer.hpp"
#include <cassert>

double time_blas_ddot(const int niter, const std::vector<double> &x, const std::vector<double> &y);

#endif // _TIME_BLAS_KERNELS_HPP_