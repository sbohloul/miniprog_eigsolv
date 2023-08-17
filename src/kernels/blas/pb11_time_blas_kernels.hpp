#ifndef _PB11_TIME_BLAS_KERNELS_HPP_
#define _PB11_TIME_BLAS_KERNELS_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <time_blas_kernels.hpp>

namespace py = pybind11;

double pb11_time_cblas_ddot(int niter, pybind11::array_t<double> x_pyarr, pybind11::array_t<double> y_pyarr);

#endif // _PB11_TIME_BLAS_KERNELS_HPP_