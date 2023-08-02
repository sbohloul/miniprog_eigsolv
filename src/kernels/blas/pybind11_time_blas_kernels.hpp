#ifndef _PYBIND11_TIME_BLAS_KERNELS_HPP_
#define _PYBIND11_TIME_BLAS_KERNELS_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <time_blas_kernels.hpp>

double pytime_cblas_ddot(int niter, pybind11::array_t<double> x_pyarr, pybind11::array_t<double> y_pyarr);

#endif // _PYBIND11_TIME_BLAS_KERNELS_HPP_