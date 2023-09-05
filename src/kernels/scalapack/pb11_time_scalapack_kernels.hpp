#ifndef _PB11_TIME_SCALAPACK_KERNELS_
#define _PB11_TIME_SCALAPACK_KERNELS_

#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mpi4py/mpi4py.h>
#include <mpi.h>
#include "time_scalapack_kernels.hpp"
#include "blacs.h"
#include "array_helper.hpp"

namespace py = pybind11;

double pb11_time_scalapack_pdgemm(int niter,
                                  const int nprow, const int npcol,
                                  py::array_t<double> a,
                                  py::array_t<double> b,
                                  py::array_t<double> c,
                                  int m, int n,
                                  int mb, int nb);

double pb11_time_scalapack_pdsyev(int niter,
                                  const int nprow, const int npcol,
                                  py::array_t<double> a_pyarr,
                                  py::array_t<double> eigval_pyarr,
                                  py::array_t<double> eigvec_pyarr,
                                  int m,
                                  int mb, int nb);

#endif