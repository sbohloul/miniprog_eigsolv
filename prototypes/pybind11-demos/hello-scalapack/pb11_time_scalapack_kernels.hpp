#ifndef _PB11_TIME_SCALAPACK_KERNELS_
#define _PB11_TIME_SCALAPACK_KERNELS_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mpi4py/mpi4py.h>
#include <mpi.h>
#include "time_scalapack_kernels.hpp"
#include "blacs_.h"
#include "array_helper.hpp"

namespace py = pybind11;

double pb11_time_scalapack_pdgemm(int niter,
                                  const int nprow, const int npcol,
                                  py::array_t<double> a,
                                  py::array_t<double> b,
                                  py::array_t<double> c,
                                  int m, int n,
                                  int mb, int nb);

#endif