#ifndef _PB11_MPISTATS_SCALAPACK_KERNELS_
#define _PB11_MPISTATS_SCALAPACK_KERNELS_

#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mpi4py/mpi4py.h>
#include "mpistats_scalapack_kernels.hpp"
#include "blacs.h"
#include "array_helper.hpp"

namespace py = pybind11;

void pb11_mpistats_scalapack_pdsyev(int mpi_region,
                                    const int nprow, const int npcol,
                                    py::array_t<double> a_pyarr,
                                    py::array_t<double> eigval_pyarr,
                                    py::array_t<double> eigvec_pyarr,
                                    int m,
                                    int mb, int nb);

#endif