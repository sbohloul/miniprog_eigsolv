#ifndef _MPISTATS_SCALAPACK_KERNELS_HPP_
#define _MPISTATS_SCALAPACK_KERNELS_HPP_

#include <vector>
#include <mpi.h>
#include "blacs.h"
#include "timer.hpp"

void mpistats_scalapack_pdsyev(int mpi_region, int nprow, int npcol, const std::vector<double> &a_glb, std::vector<double> &eigval_glb, std::vector<double> &eigvec_glb, int m, int mb, int nb);

#endif