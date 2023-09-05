#ifndef _TIME_SCALAPACK_KERNELS_HPP_
#define _TIME_SCALAPACK_KERNELS_HPP_

#include <vector>
#include "blacs.h"
#include "timer.hpp"

double time_scalapack_pdgemm(int niter, int nprow, int npcol, const std::vector<double> &a_glb, const std::vector<double> &b_glb, std::vector<double> &c_glb, int m, int n, int mb, int nb);
double time_scalapack_pdsyev(int niter, int nprow, int npcol, const std::vector<double> &a_glb, std::vector<double> &eigval_glb, std::vector<double> &eigvec_glb, int m, int mb, int nb);

#endif // _TIME_SCALAPACK_KERNELS_HPP_