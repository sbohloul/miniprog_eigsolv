#include "pybind11_time_scalapack_kernels.hpp"
#include "time_scalapack_kernels.hpp"
#include "blacs.h"

namespace py = pybind11;

double pytime_scalapack_pdgemm(int niter, int nprow, int npcol,
                               int m, int n,
                               int mb, int nb,
                               py::array_t<double> x_pyarr,
                               py::array_t<double> y_pyarr,
                               py::array_t<dobule> z_pyarr)
{

    py::buffer_info x_buf = x_pyarr.request();
    py::buffer_info y_buf = y_pyarr.request();
    py::buffer_info z_buf = z_pyarr.request();

    assert(x_buf.ndim == y_buf.ndim);
    assert(x_buf.ndim == z_buf.ndim);
    assert(x_buf.size == y_buf.size);
    assert(x_buf.size == z_buf.size);

    int nelem = x_buf.size;
    auto a_ptr = static_cast<double *>(x_buf.ptr);
    auto b_ptr = static_cast<double *>(y_buf.ptr);
    auto c_ptr = static_cast<double *>(z_buf.ptr);

    // initialize arrays
    std::vector<double> a(a_ptr, a_ptr + nelem);
    std::vector<double> b(b_ptr, b_ptr + nelem);
    std::vector<double> c(c_ptr, c_ptr + nelem);

    double t_kernel = time_scalapack_pdgemm(niter, nprow, npcol, a, b, c, m, n, mb, nb);

    return t_kernel;
}