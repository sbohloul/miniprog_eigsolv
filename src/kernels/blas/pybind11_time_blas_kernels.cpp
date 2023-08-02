#include <pybind11_time_blas_kernels.hpp>

namespace py = pybind11;

double pytime_cblas_ddot(int niter, py::array_t<double> x_pyarr, py::array_t<double> y_pyarr)
{
    py::buffer_info x_buf = x_pyarr.request();
    py::buffer_info y_buf = y_pyarr.request();

    assert(x_buf.ndim == y_buf.ndim);
    assert(x_buf.size == y_buf.size);

    int nelem = x_buf.size;
    auto x_ptr = static_cast<double *>(x_buf.ptr);
    auto y_ptr = static_cast<double *>(y_buf.ptr);

    std::vector<double> x(x_ptr, x_ptr + nelem);
    std::vector<double> y(y_ptr, y_ptr + nelem);

    double t_kernel = time_cblas_ddot(niter, x, y);
    return t_kernel;
}