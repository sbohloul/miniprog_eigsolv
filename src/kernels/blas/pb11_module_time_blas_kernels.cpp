#include <pb11_time_blas_kernels.hpp>

PYBIND11_MODULE(pb11_time_blas_kernels, m)
{
    m.doc() = "Module for timing blas kernels.";

    m.def("pb11_time_cblas_ddot", &pb11_time_cblas_ddot, "Timing cblas_ddot for x and y input vectors.");
}