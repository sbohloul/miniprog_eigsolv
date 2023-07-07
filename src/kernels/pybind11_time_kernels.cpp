#include <pybind11_time_kernels.hpp>

PYBIND11_MODULE(pybind11_time_kernels, m)
{
    m.doc() = "Module for timing kernels.";

    m.def("time_cblas_ddot", &pytime_cblas_ddot, "Timing cblas_ddot for x and y input vectors.");
}