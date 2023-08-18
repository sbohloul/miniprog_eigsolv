#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// #include <mpi4py/mpi4py.h>
#include "hello_blacs_kernels.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_pb11_hello_blacs, m)
{
    m.doc() = "pybind11 blacs plugin"; // optional module docstring

    m.def("print_info", &print_info, "A function that adds two arrays");
}
