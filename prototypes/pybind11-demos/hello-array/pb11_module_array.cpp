// #include <pybind11/pybind11.h>
#include "array.hpp"

PYBIND11_MODULE(_pb11_hello_array, m)
{
    m.doc() = "pybind11 array plugin"; // optional module docstring

    m.def("array_info", &array_info, "A function that adds two arrays");
}
