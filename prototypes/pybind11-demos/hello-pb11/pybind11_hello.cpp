#include <pybind11/pybind11.h>
#include "hello.hpp"

namespace py = pybind11;

PYBIND11_MODULE(hello, m)
{
    m.doc() = "pybind11 hello plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two integers");
}
