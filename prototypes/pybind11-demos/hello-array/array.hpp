#ifndef _ARRAY_HPP_
#define _ARRAY_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void array_info(py::array_t<double> pyarr_x);

#endif