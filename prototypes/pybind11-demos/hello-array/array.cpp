#include <iostream>
#include "array.hpp"

void array_info(py::array_t<double> pyarr_x)
{
    py::buffer_info x_buf = pyarr_x.request();

    std::cout << "ptr: " << x_buf.ptr << std::endl;
    std::cout << "itemsize: " << x_buf.itemsize << std::endl;
    std::cout << "format: " << x_buf.format << std::endl;
    std::cout << "ndim: " << x_buf.ndim << std::endl;
    std::cout << "size: " << x_buf.size << std::endl;

    for (size_t i = 0; i < x_buf.strides.size(); i++)
    {
        std::cout << "stride[" << i << "]: " << x_buf.strides[i] << std::endl;
    }
    for (size_t i = 0; i < x_buf.shape.size(); i++)
    {
        std::cout << "shape[" << i << "]: " << x_buf.shape[i] << std::endl;
    }
}
