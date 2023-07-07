#!/bin/bash

g++ -Wall -std=c++11 -fPIC -shared \
    -I./ \
    -I/home/sbohloul/Projects/scicomp/miniprog_eigsolv/src/utils \
    -I/home/sbohloul/Projects/scicomp/miniprog_eigsolv/src/interfaces \
    $(python3 -m pybind11 --includes) \
    -o pybind11_time_kernels$(python3-config --extension-suffix) \
    pybind11_time_kernels.cpp \
    pybind11_time_blas_kernels.cpp time_blas_kernels.cpp \
    /home/sbohloul/Projects/scicomp/miniprog_eigsolv/src/utils/timer.cpp \
    -L/home/sbohloul/.local/openblas/0.3.21/lib -lopenblas