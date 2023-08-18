#!/bin/bash

compile_hello=0
compile_array=0
compile_mpi=0
compile_scalapack=1

if [ $compile_hello -ne 0 ]; then
echo "compiling hello"
g++ -Wall -std=c++11 -fPIC -shared \
    -I./ \
    $(python3 -m pybind11 --includes) \
    -o hello$(python3-config --extension-suffix) \
    hello.cpp pybind11_hello.cpp
fi

if [ $compile_array -ne 0 ]; then
echo "compiling m_array"
g++ -Wall -std=c++11 -fPIC -shared \
    -I./ \
    $(python3 -m pybind11 --includes) \
    -o m_array$(python3-config --extension-suffix) \
    array.cpp pybind11_array.cpp    
fi

if [ $compile_mpi -ne 0 ]; then
echo "compiling mpi_hello"
mpic++ -Wall -std=c++11 -fPIC -shared \
    -I./ \
    $(python3 -m pybind11 --includes) \
    -I $(python -c "import mpi4py as m; print(m.get_include())") \
    -o mpi_hello$(python3-config --extension-suffix) \
    mpi_hello.cpp pybind11_mpi_hello.cpp
fi

if [ $compile_scalapack -ne 0 ]; then
echo "compiling scalapack"
mpic++ -Wall -std=c++11 -fPIC -shared \
    -I./ \
    $(python3 -m pybind11 --includes) \
    -I $(python -c "import mpi4py as m; print(m.get_include())") \
    -o _pb11time_scalapack_kernels$(python3-config --extension-suffix) \
    pb11m_time_scalapack_kernels.cpp pb11_time_scalapack_kernels.cpp \
    time_scalapack_kernels.cpp \
    timer.cpp array_helper.cpp \
    -L/home/sbohloul/.local/scalapack/2.2.0 -lscalapack
fi

    