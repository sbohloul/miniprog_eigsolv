#!/bin/bash


mpiicpc -Wall -std=c++11 -fPIC -shared -diag-disable=10441 \
    -I./ \
    $(python3 -m pybind11 --includes) \
    -I $(python -c "import mpi4py as m; print(m.get_include())") \
    -o _pb11time_scalapack_kernels$(python3-config --extension-suffix) \
    pb11m_time_scalapack_kernels.cpp pb11_time_scalapack_kernels.cpp \
    time_scalapack_kernels.cpp \
    timer.cpp array_helper.cpp \
    -L${MKLROOT}/lib/intel64 \
    -lmkl_scalapack_lp64 \
    -lmkl_blacs_intelmpi_lp64 \
    -lmkl_intel_lp64 \
    -lmkl_sequential \
    -lmkl_core \
    -lpthread \
    -lm 
    