# ============
# CXX compiler
# ============
set(CMAKE_CXX_COMPILER icpc CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -diag-disable=10441 -traceback -pedantic" CACHE STRING "" FORCE)

# ============
# MPI compiler
# ============
set(MPI_CXX_COMPILER mpiicpc CACHE STRING "" FORCE)
set(MPI_CXX_COMPILER_FLAGS "-cxx=icpc" CACHE STRING "" FORCE)

# ====
# BLAS
# ====
set(BLAS_DIR "$ENV{MKLROOT}/lib/intel64" CACHE STRING "" FORCE)
set(BLAS_LIB mkl_core pthread m dl mkl_intel_lp64 mkl_intel_thread iomp5 CACHE STRING "" FORCE)

# ========
# ScaLAPCK
# ========
set(SCALAPACK_DIR "$ENV{MKLROOT}/lib/intel64" CACHE STRING "" FORCE)
set(SCALAPACK_LIB mkl_core pthread m dl mkl_intel_lp64 mkl_intel_thread iomp5 mkl_blacs_intelmpi_lp64 mkl_scalapack_lp64 CACHE STRING "" FORCE)

set(USE_MKL ON CACHE STRING "" FORCE)

# ==============
# set(CMAKE_CXX_COMPILER_FLAGS "-Wall -std=c++11 -diag-disable=10441 -g -traceback -pedantic" CACHE STRING "" FORCE)
# set(USE_MKL ON CACHE STRING "" FORCE)
# set(MKL_DIR $ENV{MKLROOT} CACHE STRING "" FORCE)
