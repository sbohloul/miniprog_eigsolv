set(CMAKE_CXX_COMPILER mpiicpc CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -diag-disable=10441 -g -traceback -pedantic" CACHE STRING "" FORCE)

# set(CMAKE_CXX_COMPILER_FLAGS "-Wall -std=c++11 -diag-disable=10441 -g -traceback -pedantic" CACHE STRING "" FORCE)

# set(USE_MKL ON CACHE STRING "" FORCE)
# set(MKL_DIR $ENV{MKLROOT} CACHE STRING "" FORCE)
set(BLAS_DIR "$ENV{MKLROOT}/lib/intel64" CACHE STRING "" FORCE)

# set(BLAS_LIB mkl_intel_ilp64 mkl_sequential mkl_core pthread m dl CACHE STRING "" FORCE)
set(BLAS_LIB mkl_intel_lp64 mkl_sequential mkl_core pthread m dl CACHE STRING "" FORCE)

set(SCALAPACK_DIR "$ENV{MKLROOT}/lib/intel64" CACHE STRING "" FORCE)
set(SCALAPACK_LIB mkl_scalapack_lp64 mkl_blacs_intelmpi_lp64 mkl_intel_lp64 mkl_sequential mkl_core pthread m dl CACHE STRING "" FORCE)