# ============
# CXX compiler
# ============
set(CMAKE_CXX_COMPILER icpc CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -traceback" CACHE STRING "" FORCE)

# ============
# MPI compiler
# ============
set(MPI_CXX_COMPILER mpiicpc CACHE STRING "" FORCE)
set(MPI_CXX_COMPILER_FLAGS "-cxx=icpc" CACHE STRING "" FORCE)

# ====
# BLAS
# ====
# set(BLAS_DIR "$ENV{MKLROOT}/lib/intel64_lin" CACHE STRING "" FORCE)
set(BLAS_LIB mkl_intel_lp64 mkl_sequential mkl_core pthread m dl CACHE STRING "" FORCE)

# ========
# ScaLAPCK
# ========
# set(SCALAPACK_DIR "$ENV{MKLROOT}/lib/intel64_lin" CACHE STRING "" FORCE)
set(SCALAPACK_LIB mkl_scalapack_lp64 mkl_intel_lp64 mkl_sequential mkl_core mkl_blacs_intelmpi_lp64 pthread m dl CACHE STRING "" FORCE)

set(USE_MKL ON CACHE STRING "" FORCE)