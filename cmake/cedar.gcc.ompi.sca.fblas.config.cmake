# ============
# CXX compiler
# ============
set(CMAKE_CXX_COMPILER g++ CACHE STRING "" FORCE)
set(CMAKE_CXX_COMPILER_FLAGS "-Wall -std=c++11" CACHE STRING "" FORCE)

# ============
# MPI compiler
# ============
set(MPI_CXX_COMPILER mpic++ CACHE STRING "" FORCE)
set(MPI_CXX_COMPILER_FLAGS "-Wall -std=c++11" CACHE STRING "" FORCE)

# ====
# BLAS
# ====
# export OPENBLASROOT=/home/sbohloul/.local/openblas/0.3.21
# set(BLAS_DIR "$ENV{EBROOTSCALAPACK}" CACHE STRING "" FORCE)
# set(BLAS_LIB ${BLAS_DIR}/lib64/libscalapack.a CACHE STRING "" FORCE)
set(BLAS_LIB flexiblas CACHE STRING "" FORCE)

# =========
# ScaLAPACK
# =========
# export SCALAPACKROOT=~/.local/scalapack/2.2.0
# set(SCALAPACK_DIR "$ENV{EBROOTSCALAPACK}" CACHE STRING "" FORCE)
# set(SCALAPACK_LIB ${SCALAPACK_DIR}/lib64/libscalapack.a ${BLAS_LIB} -lgfortran CACHE STRING "" FORCE)
set(SCALAPACK_LIB scalapack CACHE STRING "" FORCE)

set(USE_NETLIB ON CACHE STRING "" FORCE)
