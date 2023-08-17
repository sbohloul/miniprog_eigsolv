# ============
# CXX compiler
# ============
set(CMAKE_CXX_COMPILER mpic++ CACHE STRING "" FORCE)
set(CMAKE_CXX_COMPILER_FLAGS "-Wall -std=c++11" CACHE STRING "" FORCE)

# ============
# MPI compiler
# ============
set(MPI_CXX_COMPILER mpic++ CACHE STRING "" FORCE)
set(MPI_CXX_COMPILER_FLAGS "-Wall -std=c++11" CACHE STRING "" FORCE)

# ============================
# all libs compiled with -fPIC
# ============================
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE STRING "" FORCE)

# ====
# BLAS
# ====
# OPENBLAS=/home/sbohloul/.local/openblas/0.3.21
# export OPENBLASROOT=${OPENBLAS}
set(BLAS_DIR "$ENV{OPENBLASROOT}/lib" CACHE STRING "" FORCE)
set(BLAS_LIB ${BLAS_DIR}/libopenblas.a CACHE STRING "" FORCE)

# =========
# ScaLAPACK
# =========
# export SCALAPACKROOT=~/.local/scalapack/2.1.0
set(SCALAPACK_DIR "$ENV{SCALAPACKROOT}" CACHE STRING "" FORCE)
set(SCALAPACK_LIB ${SCALAPACK_DIR}/libscalapack.a ${BLAS_LIB} -lgfortran CACHE STRING "" FORCE)

set(USE_NETLIB ON CACHE STRING "" FORCE)
