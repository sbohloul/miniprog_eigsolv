set(CMAKE_CXX_COMPILER icx CACHE STRING "" FORCE)
set(CMAKE_CXX_COMPILER_FLAGS "-std=c++11 -O3" CACHE STRING "" FORCE)

# set(USE_MKL ON CACHE STRING "" FORCE)
# set(MKL_DIR $ENV{MKLROOT} CACHE STRING "" FORCE)
set(BLAS_DIR "$ENV{MKLROOT}/lib/intel64" CACHE STRING "" FORCE)

# set(BLAS_LIB mkl_intel_ilp64 mkl_sequential mkl_core pthread m dl CACHE STRING "" FORCE)
set(BLAS_LIB mkl_intel_lp64 mkl_sequential mkl_core pthread m dl CACHE STRING "" FORCE)