set(CMAKE_CXX_COMPILER g++ CACHE STRING "" FORCE)
set(CMAKE_CXX_COMPILER_FLAGS "-std=c++11 -O3" CACHE STRING "" FORCE)

set(BLAS_DIR "$ENV{OPENBLASROOT}/lib" CACHE STRING "" FORCE)
set(BLAS_LIB openblas CACHE STRING "" FORCE)