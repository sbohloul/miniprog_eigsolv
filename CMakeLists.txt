cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# =====================
# don't build in source
# =====================
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR
        "Building in source directory is not allowed. Pleas remove CMakeCache.txt and CMakeFiles/, 
    and build out of source directory."
    )
endif()

# ==================
# project descrption
# ==================
project(
    mini_eigsolv
    VERSION 0.1.0
    LANGUAGES CXX
)

# ===
# CXX
# ===
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================
# all libs compiled with -fPIC
# ============================
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE STRING "" FORCE)

# ===
# MPI
# ===
# set(MPI_CXX_SKIP_MPICXX ON)
find_package(MPI REQUIRED COMPONENTS CXX)

message(STATUS "MPI::MPI_CXX ${MPI_CXX_LIBRARIES}")
message(STATUS "MPI::MPI_INC ${MPI_CXX_INCLUDE_DIRS}")

# =======
# Python3
# =======
# find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
    COMMAND bash -c "which python3"
    OUTPUT_VARIABLE PYTHON3_EXECUTABLE
    RESULT_VARIABLE EXIT_CODE_PYTHON3
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(EXIT_CODE_PYTHON3 EQUAL 0)
    string(REGEX REPLACE "/bin/python3$" "" PYTHON3_PATH ${PYTHON3_EXECUTABLE})
    message(STATUS "PYTHON3_PATH ${PYTHON3_PATH}")
else()
    message(FATAL_ERROR "python was not found")
endif()

set(PYTHON3_LIBRARY ${PYTHON3_PATH}/lib)
set(PYTHON3_INCLUDE_DIR $${PYTHON3_PATH}/include)

# python3 extension suffix
execute_process(
    COMMAND python3-config --extension-suffix
    OUTPUT_VARIABLE PYTHON3_EXTENSION_SUFFIX
    RESULT_VARIABLE EXIT_CODE_PYTHON3_EXTENSION_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT EXIT_CODE_PYTHON3_EXTENSION_SUFFIX EQUAL 0)
    message(FATAL_ERROR "python3-config --extension-suffix didn't return successfuly")
else()
    message(STATUS "PYTHON3_EXTENSION_SUFFIX ${PYTHON3_EXTENSION_SUFFIX}")
endif()

# ========
# Pybind11
# ========
# find_package(pybind11)
execute_process(
    COMMAND ${PYTHON3_EXECUTABLE} -c "import pybind11"
    RESULT_VARIABLE EXIT_CODE_IMPORT_PYBIND11
)

if(NOT ${EXIT_CODE_IMPORT_PYBIND11} EQUAL 0)
    message(FATAL_ERROR "Python does not have pybind11")
endif()

execute_process(
    COMMAND ${PYTHON3_EXECUTABLE} -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    RESULT_VARIABLE EXIT_CODE_PYBIND11_INCLUDES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(EXIT_CODE_PYBIND11_INCLUDES EQUAL 0)
    string(REGEX REPLACE "-I" "" PYBIND11_INCLUDE_DIR ${PYBIND11_INCLUDE_DIR})
    string(REGEX REPLACE " " ";" PYBIND11_INCLUDE_DIR ${PYBIND11_INCLUDE_DIR})
    message(STATUS "PYBIND11_INCLUDE_DIR ${PYBIND11_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Pybind11 includes not found")
endif()

# set pybind11 flags
if(
    CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
)
    set(CXX_PYBIND11_COMPILE_OPTIONS
        -shared
        -fPIC
        CACHE STRING "c++ compile flags for pybind11"
    )

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CXX_PYBIND11_COMPILE_OPTIONS
        -shared
        -fPIC
        CACHE STRING "c++ compile flags for pybind11"
    )
else()
    message(STATUS "CMAKE_CXX_COMPILER = " ${CMAKE_CXX_COMPILER})
    message(FATAL_ERROR "CMAKE_CXX_COMPILER_ID " ${CMAKE_CXX_COMPILER_ID} " not supporterd")
endif()

# ======
# mpi4py
# ======
execute_process(
    COMMAND ${PYTHON3_EXECUTABLE} -c "import mpi4py"
    RESULT_VARIABLE EXIT_CODE_IMPORT_MPI4PY
)

if(NOT ${EXIT_CODE_IMPORT_MPI4PY} EQUAL 0)
    message(FATAL_ERROR "Python does not have mpi4py")
endif()

execute_process(
    COMMAND ${PYTHON3_EXECUTABLE} -c "import mpi4py as m; print(m.get_include())"
    OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
    RESULT_VARIABLE EXIT_CODE_MPI4PY_INCLUDES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(EXIT_CODE_MPI4PY_INCLUDES EQUAL 0)
    string(REGEX REPLACE "-I" "" MPI4PY_INCLUDE_DIR ${MPI4PY_INCLUDE_DIR})
    string(REGEX REPLACE " " ";" MPI4PY_INCLUDE_DIR ${MPI4PY_INCLUDE_DIR})
    message(STATUS "MPI4PY_INCLUDE_DIR ${MPI4PY_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "mpi4py includes not found")
endif()

# =======
# options
# =======
option(USE_MKL "Use MKL for linear algebra" OFF)
option(USE_OPENBLAS "Use OpenBLAS for linear algebra" OFF)
option(USE_NETLIB "Use netlib for linear algebra" OFF)
message(STATUS "USE_MKL = " ${USE_MKL})
message(STATUS "USE_OPENBLAS = " ${USE_OPENBLAS})
message(STATUS "USE_NETLIB = " ${USE_NETLIB})

if(USE_MKL)
    add_compile_definitions(USE_MKL)
elseif(USE_NETLIB)
    add_compile_definitions(USE_NETLIB)
endif()

# =================
# install variables
# =================
set(INSTALL_BIN_DIR bin)
set(INSTALL_LIB_DIR lib)
set(INSTALL_INCLUDE_DIR include)
set(CMAKE_EXECUTABLE_SUFFIX ".x" CACHE STRING "Executable suffix")

# set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation directory")

# if(USE_MKL)
# find_package(MKL CONFIG REQUIRED)
# set(BLAS_LIB MKL::BLAS)
# elseif(USE_OPENBLAS)
# find_package(OPENBLAS CONFIG REQUIRED)
# set(BLAS_LIB ${OpenBLAS_LIBRARIES})
# else()
# message(FATAL_ERROR
# "No linear algebra library specified")
# endif()

# print
message(STATUS "BLAS_LIB ${BLAS_LIB}")
message(STATUS "BLAS_DIR ${BLAS_DIR}")
message(STATUS "SCALAPACK_LIB ${SCALAPACK_LIB}")
message(STATUS "SCALAPACK_DIR ${SCALAPACK_DIR}")

# add directories
add_subdirectory(src)
add_subdirectory(drivers)

# install(FILES MathFunctions.h DESTINATION include)