cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# ======================
# Do not build in source
# ======================
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR
        "Building in source directory is not allowed. Pleas remove CMakeCache.txt and CMakeFiles/, 
    and build out of source directory."
    )
endif()

# =======
# Project
# =======
project(
    mini_prog
    VERSION 0.1.0
    LANGUAGES Fortran
)

# =======
# Options
# =======
# option(USE_MKL "Use MKL for linear algebra" OFF)
# option(USE_NETLIB "Use NETLIB for scalapack" OFF)
# option(USE_OPENBLAS "Use OPENBLAS for linear algebra" OFF)

# if(USE_MKL)
# set(SCALAPACK_LINK_LIB ${MKL_IMPORTED_TARGETS})
# set(SCALAPACK_LINK_DIR ${MKL_ROOT})
# elseif(USE_NETLIB)
# set(SCALAPACK_LINK_LIB "scalapack")
# set(SCALAPACK_LINK_DIR ${MKL_ROOT})
# endif()
set(SCALAPACK_ROOT "/home/sbohloul/.local/scalapack/2.2.0")
set(SCALAPACK_LINK_LIB "scalapack")
set(SCALAPACK_LINK_DIR ${SCALAPACK_ROOT})

# if(NOT CMAKE_Fortran_COMPILER)
# message(FATAL_ERROR
# "Fortran compiler not specified, use CMAKE_Fortran_COMPILER variable to set it.")
# endif()
# enable_language(Fortran)
# if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
# message(STATUS "XXX INTEL")
# elseif(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
# message(STATUS "XXX GNU")
# endif()
find_package(MPI REQUIRED COMPONENTS Fortran)

# =======
# Options
# =======
option(SCALAPACK_PATH "Path to scalapack library" OFF)
option(MKL_PATH "Path to mkl" ON)

if(SCALAPACK_PATH)
    message(STATUS "scalapack path: ${SCALAPACK_PATH}")
else()
    message(STATUS "scalapack path not set")
endif()

# option(${PROJECT_NAME}_USE_XXX " Add a XXX target to use XXX " OFF)
# option(${PROJECT_NAME}_USE_XXX " Add a YYY target to use YYY " OFF)
# message(STATUS " Fortran compiler ${CMAKE_Fortran_COMPILER} ")
# message(STATUS " Fortran compiler id ${CMAKE_Fortran_COMPILER_ID} ")
# message(STATUS " Fortran flags ${CMAKE_Fortran_FLAGS} ")
# message(STATUS " Fortran min flags ${CMAKE_Fortran_MIN_FLAGS} ")
# set(CMAKE_PREFIX_PATH ${MKL_PATH} ${CMAKE_CMAKE_PREFIX_PATH})
find_package(MKL CONFIG)

if(MKL_FOUND)
    message(STATUS " MKL is found ")
    message(STATUS "${MKL_IMPORTED_TARGETS}")
else()
    message(STATUS " MKL is not found ")
endif()

message(STATUS " ================================================ ")
message(STATUS " ")

# if(MPI_FOUND)
# endif()

# find_package(MKL REQUIRED)

# if(MKL_FOUND)
# message(STATUS " MKL is found ")
# else()
# message((WARNING " MKL not found "))
# endif()

# Test a simple command with mpi
set(MPIEXEC_NUMPROCS 2)
message(STATUS " Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_NUMPROCS} ${MPIEXEC_PREFLAGS} hostname ${MPIEXEC_POSTFLAGS} ")
execute_process(COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_NUMPROCS} ${MPIEXEC_PREFLAGS} hostname ${MPIEXEC_POSTFLAGS}
    RESULT_VARIABLE CHECK_MPI_RESULT
    OUTPUT_QUIET ERROR_QUIET)

if(CHECK_MPI_RESULT EQUAL 0)
    message(STATUS " MPI command works succesfully. ")
else()
    message(FATAL_ERROR " MPI command failed. ")
endif()

# ==================
# Add subdirectories
# ==================
add_subdirectory(src)