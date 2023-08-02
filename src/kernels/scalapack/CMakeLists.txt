add_library(time_kernels_scalapack)

target_sources(time_kernels_scalapack
    PRIVATE
    time_scalapack_kernels.cpp
)

target_link_libraries(time_kernels_scalapack
    PRIVATE
    utils
    interfaces
    MPI::MPI_CXX
    ${SCALAPACK_LIB}
)

target_include_directories(time_kernels_scalapack
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
    ${SCALAPACK_DIR}
)
