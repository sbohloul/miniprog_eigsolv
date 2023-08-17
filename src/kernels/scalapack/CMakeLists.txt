add_library(time_scalapack_kernels)

target_sources(time_scalapack_kernels
    PRIVATE
    time_scalapack_kernels.cpp
)

target_link_libraries(time_scalapack_kernels
    PUBLIC
    utils
    interfaces
    MPI::MPI_CXX
    ${SCALAPACK_LIB}
)

target_include_directories(time_scalapack_kernels
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SCALAPACK_DIR}
)
