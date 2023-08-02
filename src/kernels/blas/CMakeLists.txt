add_library(time_kernels_blas)

target_sources(time_kernels_blas
    PRIVATE
    time_blas_kernels.cpp
)

target_link_libraries(time_kernels_blas
    PRIVATE
    utils
    interfaces
    ${BLAS_LIB}
)

target_include_directories(time_kernels_blas
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
    ${BLAS_DIR}
)
