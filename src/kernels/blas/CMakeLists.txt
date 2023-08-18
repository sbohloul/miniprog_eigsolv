add_library(time_blas_kernels)

target_sources(time_blas_kernels
    PRIVATE
    time_blas_kernels.cpp
)

target_link_libraries(time_blas_kernels
    PUBLIC
    ${BLAS_LIB}
    utils
    interfaces
)

target_include_directories(time_blas_kernels
    PRIVATE
    ${BLAS_DIR}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ===============
# pybind11 module
# ===============
add_library(pb11_time_blas_kernels MODULE)

target_sources(pb11_time_blas_kernels
    PRIVATE
    pb11_module_time_blas_kernels.cpp
    pb11_time_blas_kernels.cpp
)

target_link_libraries(pb11_time_blas_kernels
    PRIVATE
    time_blas_kernels
)

target_include_directories(pb11_time_blas_kernels
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYBIND11_INCLUDE_DIR}
)

target_compile_options(pb11_time_blas_kernels
    PRIVATE
    ${CXX_PYBIND11_COMPILE_OPTIONS}
)

set_target_properties(pb11_time_blas_kernels
    PROPERTIES
    SUFFIX ${PYTHON3_EXTENSION_SUFFIX}
    PREFIX ""
)

install(TARGETS pb11_time_blas_kernels
    LIBRARY
    DESTINATION ${INSTALL_BIN_DIR}
)