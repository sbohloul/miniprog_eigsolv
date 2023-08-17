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

# ===============
# pybind11 module
# ===============
add_library(pb11_time_scalapack_kernels MODULE)
target_sources(pb11_time_scalapack_kernels
    PRIVATE
    pb11_time_scalapack_kernels.cpp
    pb11_module_time_scalapack_kernels.cpp
)
target_link_libraries(pb11_time_scalapack_kernels
    PRIVATE
    time_scalapack_kernels
)
target_include_directories(pb11_time_scalapack_kernels
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYBIND11_INCLUDE_DIR}
    ${MPI4PY_INCLUDE_DIR}
)
target_compile_options(pb11_time_scalapack_kernels
    PRIVATE
    ${CXX_PYBIND11_COMPILE_OPTIONS}
)
set_target_properties(pb11_time_scalapack_kernels
    PROPERTIES
    SUFFIX ${PYTHON3_EXTENSION_SUFFIX}
    PREFIX ""
)
install(TARGETS pb11_time_scalapack_kernels
    LIBRARY DESTINATION ${INSTALL_BIN_DIR}
)