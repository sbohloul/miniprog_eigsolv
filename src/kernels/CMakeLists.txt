add_library(time_kernels OBJECT)
target_sources(time_kernels
    PRIVATE
    time_blas_kernels.cpp
    time_blas_kernels.hpp
    PUBLIC
    time_kernels.hpp
)
target_link_libraries(time_kernels
    PRIVATE
    utils
    interfaces
)
target_include_directories(time_kernels
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(pybind11_time_kernels MODULE)
target_sources(pybind11_time_kernels
    PRIVATE
    pybind11_time_kernels.cpp
    pybind11_time_blas_kernels.cpp
    pybind11_time_blas_kernels.hpp
    PUBLIC
    pybind11_time_kernels.hpp
)
target_link_directories(pybind11_time_kernels
    PRIVATE
    ${BLAS_DIR})
target_link_libraries(pybind11_time_kernels
    PRIVATE
    time_kernels
    ${BLAS_LIB}
)
target_include_directories(pybind11_time_kernels
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYBIND11_INCLUDE_DIR}
)
target_compile_options(pybind11_time_kernels
    PRIVATE
    ${CXX_PYBIND11_COMPILE_OPTIONS}
)
set_target_properties(pybind11_time_kernels
    PROPERTIES
    SUFFIX ${PYTHON3_EXTENSION_SUFFIX}
    PREFIX ""
)
install(TARGETS pybind11_time_kernels
    LIBRARY DESTINATION ${INSTALL_LIB_DIR}
)