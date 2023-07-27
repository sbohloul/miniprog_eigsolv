# blas_ddot
add_executable(driver_time_cblas_ddot driver_time_cblas_ddot.cpp)
target_link_libraries(driver_time_cblas_ddot
    PRIVATE
    time_kernels
    ${BLAS_LIB}
    ${SCALAPACK_LIB}
)
target_link_directories(driver_time_cblas_ddot
    PRIVATE
    ${BLAS_DIR}
    ${SCALAPACK_DIR}
)
install(TARGETS driver_time_cblas_ddot DESTINATION ${INSTALL_BIN_DIR})
install(FILES bench_time_cblas_ddot.sh DESTINATION ${INSTALL_BIN_DIR})
install(FILES bench_pybind11_time_cblas_ddot.py DESTINATION ${INSTALL_BIN_DIR})

# scalapack pdgemm
add_executable(driver_time_scalapack_pdgemm driver_time_scalapack_pdgemm.cpp)
target_link_libraries(driver_time_scalapack_pdgemm
    PRIVATE
    time_kernels
    utils
    interfaces
    ${SCALAPACK_LIB}
)
target_link_directories(driver_time_scalapack_pdgemm
    PRIVATE
    ${SCALAPACK_DIR}
)
install(TARGETS driver_time_scalapack_pdgemm DESTINATION ${INSTALL_BIN_DIR})

install(FILES bench_time_cblas_ddot.sh DESTINATION ${INSTALL_BIN_DIR})
install(FILES bench_pybind11_time_cblas_ddot.py DESTINATION ${INSTALL_BIN_DIR})
