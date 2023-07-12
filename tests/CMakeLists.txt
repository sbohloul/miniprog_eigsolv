add_executable(driver_time_cblas_ddot driver_time_cblas_ddot.cpp)

target_link_libraries(driver_time_cblas_ddot
    PRIVATE
    time_kernels
    ${BLAS_LIB}
)
target_link_directories(driver_time_cblas_ddot
    PRIVATE
    ${BLAS_DIR}
)

install(TARGETS driver_time_cblas_ddot DESTINATION ${INSTALL_BIN_DIR})
install(FILES bench_time_cblas_ddot.sh DESTINATION ${INSTALL_BIN_DIR})
