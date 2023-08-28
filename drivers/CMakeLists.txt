# =========
# blas_ddot
# =========
add_executable(driver_time_blas_ddot
    driver_time_blas_ddot.cpp
)

target_link_libraries(driver_time_blas_ddot
    PRIVATE
    time_blas_kernels
)

install(
    TARGETS driver_time_blas_ddot
    DESTINATION ${INSTALL_BIN_DIR}
)

install(
    FILES bench_time_blas_ddot.sh
    DESTINATION ${INSTALL_BIN_DIR}
)

install(
    FILES pydriver_time_blas_ddot.py
    DESTINATION ${INSTALL_BIN_DIR}
)

# ================
# scalapack pdgemm
# ================
add_executable(driver_time_scalapack_pdgemm
    driver_time_scalapack_pdgemm.cpp
)

target_link_libraries(driver_time_scalapack_pdgemm
    PRIVATE
    time_scalapack_kernels
)

install(
    TARGETS driver_time_scalapack_pdgemm
    DESTINATION ${INSTALL_BIN_DIR}
)

install(
    FILES bench_time_scalapack_pdgemm.sh
    DESTINATION ${INSTALL_BIN_DIR}
)

install(
    FILES pydriver_time_scalapack_pdgemm.py
    DESTINATION ${INSTALL_BIN_DIR}
)

add_subdirectory(scalapack)
