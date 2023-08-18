#include "pb11_time_scalapack_kernels.hpp"

double pb11_time_scalapack_pdgemm(int niter,
                                  const int nprow, const int npcol,
                                  py::array_t<double> a_pyarr,
                                  py::array_t<double> b_pyarr,
                                  py::array_t<double> c_pyarr,
                                  int m, int n,
                                  int mb, int nb)

{

    py::buffer_info a_buf = a_pyarr.request();
    py::buffer_info b_buf = b_pyarr.request();
    py::buffer_info c_buf = c_pyarr.request();

    assert(a_buf.ndim == b_buf.ndim);
    assert(a_buf.size == b_buf.size);

    int nelem = a_buf.size;
    auto a_ptr = static_cast<double *>(a_buf.ptr);
    auto b_ptr = static_cast<double *>(b_buf.ptr);
    auto c_ptr = static_cast<double *>(c_buf.ptr);

    // initialize global vectors
    std::vector<double> a(a_ptr, a_ptr + nelem);
    std::vector<double> b(b_ptr, b_ptr + nelem);
    std::vector<double> c(c_ptr, c_ptr + nelem);

    int myrank, nprocs;
    Cblacs_pinfo(&myrank, &nprocs);

    // for (int j = 0; j < n; j++)
    // {
    //     for (int i = 0; i < m; i++)
    //     {
    //         a[j * m + i] = static_cast<double>(i);
    //         b[j * m + i] = static_cast<double>(j);
    //         c[j * m + i] = static_cast<double>(0);
    //     }
    // }

    // if (myrank == 0)
    // {
    //     std::cout << "a: " << std::endl;
    //     print_array(a, m, n, m);
    //     std::cout << "b: " << std::endl;
    //     print_array(b, m, n, m);
    //     std::cout << "c: " << std::endl;
    //     print_array(c, m, n, m);
    // }

    double t_kernel = time_scalapack_pdgemm(niter, nprow, npcol, a, b, c, m, n, mb, nb);

    if (myrank == 0)
    {
        std::cout << "t_kernel = " << t_kernel << std::endl;
        // std::cout << "c: " << std::endl;
        // print_array(c, m, n, m);

        // std::cout << c.data() << std::endl;
        // std::cout << c_ptr << std::endl;
    }

    for (int i = 0; i < c_buf.size; i++)
    {
        *c_ptr++ = c[i];
    }

    return t_kernel;
}
