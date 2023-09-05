#include "pb11_time_scalapack_kernels.hpp"

// ======
// PDGEMM
// ======
double pb11_time_scalapack_pdgemm(int niter,
                                  const int nprow, const int npcol,
                                  py::array_t<double> a_pyarr,
                                  py::array_t<double> b_pyarr,
                                  py::array_t<double> c_pyarr,
                                  int m, int n,
                                  int mb, int nb)

{
    int myrank, nprocs;
    Cblacs_pinfo(&myrank, &nprocs);

    if (myrank == 0)
    {
        std::cout << "Inside pb11_time_scalapack_pdgemm" << std::endl;
    }

    py::buffer_info a_buf = a_pyarr.request();
    py::buffer_info b_buf = b_pyarr.request();
    py::buffer_info c_buf = c_pyarr.request();

    if (myrank == 0)
    {
        assert(a_buf.ndim == b_buf.ndim);
        assert(a_buf.size == b_buf.size);
        assert(a_buf.ndim == c_buf.ndim);
        assert(a_buf.size == c_buf.size);
    }

    int nelem = a_buf.size;
    auto a_ptr = static_cast<double *>(a_buf.ptr);
    auto b_ptr = static_cast<double *>(b_buf.ptr);
    auto c_ptr = static_cast<double *>(c_buf.ptr);

    // initialize global vectors from passe arrays
    std::vector<double> a(a_ptr, a_ptr + nelem);
    std::vector<double> b(b_ptr, b_ptr + nelem);
    std::vector<double> c(c_ptr, c_ptr + nelem);

    if (myrank == 0)
    {
        // std::cout << "a: " << std::endl;
        // print_array(a, m, n, m);
        // std::cout << "b: " << std::endl;
        // print_array(b, m, n, m);
        // std::cout << "c: " << std::endl;
        // print_array(c, m, n, m);

        // std::cout << a.data() << std::endl;
        // std::cout << a_ptr << std::endl;
        // std::cout << b.data() << std::endl;
        // std::cout << b_ptr << std::endl;
    }

    double t_kernel = time_scalapack_pdgemm(niter, nprow, npcol, a, b, c, m, n, mb, nb);

    if (myrank == 0)
    {
        std::cout << "t_kernel = " << t_kernel << std::endl;
        // std::cout << "c: " << std::endl;
        // print_array(c, m, n, m);

        // std::cout << c.data() << std::endl;
        // std::cout << c_ptr << std::endl;
    }

    // rank 0 holds global c matrix
    for (int i = 0; i < c_buf.size; i++)
    {
        *c_ptr++ = c[i];
    }

    return t_kernel;
}

// ======
// PDSYEV
// ======
double pb11_time_scalapack_pdsyev(int niter,
                                  const int nprow, const int npcol,
                                  py::array_t<double> a_pyarr,
                                  py::array_t<double> eigval_pyarr,
                                  py::array_t<double> eigvec_pyarr,
                                  int m,
                                  int mb, int nb)

{
    int myrank, nprocs;
    Cblacs_pinfo(&myrank, &nprocs);

    if (myrank == 0)
    {
        std::cout << "Inside pb11_time_scalapack_pdsyev" << std::endl;
    }

    py::buffer_info a_buf = a_pyarr.request();
    py::buffer_info eigval_buf = eigval_pyarr.request();
    py::buffer_info eigvec_buf = eigvec_pyarr.request();

    if (myrank == 0)
    {
        // assert(a_buf.ndim == b_buf.ndim);
        // assert(a_buf.size == b_buf.size);
        // assert(a_buf.ndim == c_buf.ndim);
        // assert(a_buf.size == c_buf.size);
    }

    int nelem = a_buf.size;
    int neigval = eigval_buf.size;
    auto a_ptr = static_cast<double *>(a_buf.ptr);
    auto eigval_ptr = static_cast<double *>(eigval_buf.ptr);
    auto eigvec_ptr = static_cast<double *>(eigvec_buf.ptr);

    // initialize global vectors from passed arrays
    std::vector<double> a(a_ptr, a_ptr + nelem);
    std::vector<double> eigvec(eigvec_ptr, eigvec_ptr + nelem);
    std::vector<double> eigval(eigval_ptr, eigval_ptr + neigval);

    if (myrank == 0)
    {
        // std::cout << "a: " << std::endl;
        // print_array(a, m, n, m);
        // std::cout << "b: " << std::endl;
        // print_array(b, m, n, m);
        // std::cout << "c: " << std::endl;
        // print_array(c, m, n, m);

        // std::cout << a.data() << std::endl;
        // std::cout << a_ptr << std::endl;
        // std::cout << b.data() << std::endl;
        // std::cout << b_ptr << std::endl;
    }

    double t_kernel = time_scalapack_pdsyev(niter, nprow, npcol, a, eigval, eigvec, m, mb, nb);

    if (myrank == 0)
    {
        std::cout << "t_kernel = " << t_kernel << std::endl;
        // std::cout << "c: " << std::endl;
        // print_array(c, m, n, m);

        // std::cout << c.data() << std::endl;
        // std::cout << c_ptr << std::endl;
    }

    // rank 0 holds global eigvec matrix
    for (int i = 0; i < eigvec_buf.size; i++)
    {
        *eigvec_ptr++ = eigvec[i];
    }

    // all ranks holds a copy of eigval matrix
    for (int i = 0; i < eigval_buf.size; i++)
    {
        *eigval_ptr++ = eigval[i];
    }

    return t_kernel;
}
