#include "pb11_mpistats_scalapack_kernels.hpp"

// ======
// PDSYEV
// ======
void pb11_mpistats_scalapack_pdsyev(int mpi_region,
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
        std::cout << "Inside pb11_mpistats_scalapack_pdsyev" << std::endl;
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

    mpistats_scalapack_pdsyev(mpi_region, nprow, npcol, a, eigval, eigvec, m, mb, nb);

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
}
