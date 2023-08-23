#include <iostream>
#include "time_scalapack_kernels.hpp"
#include "array_helper.hpp"

// prog.x niter m n mb nb nprow npcol
int main(int argc, char **argv)
{

    if (argc < 8)
    {
        throw std::runtime_error("***WRONG INPUTS: prog.x niter m n mb nb nprow npcol");
    }

    int niter = atoi(argv[1]);
    int m = atoi(argv[2]);
    int n = atoi(argv[3]);
    int mb = atoi(argv[4]);
    int nb = atoi(argv[5]);
    int nprow = atoi(argv[6]);
    int npcol = atoi(argv[7]);

    int myrank, nprocs;
    Cblacs_pinfo(&myrank, &nprocs);

    // initialize global vectors
    std::vector<double> a(m * n);
    std::vector<double> b(m * n);
    std::vector<double> c(m * n);

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            a[j * m + i] = static_cast<double>(i);
            b[j * m + i] = static_cast<double>(j);
            c[j * m + i] = static_cast<double>(0);
        }
    }

    if (myrank == 0)
    {
        std::cout << "a: " << std::endl;
        print_array(a, m, n, m);
        std::cout << "b: " << std::endl;
        print_array(b, m, n, m);
        std::cout << "c: " << std::endl;
        print_array(c, m, n, m);
    }

    double t_kernel = time_scalapack_pdgemm(niter, nprow, npcol, a, b, c, m, n, mb, nb);

    if (myrank == 0)
    {
        std::cout << "t_kernel = " << t_kernel << std::endl;
        std::cout << "c: " << std::endl;
        print_array(c, m, n, m);
    }

    Cblacs_exit(0);

    return 0;
}