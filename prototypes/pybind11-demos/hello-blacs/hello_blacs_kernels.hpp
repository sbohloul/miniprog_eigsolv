#ifndef _HELLO_BLACS_KERNELS_HPP_
#define _HELLO_BLACS_KERNELS_HPP_

extern "C"
{
    void Cblacs_get(int icontxt, int what, int *val);
    void Cblacs_gridinit(int *icontxt, char *order, int nprow, int npcol);

    void Cblacs_pinfo(int *rank, int *nprocs);
    void Cblacs_gridinfo(int icontxt, int *nprow, int *npcol, int *myrow, int *mycol);

    void Cblacs_gridexit(int icontxt);
    void Cblacs_exit(int doneflag);

    void blacs_barrier(const int *ConTxt, const char *scope);
    int numroc(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
}

void print_info(const int nprow, const int npcol, const int mb, const int nb, const int m, const int n);

#endif