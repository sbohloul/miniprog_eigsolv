#ifndef _BLACS_H_
#define _BLACS_H

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

    // Initialization
    void Cblacs_get(int icontxt, int what, int *val);
    void Cblacs_gridinit(int *icontxt, char *order, int nprow, int npcol);

    // Information
    void Cblacs_gridinfo(int icontxt, int *nprow, int *npcol, int *myrow, int *mycol);
    void Cblacs_pinfo(int *rank, int *nprocs);
    void blacs_pcoord(const int *ConTxt, const int *nodenum, int *prow, int *pcol);

    // Destruction
    void Cblacs_gridexit(int icontxt);
    void Cblacs_exit(int doneflag);

    //
    void Cblacs_barrier(int icontxt, char *scope);

    //
    int numroc(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);

    void descinit(int *desc, const int *m, const int *n,
                  const int *mb, const int *nb, const int *irsrc,
                  const int *icsrc, const int *ictxt, const int *lld,
                  int *info);

    // send/recieve
    void igerv2d(const int *ConTxt, const int *m, const int *n, int *A, const int *lda, const int *rsrc, const int *csrc);
    void igesd2d(const int *ConTxt, const int *m, const int *n, int *A, const int *lda, const int *rdest, const int *cdest);

    // redist
    void Cpsgemr2d(int m, int n,
                   const float *a, int ia, int ja, const int *desca,
                   float *b, int ib, int jb, const int *descb,
                   int ictxt);
    void Cpdgemr2d(int m, int n,
                   const double *a, int ia, int ja, const int *desca,
                   double *b, int ib, int jb, const int *descb,
                   int ictxt);
    // void Cpcgemr2d(int m, int n,
    //                const MKL_Complex8 *a, int ia, int ja, const int *desca,
    //                MKL_Complex8 *b, int ib, int jb, const int *descb,
    //                int ictxt);
    // void Cpzgemr2d(int m, int n,
    //                const MKL_Complex16 *a, int ia, int ja, const int *desca,
    //                MKL_Complex16 *b, int ib, int jb, const int *descb,
    //                int ictxt);
    void Cpigemr2d(int m, int n,
                   const int *a, int ia, int ja, const int *desca,
                   int *b, int ib, int jb, const int *descb,
                   int ictxt);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif