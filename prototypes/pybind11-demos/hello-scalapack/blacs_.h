#ifndef _BLACS_H_
#define _BLACS_H_

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

#ifndef MKL_INT
#define MKL_INT int
#endif

#ifdef USE_MKL
#define numroc numroc
#define descinit descinit
#define pdgemm pdgemm
#else
#define numroc numroc_
#define descinit descinit_
#define pdgemm pdgemm_
#endif

    // Initialization
    void Cblacs_get(int icontxt, int what, int *val);
    void Cblacs_gridinit(int *icontxt, char *order, int nprow, int npcol);

    // Information
    void Cblacs_gridinfo(int icontxt, int *nprow, int *npcol, int *myrow, int *mycol);
    void blacs_gridinfo(const MKL_INT *ConTxt, MKL_INT *nprow, MKL_INT *npcol, MKL_INT *myrow, MKL_INT *mycol);

    void Cblacs_pinfo(int *rank, int *nprocs);
    void blacs_pcoord(const int *ConTxt, const int *nodenum, int *prow, int *pcol);

    // Destruction
    void Cblacs_gridexit(int icontxt);
    void Cblacs_exit(int doneflag);
    void blacs_gridexit(int *ConTxt);
    void blacs_exit(const int *notDone);

    //
    void Cblacs_barrier(int icontxt, char *scope);
    void blacs_barrier(const int *ConTxt, const char *scope);

    // local & global conversions
    int numroc(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
    // int indxl2g_(int indxloc, int nb, int iproc, int isrcproc, int nprocs);
    int indxg2l_(int indxglob, int nb, int iproc, int isrcproc, int nprocs);
    int indxg2p_(int indxglob, int nb, int iproc, int isrcproc, int nprocs);
    int indxl2g_(int *indxloc, int *nb, int *iproc, int *isrcproc, int *nprocs);

    // array desc
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

    // PBLAS
    void pdgemm(const char *transa, const char *transb,
                const int *m, const int *n, const int *k,
                const double *alpha,
                const double *a, const int *ia, const int *ja, const int *desca,
                const double *b, const int *ib, const int *jb, const int *descb,
                const double *beta,
                double *c, const int *ic, const int *jc, const int *descc);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif