#ifndef _BLAS_H_
#define _BLAS_H_

#ifdef USE_MKL
#include <mkl.h>
#elif defined(USE_OPENBLAS)
#include <cblas.h>
#else
#ifdef __cplusplus
extern "C"
{
    /* Assume C decleratio for C++ */
#endif /* __cplusplus*/

    /* */
    enum CBLAS_ORDER
    {
        CblasRowMajor = 101,
        CblasColMajor = 102
    };

    enum CBLAS_TRANSPOSE
    {
        CblasNoTrans = 111,
        CblasTrans = 112,
        CblasConjTrans = 113,
        CblasConjNoTrans = 114
    };

    typedef CBLAS_ORDER CBLAS_ORDER;
    typedef CBLAS_TRANSPOSE CBLAS_TRANSPOSE;

    /* */
    double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy);

    void cblas_dcopy(const int n, const double *x, const int incx, double *y, const int incy);

    void cblas_dscal(const int n, const double alpha, double *x, const int inxc);

    void cblas_dgemv(const CBLAS_ORDER order,
                     const CBLAS_TRANSPOSE trans,
                     const int m, const int n,
                     const double alpha,
                     const double *A, const int lda,
                     const double *x, const int incx,
                     const double beta,
                     double *y, const int incy);

    void cblas_daxpy(const int n,
                     const double a,
                     const double *x, const int incx,
                     const double *y, const int incy);

#ifdef __cplusplus
}
#endif /* __cplusplus*/

#endif
#endif