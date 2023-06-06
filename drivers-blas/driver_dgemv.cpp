#include <iostream>
#include <matrix.hpp>
#include <blas.h>

using namespace std;

// enum CBLAS_ORDER
// {
//     CblasRowMajor = 101,
//     CblasColMajor = 102
// };

// enum CBLAS_TRANSPOSE
// {
//     CblasNoTrans = 111,
//     CblasTrans = 112,
//     CblasConjTrans = 113,
//     CblasConjNoTrans = 114
// };

// typedef CBLAS_ORDER CBLAS_ORDER;
// typedef CBLAS_TRANSPOSE CBLAS_TRANSPOSE;

// extern "C"
// {
//     void cblas_dgemv(const int order,
//                      const int trans,
//                      const int m, const int n,
//                      const double alpha,
//                      const double *A, const int lda,
//                      const double *x, const int incx,
//                      const double beta,
//                      double *y, const int incy);
// }

int main(int argc, char **argv)
{
    const int M = 5;
    const int N = 5;

    Matrix<double> m1(M, N, 2.0);
    Vector<double> v1(N, 1.0);

    cout << "m1 = " << endl;
    cout << m1;
    cout << "v1 = " << endl;
    cout << v1;

    Vector<double> v2(N);
    const double alpha = 1;
    const double beta = 1;
    const int lda = N;
    const int incx = 1;
    const int incy = 1;
    const CBLAS_ORDER order = CblasRowMajor;
    const CBLAS_TRANSPOSE trans = CblasNoTrans;

    cout << " y := (alpah * A * v) + (beta * y)" << endl;
    cout << "alpah = " << alpha << endl;
    cout << "beta = " << beta << endl;

    cblas_dgemv(order, trans, M, N, alpha, m1.data(), lda, v1.data(), incx, beta, v2.data(), incy);

    cout << "v2 = " << endl;
    cout << v2;
    return 0;
}