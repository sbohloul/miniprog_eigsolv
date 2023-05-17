#include <iostream>
#include <vector>
#include <blas.h>
#include <parameters.h>

int main(int argc, char **argv)
{
    std::cout << "dgemv for a*A*x + b*y" << std::endl;

    // parameters
    const int n = VECTOR_SIZE;
    const int N = MATRIX_SIZE;
    const CBLAS_ORDER order = CblasRowMajor;
    const CBLAS_TRANSPOSE trans = CblasNoTrans;

    std::cout
        << "n = " << n << std::endl;
    std::cout << "N = " << N << std::endl;

    // init matrix A
    std::vector<double> A(N);

    for (vec_size_db i = 0; i < A.size(); i++)
    {
        A[i] = static_cast<double>(i);
    }

    // print matrix A
    std::cout << "A = ";
    for (const auto &element : A)
    {
        std::cout << element << " ";
    }
    std::cout << std::endl;

    // inti vector x
    std::vector<double> x(n);

    for (vec_size_db i = 0; i < x.size(); i++)
    {
        x[i] = static_cast<double>(i);
    }

    // print vector x
    std::cout << "x = ";
    for (const auto &element : x)
    {
        std::cout << element << " ";
    }
    std::cout << std::endl;

    std::vector<double> y_ref(n);

    //  reference results
    for (vec_size_db i = 0; i < n; i++)
    {
        for (vec_size_db j = 0; j < n; j++)
        {
            y_ref[i] += A[i * n + j] * x[j];
        }
    }

    // print vector y_ref
    std::cout << "y_ref = ";
    for (const auto &element : y_ref)
    {
        std::cout << element << " ";
    }
    std::cout << std::endl;

    // call cblas
    std::vector<double> y(n);
    const int alpha = ONE_INT;
    const int beta = ONE_INT;
    const int lda = n;
    const int incx = ONE_INT;
    const int incy = ONE_INT;

    cblas_dgemv(order, trans, n, n, alpha, A.data(), lda, x.data(), incx, beta, y.data(), incy);

    // print vector y
    std::cout << "y = ";
    for (const auto &element : y)
    {
        std::cout << element << " ";
    }
    std::cout << std::endl;

    return 0;
}