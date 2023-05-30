#include <iostream>
#include <matrix.hpp>

using namespace std;

extern "C"
{
    double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy);
}

int main(int argc, char **argv)
{

    int M = 1;
    int N = 4;

    MatrixV<double> v1(M, N);
    MatrixV<double> v2(M, N);

    // initialize vectors
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            v1(i, j) = 1.0;
            v2(i, j) = 2.0;
        }
    }

    // print vectors
    cout << v1;
    cout << v2;
    cout << v1.data() << endl;
    cout << v2.data() << endl;

    double d1 = 0.0;
    for (int i = 0; i < N; i++)
    {
        d1 += v1.data()[i] * v2.data()[i];
    }
    cout << "d1 = " << d1 << endl;

    double d2 = cblas_ddot(N, v1.data(), 1, v2.data(), 1);
    cout << "d2 = " << d2 << endl;
    return 0;
}