#include <iostream>
#include <matrix.hpp>

using namespace std;

extern "C"
{
    double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy);
}

int main(int argc, char **argv)
{

    int N = 4;

    VectorV<double> v1(N);
    VectorV<double> v2(N);

    // initialize vectors
    for (int i = 0; i < N; i++)
    {
        v1(i) = 1.0;
    }

    for (int i = 0; i < size(v2); i++)
    {
        v2.data()[i] = 2.0;
    }

    // print vectors
    cout << v1;
    cout << v2;
    cout << v1.data() << endl;
    cout << v2.data() << endl;

    // reference result
    double d1 = 0.0;
    for (int i = 0; i < N; i++)
    {
        d1 += v1.data()[i] * v2.data()[i];
    }
    cout << "d1 = " << d1 << endl;

    // call blas
    double d2 = cblas_ddot(N, v1.data(), 1, v2.data(), 1);
    cout << "d2 = " << d2 << endl;
    return 0;
}