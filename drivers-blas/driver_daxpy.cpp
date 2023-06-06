#include <iostream>
#include <matrix.hpp>
#include <blas.h>

using namespace std;

// extern "C"
// {
//     void cblas_daxpy(const int n, const double a, const double *x, const int incx, const double *y, const int incy);
// };

int main(int argc, char **argv)
{
    const int N = 5;

    Vector<double> v1(N, 1.0);
    Vector<double> v2(N, 2.0);
    cout << "v1 = " << endl;
    cout << v1;
    cout << "v2 = " << endl;
    cout << v2;

    const double a = 3.0;
    cblas_daxpy(N, a, v1.data(), 1, v2.data(), 1);

    cout << "a = " << a << endl;
    cout << "a * v1 + v2 = " << endl;
    cout << v2;

    return 0;
}