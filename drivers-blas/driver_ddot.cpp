#include <iostream>
#include <matrix.hpp>
#include <blas.h>

using namespace std;

// extern "C"
// {
//     double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy);
// }

int main(int argc, char **argv)
{

    const int N = 5;

    Vector<double> v1(N, 1.0);
    Vector<double> v2(N, 2.0);

    // print vectors
    cout << "v1 = " << endl;
    cout << v1;
    cout << "v2 = " << endl;
    cout << v2;

    // reference result
    double ref_result{0.0};
    for (int i = 0; i < N; i++)
    {
        ref_result += v1(i) * v2(i);
    }
    cout << "<v1|v2>" << endl;
    cout << "ref_result = " << ref_result << endl;

    // call blas
    double results = cblas_ddot(N, v1.data(), 1, v2.data(), 1);
    cout << "results = " << results << endl;

    return 0;
}