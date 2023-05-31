#include <iostream>
#include <matrix.hpp>

using namespace std;

int main(int argc, char **argv)
{

    const int M = 2;
    const int N = 3;
    MatrixP<double> mwp(M, N);

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            mwp(i, j) = static_cast<double>(i * N + j);
        }
    }

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << mwp(i, j) << endl;
        }
    }

    cout << mwp.data() << endl;
    for (int i = 0; i < M * N; i++)
    {
        cout << mwp.data()[i] << endl;
        cout << &mwp.data()[i] << endl;
        cout << mwp.data() + i << endl;
    }

    cout << mwp;

    return 0;
}