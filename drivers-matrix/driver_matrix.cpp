#include <iostream>
#include <matrixutils.hpp>

using namespace std;

int main()
{
    int M = 2;
    int N = 3;

    Matrix<int> m(M, N);

    cout << "m.data() " << m.data() << endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            m(i, j) = i * N + j;
        }
    }
    cout << m;

    Matrix<int> m2 = m + m;
    cout << m2;

    Matrix<int> m3(M, N, 1);
    cout << m3;

    return 0;
}