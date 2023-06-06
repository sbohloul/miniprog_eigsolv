#include <iostream>
#include <matrixutils.hpp>

using namespace std;

int main()
{
    int N = 5;
    Vector<int> v(N);

    cout << "num_rows()" << endl;
    cout << v.num_rows() << endl;

    for (int i = 0; i < N; i++)
    {
        v(i) = i;
    }

    cout << "v(2)" << endl;
    cout << v(2) << endl;

    cout << v;
    cout << "&v " << &v << endl;

    Vector<int> v2(N);

    for (int i = 0; i < N; i++)
    {
        v2.data()[i] = 1;
    }
    cout << v2;
    cout << "&v2 " << &v2 << endl;

    v2 = v;

    cout << "&v2 " << &v2 << endl;
    cout << v2;

    v2 = v + v;
    cout << "&v2 " << &v2 << endl;
    cout << v2;

    v2 = v + v + v2;
    cout << "&v2 " << &v2 << endl;
    cout << v2;

    return 0;
}