#include <iostream>
#include <matrix.hpp>

using namespace std;

int main(int argc, char **argv)
{

    int M1{3};
    VectorV<int> vwv_int(M1);

    for (int i = 0; i < size(vwv_int); i++)
    {
        cout << i << endl;
        vwv_int(i) = i;
    }

    int M2{4};
    VectorV<double> vwv_dbl(M2);
    for (int i = 0; i < size(vwv_dbl); i++)
    {
        vwv_dbl(i) = static_cast<double>(i) / 10.0;
    }

    cout << vwv_int;
    cout << vwv_dbl;

    int *data = vwv_int.data();
    for (int i = 0; i < size(vwv_int); i++)
    {
        data[i] += 1;
    }
    cout << vwv_int;

    return 0;
}