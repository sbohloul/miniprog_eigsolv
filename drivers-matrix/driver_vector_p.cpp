#include <iostream>
#include <matrix.hpp>

using namespace std;

int main(int argc, char **argv)
{
    const int N = 5;

    VectorP<int> vwp(N);

    cout << vwp.data() << endl;

    return 0;
}