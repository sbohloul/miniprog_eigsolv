#include <iostream>
#include <vector>
#include <numeric>
#include <timer.hpp>

using namespace std;

constexpr size_t nelem = 10000;

int main(int argc, char **argv)
{
    cout << endl;
    cout << "Sum of array: explicit" << endl;

    vector<double> v(nelem, 1.0);

    // Explicit loop
    Timer time;

    double sum_loop = 0.0;
    time.start();
    for (size_t i = 0; i < nelem; i++)
    {
        sum_loop += v[i];
    }
    time.stop();
    double t_loop = time.duration();

    // accumulate
    time.start();
    double sum_accm = accumulate(v.begin(), v.end(), 0.0);
    time.stop();
    double t_accm = time.duration();

    cout << "sum_loop = " << sum_loop << endl;
    cout << "sum_accm = " << sum_accm << endl;

    cout << "t_loop = " << t_loop << endl;
    cout << "t_accm = " << t_accm << endl;

    return 0;
}