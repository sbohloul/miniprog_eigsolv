#include <iostream>
#include "array_helper.hpp"

void print_array(std::vector<double> &a, int m, int n, int ld)
{
    for (int i = 0; i < m; i++)
    {
        std::cout << "[";
        for (int j = 0; j < n; j++)
        {
            std::cout << a[j * ld + i];
            if (j != n - 1)
            {
                std::cout << ", ";
            }
            else
            {
                std::cout << "]" << std::endl;
            }
        }
    }
}

double i_dot_j(int i, int j)
{
    return i + j / 10.0;
}
