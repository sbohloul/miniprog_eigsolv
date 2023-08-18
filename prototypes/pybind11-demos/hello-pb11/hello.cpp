#include <iostream>
#include "hello.hpp"

int add(int i, int j)
{
    std::cout << "&i: " << &i << std::endl;
    std::cout << "&j: " << &j << std::endl;
    return i + j;
}
