#include <iostream>
#include <thread>
#include <vector>

using namespace std;

void print_hello(int i)
{
    cout << "Hello, World! My input i is " << i << endl;
}

int main()
{
    int i = 1;
    thread t(print_hello, i);

    t.join();

    return 0;
}
