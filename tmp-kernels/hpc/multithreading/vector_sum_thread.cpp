#include <iostream>
#include <thread>
#include <cassert>
#include <vector>
#include <timer.hpp>

#define N 100000
#define BLOCKSIZE 10

// void partial_sum(double &psum, const std::vector<double> &v, const int tid, const int nblocks, const int nthreads)
// {

//     double local_psum = 0.0;
//     for (int iblk = tid; iblk < nblocks; iblk += nthreads)
//     {
//         int istart = iblk * BLOCKSIZE;
//         int iend = iblk * BLOCKSIZE + BLOCKSIZE;

//         for (int i = istart; i < iend; i++)
//         {
//             local_psum += v[i];
//         }
//     }
//     psum = local_psum;
// }

int main()
{
    // int nthreads = std::thread::hardware_concurrency();
    // std::vector<double> psum(nthreads, 0.0);
    // std::vector<std::thread> threads;

    // int nblocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;

    // for (int i = 0; i < nthreads; i++)
    // {
    //     threads.push_back(std::thread(partial_sum, std::ref(psum[i], v), i, nblocks, nthreads));
    // }

    return 0;
}