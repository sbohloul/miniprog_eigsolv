#include <iostream>
#include <vector>
#include <omp.h>
#include <cassert>
#include <timer.hpp>
#include <thread>

using namespace std;

#define block_size 100
#define N 10000000
#define NTHRDS 4

// brute-force
double sum_naive(vector<double> &v)
{
    double sum = 0.0;

    for (int i = 0; i < v.size(); i++)
    {
        sum += v[i];
    }

    return sum;
}

// omp manual work distribution
double sum_omp_naive(vector<double> &v)
{
    double sum = 0.0;
    int nthreads;
    int nblocks;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
#pragma omp single
        {
            nthreads = omp_get_num_threads();
            nblocks = (v.size() + block_size - 1) / block_size;
        }

        double psum = 0.0;
        for (int i = tid; i < nblocks; i += nthreads)
        {
            int ind1 = i * block_size;
            int ind2 = (i + 1) * block_size;
            for (int j = ind1; j < ind2; j++)
            {
                psum += v[j];
            }
        }

#pragma omp critical
        {
            sum += psum;
        }
    }

    return sum;
}

// omp parallel for with dynamic scheduling
double sum_omp(vector<double> &v)
{
    double sum = 0.0;

#pragma omp parallel for schedule(dynamic, block_size) reduction(+ : sum)
    for (int i = 0; i < v.size(); i++)
    {
        sum += v[i];
    }

    return sum;
}

// thread

void partial_sum(double &psum, vector<double> &v, int tid, int nblocks, int nthreads)
{
    double psum_local = 0.0; // to avoid false sharing of psum
    for (int iblk = tid; iblk < nblocks; iblk += nthreads)
    {
        int istart = iblk * block_size;
        for (int i = istart; i < istart + block_size; i++)
        {
            psum_local += v[i];
        }
    }

    psum = psum_local;
}

double sum_thread(vector<double> &v)
{
    vector<std::thread> threads;

    // int nthreads = thread::hardware_concurrency();

    int pad = 64 / sizeof(double);
    vector<double> psum(NTHRDS * pad, 0.0);

    int nblocks = (v.size() + block_size - 1) / block_size;

    for (int i = 0; i < NTHRDS; i++)
    {
        threads.push_back(std::thread(partial_sum, std::ref(psum[i * NTHRDS]), std::ref(v), i, nblocks, NTHRDS));
    }

    for (auto &t : threads)
    {
        t.join();
    }

    // for (int i = 0; i < psum.size(); i++)
    // {
    //     cout << "psum = " << psum[i] << endl;
    // }

    double sum = 0.0;
    for (int i = 0; i < NTHRDS; i++)
    {
        sum += psum[i * NTHRDS];
    }

    return sum;
}

int main()
{
    vector<double> v(N, 1.0);

    Timer timer;

    timer.start();
    double naive_val = sum_naive(v);
    timer.stop();
    assert(naive_val / N == 1.0);
    cout << "sum_naive(v) = " << naive_val << endl;
    cout << "t = " << timer.duration() << endl;

    timer.start();
    double omp_naive_val = sum_omp_naive(v);
    timer.stop();
    assert(omp_naive_val / N == 1.0);
    cout << "sum_omp_naive(v) = " << omp_naive_val << endl;
    cout << "t = " << timer.duration() << endl;

    timer.start();
    double omp_val = sum_omp(v);
    timer.stop();
    assert(omp_val / N == 1.0);
    cout << "sum_omp(v) = " << omp_val << endl;
    cout << "t = " << timer.duration() << endl;

    timer.start();
    double thread_val = sum_thread(v);
    timer.stop();
    assert(thread_val / N == 1.0);
    cout << "sum_thread(v) = " << thread_val << endl;
    cout << "t = " << timer.duration() << endl;

    return 0;
}