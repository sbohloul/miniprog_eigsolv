#include <iostream>
#include <timer.hpp>
#include <vector>
#include <cassert>
#include <thread>
#include <mutex>

#define SIZE 100000000
#define BLOCKSIZE 1024
#define NTHREADS 8

std::mutex sum_mutex;

void dot_sequential(double &dot_val, double &time_val, std::vector<double> v1, std::vector<double> v2)
{
    Timer t;

    t.start();
    for (int i = 0; i < v1.size(); i++)
    {
        dot_val += v1[i] * v2[i];
    }
    t.stop();
    time_val = t.duration();
}

void dot_omp(double &dot_val, double &time_val, std::vector<double> v1, std::vector<double> v2)
{
    Timer t;

    t.start();
#pragma omp parallel for schedule(dynamic, BLOCKSIZE) reduction(+ : dot_val)
    for (int i = 0; i < v1.size(); i++)
    {
        dot_val += v1[i] * v2[i];
    }
    t.stop();
    time_val = t.duration();
}

void pdot(double &dot_val, int const tid, int const &nblocks, const std::vector<double> &v1, const std::vector<double> &v2)
{
    double dot_local = 0.0;

    for (int iblk = tid; iblk < nblocks; iblk += NTHREADS)
    {
        int idx_begin = iblk * BLOCKSIZE;
        int idx_end = (iblk + 1) * BLOCKSIZE;
        if (idx_end > SIZE)
        {
            idx_end = SIZE;
        }

        for (int i = idx_begin; i < idx_end; i++)
        {
            dot_local += v1[i] * v2[i];
        }
    }

    // lock
    std::lock_guard<std::mutex> sum_lock(sum_mutex);
    dot_val += dot_local;
}

void dot_thread(double &dot_val, double &time_val, std::vector<double> v1, std::vector<double> v2)
{
    std::vector<std::thread> threads;
    int nblocks = (SIZE + BLOCKSIZE - 1) / BLOCKSIZE;

    Timer t;
    t.start();
    for (int tid = 0; tid < NTHREADS; tid++)
    {
        threads.push_back(std::thread(pdot, std::ref(dot_val), tid, std::ref(nblocks), std::ref(v1), std::ref(v2)));
    }

    for (auto &thd : threads)
    {
        thd.join();
    }
    t.stop();
    time_val = t.duration();
}

int main()
{
    double val1 = 1.0;
    double val2 = 2.0;
    double ref_val = val1 * val2;

    std::vector<double> v1(SIZE, val1);
    std::vector<double> v2(SIZE, val2);

    // sequential
    double r_seq = 0.0;
    double t_seq = 0.0;
    dot_sequential(r_seq, t_seq, v1, v2);

    assert(r_seq / SIZE == ref_val);
    std::cout << std::endl;
    std::cout << "r_seq / N = " << r_seq / SIZE << std::endl;
    std::cout << "t_seq = " << t_seq << std::endl;

    // omp
    double r_omp = 0.0;
    double t_omp = 0.0;
    dot_omp(r_omp, t_omp, v1, v2);

    assert(r_omp / SIZE == ref_val);
    std::cout << std::endl;
    std::cout << "r_omp / N = " << r_omp / SIZE << std::endl;
    std::cout << "t_omp = " << t_omp << std::endl;

    // thread
    double r_thrd = 0.0;
    double t_thrd = 0.0;
    dot_thread(r_thrd, t_thrd, v1, v2);

    assert(r_thrd / SIZE == ref_val);
    std::cout << std::endl;
    std::cout << "r_thrd / N = " << r_thrd / SIZE << std::endl;
    std::cout << "t_thrd = " << t_thrd << std::endl;

    return 0;
}