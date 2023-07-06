#include <iostream>
#include <vector>
#include <cassert>
#include <timer.hpp>
#include <omp.h>
#include <thread>
#include <mutex>

#define SIZE 100000000
#define BLOCK_SIZE 1000
#define THREAD_SIZE 16

static std::mutex barrier;

void sum_sequential(double &sum_val, double &time_val, std::vector<double> &v)
{
    Timer t;
    t.start();
    for (int i = 0; i < v.size(); i++)
    {
        sum_val += v[i];
    }
    t.stop();
    time_val = t.duration();
}

void sum_omp(double &sum_val, double &time_val, std::vector<double> &v)
{
    Timer t;

    t.start();
#pragma omp parallel for schedule(dynamic, BLOCK_SIZE) reduction(+ : sum_val)
    for (int i = 0; i < v.size(); i++)
    {
        sum_val += v[i];
    }
    t.stop();
    time_val = t.duration();
}

void psum_v1(double &psum, const int tid, const std::vector<double> &v, const int &block_size)
{
    double psum_local = 0.0;

    int istart = tid * block_size;
    int iend = (tid + 1) * block_size;
    if (iend > SIZE)
    {
        iend = SIZE;
    }

    for (int i = istart; i < iend; i++)
    {
        psum_local += v[i];
    }

    std::lock_guard<std::mutex> block_threads_until_finish_this_job(barrier);
    psum += psum_local;
}

void sum_thread_v1(double &sum_val, double &time_val, std::vector<double> &v)
{
    std::vector<std::thread> threads;
    std::vector<double> psum(THREAD_SIZE, 0.0);

    int block_size = (SIZE + THREAD_SIZE - 1) / THREAD_SIZE;

    Timer t;

    t.start();
    for (int tid = 0; tid < THREAD_SIZE; tid++)
    {
        threads.push_back(std::thread(psum_v1, std::ref(psum[tid]), tid, std::ref(v), std::ref(block_size)));
    }

    for (auto &thrd : threads)
    {
        thrd.join();
    }

    for (int i = 0; i < THREAD_SIZE; i++)
    {
        sum_val += psum[i];
    }

    t.stop();
    time_val = t.duration();
}

void psum_v2(double &sum_val, const int tid, const std::vector<double> &v, const int &block_size)
{
    double psum_local = 0.0;

    int istart = tid * block_size;
    int iend = (tid + 1) * block_size;
    if (iend > SIZE)
    {
        iend = SIZE;
    }

    for (int i = istart; i < iend; i++)
    {
        psum_local += v[i];
    }

    std::lock_guard<std::mutex> block_threads_until_finish_this_job(barrier);
    sum_val += psum_local;
}

void sum_thread_v2(double &sum_val, double &time_val, std::vector<double> &v)
{
    std::vector<std::thread> threads;

    int block_size = (SIZE + THREAD_SIZE - 1) / THREAD_SIZE;

    Timer t;

    t.start();
    for (int tid = 0; tid < THREAD_SIZE; tid++)
    {
        threads.push_back(std::thread(psum_v2, std::ref(sum_val), tid, std::ref(v), std::ref(block_size)));
    }

    for (auto &thrd : threads)
    {
        thrd.join();
    }

    t.stop();
    time_val = t.duration();
}

int main()
{
    std::vector<double> v(SIZE, 1.0);
    Timer timer;

    // sequential
    double s_seq = 0.0;
    double t_seq = 0.0;
    sum_sequential(s_seq, t_seq, v);

    assert(s_seq / SIZE == 1);
    std::cout << std::endl;
    std::cout << "sum_sequential(v) / N = " << s_seq / SIZE << std::endl;
    std::cout << "t_seq = " << t_seq << std::endl;

    // omp parallel for
    double s_omp = 0.0;
    double t_omp = 0.0;
    sum_omp(s_omp, t_omp, v);

    assert(s_omp / SIZE == 1);
    std::cout << std::endl;
    std::cout << "sum_omp(v) / N = " << s_omp / SIZE << std::endl;
    std::cout << "t_omp = " << t_omp << std::endl;

    // thread v1
    double s_thrd_v1 = 0.0;
    double t_thrd_v1 = 0.0;
    sum_thread_v1(s_thrd_v1, t_thrd_v1, v);

    assert(s_thrd_v1 / SIZE == 1);
    std::cout << std::endl;
    std::cout << "sum_thread_v1(v) / N = " << s_thrd_v1 / SIZE << std::endl;
    std::cout << "t_thrd_v1 = " << t_thrd_v1 << std::endl;

    // thread v2
    double s_thrd_v2 = 0.0;
    double t_thrd_v2 = 0.0;
    sum_thread_v2(s_thrd_v2, t_thrd_v2, v);

    assert(s_thrd_v2 / SIZE == 1);
    std::cout << std::endl;
    std::cout << "sum_thread_v2(v) / N = " << s_thrd_v2 / SIZE << std::endl;
    std::cout << "t_thrd_v2 = " << t_thrd_v2 << std::endl;

    return 0;
}