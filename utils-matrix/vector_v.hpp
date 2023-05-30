#ifndef __VECTOR_V_HPP__
#define __VECTOR_V_HPP__

#include <iostream>
#include <matrix_v.hpp>

/* =======================================
Vector class using vector for data storage
======================================= */
template <typename T>
class VectorV : public MatrixV<T>
{
public:
    VectorV(size_t cols);
    ~VectorV();

    T &operator()(size_t j);
    const T &operator()(size_t j) const;
};

// Constructor
template <typename T>
VectorV<T>::VectorV(size_t cols) : MatrixV<T>(1, cols){};

// Destructor
template <typename T>
VectorV<T>::~VectorV(){};

// Access elements
template <typename T>
T &VectorV<T>::operator()(size_t j)
{
    return MatrixV<T>::operator()(0, j);
}

template <typename T>
const T &VectorV<T>::operator()(size_t j) const
{
    return MatrixV<T>::operator()(0, j);
}

#endif /*__VECTOR_V_HPP__*/