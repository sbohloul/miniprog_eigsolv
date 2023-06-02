#ifndef __VECTOR_P_HPP__
#define __VECTOR_P_HPP__

#include <iostream>
#include <matrix_p.hpp>

/* ========================================
Vector class using pointer for data storage
======================================== */
template <typename T>
class VectorP : public MatrixP<T>
{
public:
    VectorP(size_t cols);
    ~VectorP();
    T &operator()(size_t j);
    const T &operator()(size_t j) const;
};

// Constructor
template <typename T>
VectorP<T>::VectorP(size_t cols) : MatrixP<T>(1, cols){};

// Destructor
template <typename T>
VectorP<T>::~VectorP(){};

// Access elementsVectorP<T>::
template <typename T>
T &VectorP<T>::operator()(size_t j)
{
    return MatrixP<T>::operator()(0, j);
}

template <typename T>
const T &VectorP<T>::operator()(size_t j) const
{
    return MatrixP<T>::operator()(0, j);
}

#endif /*__VECTOR_P_HVectorP<T>::PP__*/