#ifndef __MATRIX_P_HPP__
#define __MATRIX_P_HPP__

#include <iostream>
#include <matrix_base.hpp>

/* =======================
Matrix class using pointer
======================= */
template <typename T>
class MatrixP : public MatrixBase
{
public:
    MatrixP(size_t rows, size_t cols);
    ~MatrixP();
    T &operator()(size_t i, size_t j);
    const T &operator()(size_t i, size_t j) const;
    T *data();
    const T *data() const;

    template <typename U>
    friend std::ostream &operator<<(std::ostream &os, const MatrixP<U> &m);

protected:
    T *data_;
};

// Constructor
template <typename T>
MatrixP<T>::MatrixP(size_t rows, size_t cols) : MatrixBase(rows, cols)
{
    data_ = new T[rows_ * cols_];
}

// Destructure
template <typename T>
MatrixP<T>::~MatrixP()
{
    delete[] data_;
}

// Element access
template <typename T>
T &MatrixP<T>::operator()(size_t i, size_t j)
{
    return data_[i * cols_ + j];
}

template <typename T>
const T &MatrixP<T>::operator()(size_t i, size_t j) const
{
    return data_[i * cols_ + j];
}

// Data access
template <typename T>
T *MatrixP<T>::data()
{
    return data_ == nullptr ? 0 : data_;
}

template <typename T>
const T *MatrixP<T>::data() const
{
    return data_ == nullptr ? 0 : data_;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const MatrixP<T> &m)
{
    for (size_t i = 0; i < m.rows_; i++)
    {
        for (size_t j = 0; j < m.cols_; j++)
        {
            os << " " << m(i, j);
        }
        os << std::endl;
    }
    os << std::endl;

    return os;
}

#endif /*__MATRIX_P_HPP__*/