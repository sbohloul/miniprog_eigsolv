#ifndef __MATRIX_V_HPP__
#define __MATRIX_V_HPP__

#include <iostream>
#include <vector>
#include <matrix_base.hpp>

/* =======================================
Matrix class using vector for data storage
======================================= */
template <typename T>
class MatrixV : public MatrixBase
{
public:
    MatrixV(size_t rows, size_t cols);
    ~MatrixV();
    T &operator()(size_t i, size_t j);
    const T &operator()(size_t i, size_t j) const;
    T *data();
    const T *data() const;

    template <typename U>
    friend std::ostream &operator<<(std::ostream &os, const MatrixV<U> &m);

protected:
    std::vector<T> data_;
};

// Constructor
template <typename T>
MatrixV<T>::MatrixV(size_t rows, size_t cols) : MatrixBase(rows, cols), data_(rows * cols){};

// Destructor
template <typename T>
MatrixV<T>::~MatrixV(){};

// Access elements
template <typename T>
T &MatrixV<T>::operator()(size_t i, size_t j)
{
    return data_[i * cols_ + j];
}

template <typename T>
const T &MatrixV<T>::operator()(size_t i, size_t j) const
{
    return data_[i * cols_ + j];
}

// Access data
template <typename T>
T *MatrixV<T>::data()
{
    return data_.empty() ? 0 : &data_.front();
}

template <typename T>
const T *MatrixV<T>::data() const
{
    return data_.empty() ? 0 : &data_.front();
}

// Print
template <typename T>
std::ostream &operator<<(std::ostream &os, const MatrixV<T> &m)
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

#endif /* __MATRIX_V_HPP__*/