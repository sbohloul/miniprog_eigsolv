#ifndef __MATRIX_WITH_VECTOR_HPP__
#define __MATRIX_WITH_VECTOR_HPP__

#include <iostream>
#include <vector>

template <typename T>
class Matrix
{
public:
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), data_(rows * cols) {}
    Matrix(size_t rows, size_t cols, T init) : rows_(rows), cols_(cols), data_(rows * cols, init) {}
    ~Matrix() {}

    size_t num_rows() const { return rows_; }
    size_t num_cols() const { return cols_; }

    /* Element access */
    T &operator()(size_t i, size_t j) { return data_[i * cols_ + j]; }
    const T &operator()(size_t i, size_t j) const { return data_[i * cols_ + j]; }

    /* Pointer to data */
    T *data() { return data_.empty() ? 0 : &data_.front(); };
    const T *data() const { return data_.empty() ? 0 : &data_.front(); };

private:
    size_t rows_;
    size_t cols_;
    std::vector<T> data_;
};

/* ================ */
/* Helper functions */
/* ================ */

/* Math operations */
template <typename T>
Matrix<T> operator+(const Matrix<T> &a, const Matrix<T> &b)
{
    Matrix<T> c(a.num_rows(), a.num_cols());
    for (size_t i = 0; i < c.num_rows(); i++)
    {
        for (size_t j = 0; j < c.num_cols(); j++)
        {
            c(i, j) = a(i, j) + b(i, j);
        }
    }
    return c;
}

/* print */
template <typename T>
std::ostream &operator<<(std::ostream &os, const Matrix<T> &m)
{
    for (size_t i = 0; i < m.num_rows(); i++)
    {
        for (size_t j = 0; j < m.num_cols(); j++)
        {
            os << " " << m(i, j);
        }
        os << std::endl;
    }
    os << std::endl;

    return os;
}

#endif /* __MATRIX_WITH_VECTOR_HPP__ */