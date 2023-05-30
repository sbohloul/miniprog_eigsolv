#ifndef __MATRIX_BASE_HPP__
#define __MATRIX_BASE_HPP__

#include <iostream>
#include <vector>

class MatrixBase
{
public:
    MatrixBase(size_t nrows, size_t ncols);
    virtual ~MatrixBase();
    size_t getRows() const;
    size_t getCols() const;
    friend std::vector<size_t> shape(const MatrixBase &m);
    friend size_t size(const MatrixBase &m);

protected:
    size_t rows_;
    size_t cols_;
};

MatrixBase::MatrixBase(size_t rows, size_t cols) : rows_(rows), cols_(cols) {}

MatrixBase::~MatrixBase() {}

size_t MatrixBase::getRows() const
{
    return rows_;
}

size_t MatrixBase::getCols() const
{
    return cols_;
}

std::vector<size_t> shape(const MatrixBase &m)
{
    std::vector<size_t> shape_(2);
    shape_[0] = m.rows_;
    shape_[1] = m.cols_;

    return shape_;
}

size_t size(const MatrixBase &m)
{
    return m.rows_ * m.cols_;
}

#endif