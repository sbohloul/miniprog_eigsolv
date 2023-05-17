#include <iostream>
#include <vector>

using namespace std;

class Matrix
{

private:
    size_t nrows_;
    size_t ncols_;

public:
    double **data_;

    Matrix(size_t nrows, size_t ncols) : nrows_(nrows), ncols_(ncols)
    {
        data_ = new double *[nrows_];

        for (size_t i = 0; i < nrows_; i++)
        {
            data_[i] = new double[ncols_];
        }

        for (size_t i = 0; i < nrows_; i++)
        {
            for (size_t j = 0; j < ncols_; j++)
            {
                data_[i][j] = 0.0;
            }
        }
    }

    // get number of rows
    size_t getNumRows() const
    {
        return this->nrows_;
    }

    // get number of columns
    size_t getNumCols() const
    {
        return this->ncols_;
    }
};

// Get shape of a matrix
std::vector<size_t> shape(Matrix const &m)
{
    std::vector<size_t> shape_(2);
    shape_[0] = m.getNumRows();
    shape_[1] = m.getNumCols();
    return shape_;
}

// Get size of a matrix
size_t size(const Matrix &m)
{
    return m.getNumRows() * m.getNumCols();
}

// driver
int main(int argc, char **argv)
{
    cout << "Driver of matrix class." << endl;

    Matrix m(3, 4);

    std::cout << "size(m) = " << size(m) << std::endl;

    int M = shape(m)[0];
    int N = shape(m)[1];
    std::cout << "shape(m) = " << M << "," << N << std::endl;

    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            std::cout << "m[" << i << "," << j << "] = " << m.data_[i][j] << std::endl;
        }
    }

    return 0;
}