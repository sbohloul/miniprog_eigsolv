#ifndef __VECTOR_WITH_HPP__
#define __VECTOR_WITH_HPP__

#include <iostream>
#include <vector>

template <typename T>
class Vector
{
private:
    size_t rows_;
    std::vector<T> data_;

public:
    Vector(size_t rows) : rows_(rows), data_(rows) {}
    Vector(size_t rows, T init) : rows_(rows), data_(rows, init) {}
    Vector(size_t rows, T *init) : rows_(rows), data_(init, init + rows) {}

    ~Vector() {}

    size_t num_rows() const { return rows_; };

    /* Element access*/
    T &operator()(size_t i) { return data_[i]; };
    const T &operator()(size_t i) const { return data_[i]; };

    /* Pointer to data*/
    T *data() { return data_.empty() ? 0 : &data_.front(); }
    const T *data() const { return data_.empty() ? 0 : &data_.front(); }

    /* */
    // Vector<T> &operator=(const Vector<T> &other)
    // {

    //     std::cout << "this " << this << std::endl;
    //     std::cout << "*this " << this->operator()(2) << std::endl;

    //     if (&other == this)
    //     {
    //         return *this;
    //     }
    //     for (size_t i = 0; i < rows_; i++)
    //     {
    //         this->operator()(i) = other(i);
    //     }

    //     return *this;
    // }
};

/* Helper functions */
template <typename T>
Vector<T> operator+(const Vector<T> &a, const Vector<T> &b)
{
    Vector<T> c(a.num_rows());
    for (size_t i = 0; i < c.num_rows(); i++)
    {
        c(i) = a(i) + b(i);
    }
    return c;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, Vector<T> &v)
{
    for (size_t i = 0; i < v.num_rows(); i++)
    {
        os << " " << v(i);
    }
    os << std::endl;

    return os;
}

#endif /* __VECTOR_WITH_HPP__ */