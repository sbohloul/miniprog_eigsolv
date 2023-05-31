#include <iostream>
#include <matrix.hpp>

#define EXIT_SUCCESS 0
#define EXIT_ERROR_1 1

using namespace std;

int main(int argc, char **argv)
{

     // base
     cout << "========" << endl;
     MatrixBase m_base(2, 3);
     cout << "m_base" << endl;
     cout << "getRows"
          << "\t" << m_base.getRows() << endl;
     cout << "getCols"
          << "\t" << m_base.getCols() << endl;

     // int
     cout << "========" << endl;
     MatrixV<int> m_vect_int(3, 4);
     cout << "m_vect_int" << endl;
     cout << "getRows"
          << "\t" << m_vect_int.getRows() << endl;
     cout << "getCols"
          << "\t" << m_vect_int.getCols() << endl;

     std::vector<size_t> shp_int = shape(m_vect_int);
     cout << "shape: " << shp_int[0] << "," << shp_int[1] << endl;

     cout << "Access" << endl;
     for (size_t i = 0; i < shp_int[0]; i++)
     {
          for (size_t j = 0; j < shp_int[1]; j++)
          {
               m_vect_int(i, j) = i * shp_int[1] + j;
          }
     }
     cout << "m_vect_int(1, 2) " << m_vect_int(1, 2) << endl;

     cout << "const access" << endl;
     const int a = m_vect_int(0, 1);
     cout << a << endl;

     // double
     cout << "========" << endl;
     MatrixV<double> m_vect_dbl(5, 6);
     cout << "m_vect_dbl" << endl;
     cout << "getRows"
          << "\t" << m_vect_dbl.getRows() << endl;
     cout << "getCols"
          << "\t" << m_vect_dbl.getCols() << endl;

     std::vector<size_t> shp_dbl = shape(m_vect_dbl);
     cout << "shape: " << shp_dbl[0] << "," << shp_dbl[1] << endl;

     cout << "Access" << endl;
     for (size_t i = 0; i < shp_dbl[0]; i++)
     {
          for (size_t j = 0; j < shp_dbl[1]; j++)
          {
               m_vect_dbl(i, j) = static_cast<double>(i) + static_cast<double>(j) / 10.0;
          }
     }
     cout << "m_vect_dbl(1, 2) " << m_vect_dbl(1, 2) << endl;
     cout << "const access" << endl;
     const int b = m_vect_dbl(0, 1);
     cout << b << endl;

     cout << "========" << endl;
     cout << "print" << endl;
     cout << m_vect_int;
     cout << m_vect_dbl;

     cout << "get data int" << endl;
     int *data_int = m_vect_int.data();
     for (int i = 0; i < shp_int[0] * shp_int[1]; i++)
     {
          cout << data_int + i << endl;
          cout << *(data_int + i) << endl;
     }

     cout << "get data dbl" << endl;
     double *data_dbl = m_vect_dbl.data();
     for (int i = 0; i < shp_dbl[0] * shp_dbl[1]; i++)
     {
          cout << data_dbl + i << endl;
          cout << *(data_dbl + i) << endl;
     }

     return EXIT_SUCCESS;
}

// int main(int argc, char **argv)
// {
//     std::cout << "Driver of matrix class" << std::endl;

//     // size_t
//     std::size_t x;
//     std::cout << x << std::endl;
//     std::cout << sizeof(x) << std::endl;

//     // p and p of p
//     double *p1;
//     double **p2;
//     p1 = new double(10.0);
//     p2 = new double *;
//     p2 = &p1;
//     std::cout << p1 << std::endl;
//     std::cout << *p1 << std::endl;
//     std::cout << p2 << std::endl;
//     std::cout << *p2 << std::endl;
//     std::cout << "=====" << std::endl;

//     // basic constructors
//     double x1 = double();
//     std::cout << x1 << std::endl;

//     std::vector<int> v(3, int());
//     for (auto &element : v)
//     {
//         std::cout << element << std::endl;
//     }

//     // Construct matrix
//     MatrixPointer m_p(2, 3);
//     MatrixPointerToPointer m_p2p(2, 3);
//     MatrixV m_v(2, 3);

//     // rows and cols
//     std::cout << "nrows"
//               << "\t"
//               << "ncols" << std::endl;
//     std::cout << m_p.getNumRows() << "\t" << m_p.getNumCols() << std::endl;
//     std::cout << m_p2p.getNumRows() << "\t" << m_p2p.getNumCols() << std::endl;
//     std::cout << m_v.getNumRows() << "\t" << m_v.getNumCols() << std::endl;

//     // shape
//     std::cout << "shape" << std::endl;
//     std::cout << shape(m_p)[0] << "\t" << shape(m_p)[1] << std::endl;
//     std::cout << shape(m_p2p)[0] << "\t" << shape(m_p2p)[1] << std::endl;
//     std::cout << shape(m_v)[0] << "\t" << shape(m_v)[1] << std::endl;

//     // data_
//     double *data_p = m_p.getData();
//     double **data_p2p = m_p2p.getData();
//     std::vector<double> data_v = m_v.getData();

//     std::cout << "data_p" << std::endl;
//     std::cout << data_p << std::endl;
//     std::cout << data_p[0] << std::endl;
//     std::cout << data_p[1] << std::endl;
//     std::cout << "data_p2p" << std::endl;
//     std::cout << data_p2p << std::endl;
//     std::cout << data_p2p[0] << std::endl;
//     std::cout << data_p2p[1] << std::endl;
//     std::cout << data_p2p[0][0] << std::endl;
//     std::cout << data_p2p[1][0] << std::endl;
//     std::cout << "data_v" << std::endl;
//     for (auto const &elemnent : data_v)
//     {
//         std::cout << elemnent << std::endl;
//     }
//     std::cout << "=====" << std::endl;

//     return 0;
// }