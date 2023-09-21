/*
  Copyright 2022-2023 SINTEF AS

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <config.h>

#define BOOST_TEST_MODULE TestCuJac
#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <dune/istl/bcrsmatrix.hh>
#include <opm/simulators/linalg/cuistl/CuJac.hpp>
#include <opm/simulators/linalg/cuistl/CuSparseMatrix.hpp>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <opm/simulators/linalg/cuistl/detail/cusparse_matrix_operations.hpp>
#include <opm/simulators/linalg/cuistl/detail/vector_operations.hpp>
#include <opm/simulators/linalg/cuistl/detail/fix_zero_diagonal.hpp>
#include <opm/simulators/linalg/cuistl/PreconditionerAdapter.hpp>
#include <string>

using NumericTypes = boost::mpl::list<double, float>;

BOOST_AUTO_TEST_CASE_TEMPLATE(FlattenAndInvertDiagonalWith3By3Blocks, T, NumericTypes)
{
    const size_t blocksize = 3;
    const size_t N = 2;
    const int nonZeroes = 3;
    using M = Dune::FieldMatrix<T, blocksize, blocksize>;
    using SpMatrix = Dune::BCRSMatrix<M>;
    /*
        create this sparse matrix
        | |1 2 3| | 1  0  0| |
        | |5 2 3| | 0  1  0| |
        | |2 1 1| | 0  0  1| |
        |                    |
        | |0 0 0| |-1  0  0| |
        | |0 0 0| | 0 -1  0| |
        | |0 0 0| | 0  0 -1| |

        The diagonal elements inverted, and put in a vector should look like this
        | |-1/4  1/4  0| |
        | | 1/4 -4/5  3| |
        | | 1/4  3/4 -2| |
        |                |
        | |  -1    0  0| |
        | |   0   -1  0| |
        | |   0    0 -1| |
        
    */

    SpMatrix B(N, N, nonZeroes, SpMatrix::row_wise);
    for (auto row = B.createbegin(); row != B.createend(); ++row) {
        row.insert(row.index());
        if (row.index() == 0) {
            row.insert(row.index() + 1);
        }
    }

    B[0][0][0][0]=1.0;
    B[0][0][0][1]=2.0;
    B[0][0][0][2]=3.0;
    B[0][0][1][0]=5.0;
    B[0][0][1][1]=2.0;
    B[0][0][1][2]=3.0;
    B[0][0][2][0]=2.0;
    B[0][0][2][1]=1.0;
    B[0][0][2][2]=1.0;

    B[0][1][0][0]=1.0;
    B[0][1][1][1]=1.0;
    B[0][1][2][2]=1.0;

    B[1][1][0][0]=-1.0;
    B[1][1][1][1]=-1.0;
    B[1][1][2][2]=-1.0;

    Opm::cuistl::CuSparseMatrix<T> m = Opm::cuistl::CuSparseMatrix<T>::fromMatrix(Opm::cuistl::detail::makeMatrixWithNonzeroDiagonal(B));
    Opm::cuistl::CuVector<T> d_invDiag(blocksize*blocksize*N);

    Opm::cuistl::detail::invertDiagonalAndFlatten(m.getNonZeroValues().data(), m.getRowIndices().data(), m.getColumnIndices().data(), N, blocksize, d_invDiag.data());

    std::vector<T> expected_inv_diag{-1.0/4.0,1.0/4.0,0.0,1.0/4.0,-5.0/4.0,3.0,1.0/4.0,3.0/4.0,-2.0,-1.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,-1.0};
    std::vector<T> computed_inv_diag = d_invDiag.asStdVector();

    BOOST_REQUIRE_EQUAL(expected_inv_diag.size(), computed_inv_diag.size());
    for (size_t i = 0; i < expected_inv_diag.size(); i++){
        BOOST_CHECK_CLOSE(expected_inv_diag[i], computed_inv_diag[i], 1e-7);
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(FlattenAndInvertDiagonalWith2By2Blocks, T, NumericTypes)
{
    const size_t blocksize = 2;
    const size_t N = 2;
    const int nonZeroes = 3;
    using M = Dune::FieldMatrix<T, blocksize, blocksize>;
    using SpMatrix = Dune::BCRSMatrix<M>;
    /*
        create this sparse matrix
        | |  1 2| | 1  0| |
        | |1/2 2| | 0  1| |
        |                 | 
        | |  0 0| |-1  0| |
        | |  0 0| | 0 -1| |

        The diagonal elements inverted, and put in a vector should look like this
        | |   2 - 2| |
        | |-1/2   1| |
        |            |
        | |  -1   0| |
        | |   0  -1| |
        
    */

    SpMatrix B(N, N, nonZeroes, SpMatrix::row_wise);
    for (auto row = B.createbegin(); row != B.createend(); ++row) {
        row.insert(row.index());
        if (row.index() == 0) {
            row.insert(row.index() + 1);
        }
    }

    B[0][0][0][0]=1.0;
    B[0][0][0][1]=2.0;
    B[0][0][1][0]=1.0/2.0;
    B[0][0][1][1]=2.0;

    B[0][1][0][0]=1.0;
    B[0][1][1][1]=1.0;

    B[1][1][0][0]=-1.0;
    B[1][1][1][1]=-1.0;

    Opm::cuistl::CuSparseMatrix<T> m = Opm::cuistl::CuSparseMatrix<T>::fromMatrix(Opm::cuistl::detail::makeMatrixWithNonzeroDiagonal(B));
    Opm::cuistl::CuVector<T> d_invDiag(blocksize*blocksize*N);

    Opm::cuistl::detail::invertDiagonalAndFlatten(m.getNonZeroValues().data(), m.getRowIndices().data(), m.getColumnIndices().data(), N, blocksize, d_invDiag.data());

    std::vector<T> expected_inv_diag{2.0,-2.0,-1.0/2.0,1.0,-1.0,0.0,0.0,-1.0};
    std::vector<T> computed_inv_diag = d_invDiag.asStdVector();

    BOOST_REQUIRE_EQUAL(expected_inv_diag.size(), computed_inv_diag.size());
    for (size_t i = 0; i < expected_inv_diag.size(); i++){
        BOOST_CHECK_CLOSE(expected_inv_diag[i], computed_inv_diag[i], 1e-7);
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ElementWiseMultiplicationOf3By3BlockVectorAndVectorVector, T, NumericTypes)
{
    /*
        Example in the test for multiplying by element a blockvector with a vector of vectors
        | |1 2 3| |   | |3| |   | |10| |
        | |5 2 3| | X | |2| | = | |22| |
        | |2 1 1| |   | |1| |   | |10| |
    */

    const size_t blocksize = 3;
    const size_t N = 1;
    
    std::vector<T> h_blockVector({1.0,2.0,3.0,5.0,2.0,3.0,2.0,1.0,2.0});
    std::vector<T> h_vecVector({3.0,2.0,1.0});
    Opm::cuistl::CuVector<T> d_blockVector(h_blockVector);
    Opm::cuistl::CuVector<T> d_vecVector(h_vecVector); 

    Opm::cuistl::detail::blockVectorMultiplicationAtAllIndices(d_blockVector.data(), N, blocksize, d_vecVector.data());

    std::vector<T> expected_vec{10.0,22.0,10.0};
    std::vector<T> computed_vec = d_vecVector.asStdVector();

    BOOST_REQUIRE_EQUAL(expected_vec.size(), computed_vec.size());
    for (size_t i = 0; i < expected_vec.size(); i++){
        BOOST_CHECK_CLOSE(expected_vec[i], computed_vec[i], 1e-7);
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ElementWiseMultiplicationOf2By2BlockVectorAndVectorVector, T, NumericTypes)
{
    /*
        Example in the test for multiplying by element a blockvector with a vector of vectors
        | |1 2| |   | |1| |   | | 7| |
        | |3 4| | X | |3| | = | |15| |
        |       |   |     |   |      |
        | |4 3| |   | |2| |   | |20| |
        | |2 1| |   | |4| |   | | 8| |
    */

    const size_t blocksize = 2;
    const size_t N = 2;
    
    std::vector<T> h_blockVector({1.0,2.0,3.0,4.0,4.0,3.0,2.0,1.0});
    std::vector<T> h_vecVector({1.0,3.0,2.0,4.0});
    Opm::cuistl::CuVector<T> d_blockVector(h_blockVector);
    Opm::cuistl::CuVector<T> d_vecVector(h_vecVector); 

    Opm::cuistl::detail::blockVectorMultiplicationAtAllIndices(d_blockVector.data(), N, blocksize, d_vecVector.data());

    std::vector<T> expected_vec{7.0,15.0,20.0,8.0};
    std::vector<T> computed_vec = d_vecVector.asStdVector();

    BOOST_REQUIRE_EQUAL(expected_vec.size(), computed_vec.size());
    for (size_t i = 0; i < expected_vec.size(); i++){
        BOOST_CHECK_CLOSE(expected_vec[i], computed_vec[i], 1e-7);
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CUJACApplyIsEqualToDuneSeqJacApply, T, NumericTypes)
{

     /*
        Test data to validate jacobi preconditioner, expected result is x_1
            | |3 1|  | 1  0|       | |1| |     | |2| |       | |   1| |
            | |2 1|  | 0  1|       | |2| |     | |1| |       | |   0| |
        A = |              | x_0 = |     | b = |     | x_1 = |        |
            | |0 0|  |-1  0|       | |1| |     | |3| |       | |  -1| |
            | |0 0|  | 0 -1|       | |1| |     | |4| |       | |-1.5| |
    */
    const int N = 2;
    const int blocksize = 2;
    const int nonZeroes = 3;
    using M = Dune::FieldMatrix<T, blocksize, blocksize>;
    using SpMatrix = Dune::BCRSMatrix<M>;
    using Vector = Dune::BlockVector<Dune::FieldVector<T, 2>>;
    using cujac = Opm::cuistl::CuJac<SpMatrix, Opm::cuistl::CuVector<T>, Opm::cuistl::CuVector<T>>;
    
    SpMatrix B(N, N, nonZeroes, SpMatrix::row_wise);
    for (auto row = B.createbegin(); row != B.createend(); ++row) {
        row.insert(row.index());
        if (row.index() == 0) {
            row.insert(row.index() + 1);
        }
    }

    B[0][0][0][0]=3.0;
    B[0][0][0][1]=1.0;
    B[0][0][1][0]=2.0;
    B[0][0][1][1]=1.0;

    B[0][1][0][0]=1.0;
    B[0][1][1][1]=1.0;

    B[1][1][0][0]=-1.0;
    B[1][1][1][1]=-1.0;

    auto CuJac = Opm::cuistl::PreconditionerAdapter<Vector, Vector, cujac>(std::make_shared<cujac>(B, 0.5));
    
    Vector h_dune_x(2);
    h_dune_x[0][0] = 1.0;
    h_dune_x[0][1] = 2.0;
    h_dune_x[1][0] = 1.0;
    h_dune_x[1][1] = 1.0;

    Vector h_dune_b(2);
    h_dune_b[0][0] = 2.0;
    h_dune_b[0][1] = 1.0;
    h_dune_b[1][0] = 3.0;
    h_dune_b[1][1] = 4.0;
    
    std::cout<<"HOST VECTORS:\n" << h_dune_x << std::endl << h_dune_b << std::endl;

    CuJac.apply(h_dune_x, h_dune_b);
    BOOST_CHECK_CLOSE(h_dune_x[0][0], 1.0, 1e-7);
    BOOST_CHECK_CLOSE(h_dune_x[0][1], 0.0, 1e-7);
    BOOST_CHECK_CLOSE(h_dune_x[1][0], -1.0, 1e-7);
    BOOST_CHECK_CLOSE(h_dune_x[1][1], -3.0/2.0, 1e-7);
    BOOST_CHECK(true);
}