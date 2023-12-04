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

#define BOOST_TEST_MODULE TestCuSparseMatrixOperations
#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <dune/istl/bcrsmatrix.hh>
#include <opm/simulators/linalg/cuistl/CuSparseMatrix.hpp>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <opm/simulators/linalg/cuistl/PreconditionerAdapter.hpp>
#include <opm/simulators/linalg/cuistl/detail/cusparse_matrix_operations.hpp>
#include <opm/simulators/linalg/cuistl/detail/fix_zero_diagonal.hpp>

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

    B[0][0][0][0] = 1.0;
    B[0][0][0][1] = 2.0;
    B[0][0][0][2] = 3.0;
    B[0][0][1][0] = 5.0;
    B[0][0][1][1] = 2.0;
    B[0][0][1][2] = 3.0;
    B[0][0][2][0] = 2.0;
    B[0][0][2][1] = 1.0;
    B[0][0][2][2] = 1.0;

    B[0][1][0][0] = 1.0;
    B[0][1][1][1] = 1.0;
    B[0][1][2][2] = 1.0;

    B[1][1][0][0] = -1.0;
    B[1][1][1][1] = -1.0;
    B[1][1][2][2] = -1.0;

    Opm::cuistl::CuSparseMatrix<T> m = Opm::cuistl::CuSparseMatrix<T>::fromMatrix(B);
    Opm::cuistl::CuVector<T> dInvDiag(blocksize * blocksize * N);

    Opm::cuistl::detail::invertDiagonalAndFlatten(m.getNonZeroValues().data(),
                                                  m.getRowIndices().data(),
                                                  m.getColumnIndices().data(),
                                                  N,
                                                  blocksize,
                                                  dInvDiag.data());

    std::vector<T> expectedInvDiag {-1.0 / 4.0,
                                    1.0 / 4.0,
                                    0.0,
                                    1.0 / 4.0,
                                    -5.0 / 4.0,
                                    3.0,
                                    1.0 / 4.0,
                                    3.0 / 4.0,
                                    -2.0,
                                    -1.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    -1.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    -1.0};
    std::vector<T> computedInvDiag = dInvDiag.asStdVector();

    BOOST_REQUIRE_EQUAL(expectedInvDiag.size(), computedInvDiag.size());
    for (size_t i = 0; i < expectedInvDiag.size(); ++i) {
        BOOST_CHECK_CLOSE(expectedInvDiag[i], computedInvDiag[i], 1e-7);
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

    B[0][0][0][0] = 1.0;
    B[0][0][0][1] = 2.0;
    B[0][0][1][0] = 1.0 / 2.0;
    B[0][0][1][1] = 2.0;

    B[0][1][0][0] = 1.0;
    B[0][1][1][1] = 1.0;

    B[1][1][0][0] = -1.0;
    B[1][1][1][1] = -1.0;

    Opm::cuistl::CuSparseMatrix<T> m = Opm::cuistl::CuSparseMatrix<T>::fromMatrix(B);
    Opm::cuistl::CuVector<T> dInvDiag(blocksize * blocksize * N);

    Opm::cuistl::detail::invertDiagonalAndFlatten(m.getNonZeroValues().data(),
                                                  m.getRowIndices().data(),
                                                  m.getColumnIndices().data(),
                                                  N,
                                                  blocksize,
                                                  dInvDiag.data());

    std::vector<T> expectedInvDiag {2.0, -2.0, -1.0 / 2.0, 1.0, -1.0, 0.0, 0.0, -1.0};
    std::vector<T> computedInvDiag = dInvDiag.asStdVector();

    BOOST_REQUIRE_EQUAL(expectedInvDiag.size(), computedInvDiag.size());
    for (size_t i = 0; i < expectedInvDiag.size(); ++i) {
        BOOST_CHECK_CLOSE(expectedInvDiag[i], computedInvDiag[i], 1e-7);
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(testMvHelperFunction, T, NumericTypes)
{
    size_t N = 3;
    std::vector<T> A(N*N), b(N), c(N);
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            A[i*N + j] = i*N + j;
        }
        b[i] = i;
        c[i] = N - i;
    }

    bool setEqual = true;

    for (size_t i = 0; i < N; ++i) {
        if (setEqual){
            c[i] = 0;
        }

        for (size_t j = 0; j < N; ++j) {
            if (setEqual){
                c[i] += A[i * N + j] * b[j];
            }
            else if (!setEqual){
                c[i] -= A[i * N + j] * b[j];
            }
        }
    }

    std::vector<T> expected_ans = {5.0, 14.0, 23.0};
    for (int i = 0; i < N; i++){

        BOOST_CHECK_CLOSE(c[i], expected_ans[i], 1e-7);
    }

    setEqual = false;
    expected_ans = {0.0, 0.0, 0.0};

    for (size_t i = 0; i < N; ++i) {
        if (setEqual){
            c[i] = 0;
        }

        for (size_t j = 0; j < N; ++j) {
            if (setEqual){
                c[i] += A[i * N + j] * b[j];
            }
            else if (!setEqual){
                c[i] -= A[i * N + j] * b[j];
            }
        }
    }

    for (int i = 0; i < N; i++){

        BOOST_CHECK_CLOSE(c[i], expected_ans[i], 1e-7);
    }
}

// BOOST_AUTO_TEST_CASE_TEMPLATE(moveToReorderedMatrix, T, NumericTypes)
// {
//     const size_t blocksize = 2;
//     const size_t N = 2;
//     const int nonZeroes = 3;
//     using M = Dune::FieldMatrix<T, blocksize, blocksize>;
//     using SpMatrix = Dune::BCRSMatrix<M>;

//     SpMatrix A(N, N, nonZeroes, SpMatrix::row_wise);
//     for (auto row = A.createbegin(); row != A.createend(); ++row) {
//         row.insert(row.index());
//         if (row.index() == 0) {
//             row.insert(row.index() + 1);
//         }
//     }
//     SpMatrix B(N, N, nonZeroes, SpMatrix::row_wise);
//     for (auto row = B.createbegin(); row != B.createend(); ++row) {
//         row.insert(row.index());
//         if (row.index() == 1) {
//             row.insert(row.index() - 1);
//         }
//     }
//     A[0][0][0][0] = 1.0;
//     A[0][1][0][1] = 2.0;
//     A[1][1][1][0] = 3.0;
//     B[0][0][0][0] = -1.0;
//     B[1][0][0][0] = -1.0;
//     B[1][1][0][0] = -1.0;

//     Opm::cuistl::CuSparseMatrix<T> mA = Opm::cuistl::CuSparseMatrix<T>::fromMatrix(A);
//     Opm::cuistl::CuSparseMatrix<T> mB = Opm::cuistl::CuSparseMatrix<T>::fromMatrix(B);

//     Opm::cuistl::CuVector<int> naturalToReorder({1, 0});

//     Opm::cuistl::detail::moveMatDataToReordered2<T, 2>(mA.getNonZeroValues().data(), mA.getRowIndices().data(), mA.getColumnIndices().data(), mB.getNonZeroValues().data(), mB.getRowIndices().data(), mB.getColumnIndices().data(), naturalToReorder.data(), N, nonZeroes);

//     std::vector<T> orig(mA.getNonZeroValues().asStdVector());
//     std::vector<T> res(mB.getNonZeroValues().asStdVector());

//     printf("=======\n");
//     for (auto e : orig){
//         printf("%lf ", e);
//     }
//     printf("\n");
//     for (auto e : res){
//         printf("%lf ", e);
//     }
//     printf("=======\n");
// }