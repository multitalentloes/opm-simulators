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

#define BOOST_TEST_MODULE TestCudaSafeCall
#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/preconditioners.hh>
#include <opm/simulators/linalg/cuistl/CuJac.hpp>
#include <opm/simulators/linalg/cuistl/CuSparseMatrix.hpp>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <opm/simulators/linalg/cuistl/detail/cusparse_matrix_operations.hpp>
#include <opm/simulators/linalg/cuistl/detail/fix_zero_diagonal.hpp>

using NumericTypes = boost::mpl::list<double, float>;

BOOST_AUTO_TEST_CASE_TEMPLATE(FlattenAndInvertDiagonal, T, NumericTypes)
{
    const int N = 2;
    const int nonZeroes = 3;
    using M = Dune::FieldMatrix<T, 3, 3>;
    using SpMatrix = Dune::BCRSMatrix<M>;
    using Vector = Dune::BlockVector<Dune::FieldVector<T, 3>>;
    using cujac = Opm::cuistl::CuJac<SpMatrix, Opm::cuistl::CuVector<T>, Opm::cuistl::CuVector<T>>;
    /*
        create a matrix with this shape
        | |1 2 0| | 1  0  0| |
        | |0 1 0| | 0  1  0| |
        | |0 0 1| | 0  0  1| |
        |                    | 
        | |0 0 0| |-1  0  0| |
        | |0 0 0| | 0 -1  0| |
        | |0 0 0| | 0  0 -1| |

        we want to end up with this vector
        | | 1 -2  0| |
        | | 0  1  0| |
        | | 0  0  1| |
        |            |
        | |-1  0  0| |
        | | 0 -1  0| |
        | | 0  0 -1| |
        
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
    B[0][0][1][1]=1.0;
    B[0][0][2][2]=1.0;

    B[0][1][0][0]=1.0;
    B[0][1][1][1]=1.0;
    B[0][1][2][2]=1.0;

    B[1][1][0][0]=1.0;
    B[1][1][1][1]=1.0;
    B[1][1][2][2]=1.0;

    Opm::cuistl::CuSparseMatrix<T> m = Opm::cuistl::CuSparseMatrix<T>::fromMatrix(Opm::cuistl::detail::makeMatrixWithNonzeroDiagonal(B));
    auto vec = m.getNonZeroValues().data();
    BOOST_CHECK(true);
    // TODO call the flattening here and check the results
    // detail::flatten(nonZeroValues, rowIndices, columnIndices, numberOfRows, detail::to_size_t(blockSize), d_mDiagInv.data());
}

BOOST_AUTO_TEST_CASE(CUJACApplyIsEqualToDuneSeqJacApply)
{
    // const int N = 2;
    // const int nonZeroes = 3;
    // using M = Dune::FieldMatrix<T, 3, 3>;
    // using SpMatrix = Dune::BCRSMatrix<M>;
    // using Vector = Dune::BlockVector<Dune::FieldVector<T, 3>>;
    // using cujac = Opm::cuistl::CuJac<SpMatrix, Opm::cuistl::CuVector<T>, Opm::cuistl::CuVector<T>>;
    // /*
    //     create a matrix with this shape
    //     | |1 2 0| | 1  0  0| |
    //     | |0 1 0| | 0  1  0| |
    //     | |0 0 1| | 0  0  1| |
    //     |                    | 
    //     | |0 0 0| |-1  0  0| |
    //     | |0 0 0| | 0 -1  0| |
    //     | |0 0 0| | 0  0 -1| |

    //     we want to end up with this vector
    //     | | 1 -2  0| |
    //     | | 0  1  0| |
    //     | | 0  0  1| |
    //     |            |
    //     | |-1  0  0| |
    //     | | 0 -1  0| |
    //     | | 0  0 -1| |
        
    // */

    // SpMatrix B(N, N, nonZeroes, SpMatrix::row_wise);
    // for (auto row = B.createbegin(); row != B.createend(); ++row) {
    //     row.insert(row.index());
    //     if (row.index() == 0) {
    //         row.insert(row.index() + 1);
    //     }
    // }

    // B[0][0][0][0]=1.0;
    // B[0][0][0][1]=2.0;
    // B[0][0][1][1]=1.0;
    // B[0][0][2][2]=1.0;

    // B[0][1][0][0]=1.0;
    // B[0][1][1][1]=1.0;
    // B[0][1][2][2]=1.0;

    // B[1][1][0][0]=1.0;
    // B[1][1][1][1]=1.0;
    // B[1][1][2][2]=1.0;

    // auto duneILU = Dune::SeqJac<SpMatrix, Vector, Vector>(B, 1.0);
    // auto cuILU = Opm::cuistl::PreconditionerAdapter<Vector, Vector, cujac>(std::make_shared<cujac>(B, 1.0));

    // // check for the standard basis {e_i}
    // // (e_i=(0,...,0, 1 (i-th place), 0, ..., 0))
    // for (int i = 0; i < N; ++i) {
    //     Vector inputVector(N);
    //     inputVector[i][0] = 1.0;
    //     Vector outputVectorDune(N);
    //     Vector outputVectorCuistl(N);
    //     duneILU.apply(outputVectorDune, inputVector);
    //     cuILU.apply(outputVectorCuistl, inputVector);

    //     for (int component = 0; component < N; ++component) {
    //         BOOST_CHECK_CLOSE(outputVectorDune[component][0],
    //                           outputVectorCuistl[component][0],
    //                           std::numeric_limits<T>::epsilon() * 1000);
    //     }
    // }
}