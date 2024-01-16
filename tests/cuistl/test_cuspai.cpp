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

#define BOOST_TEST_MODULE TestCuSPAI
#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <dune/istl/bcrsmatrix.hh>
#include <opm/simulators/linalg/cuistl/CuSPAI.hpp>
#include <dune/common/dynmatrix.hh>

#include <vector>
#include <random>

using NumericTypes = boost::mpl::list<double>;

BOOST_AUTO_TEST_CASE_TEMPLATE(CUSPAI_TEST_LEAST_SQUARES, T, NumericTypes)
{
    /*
       Test data to least squares solver
           | |2 1| |       | |1 2| |
           | |1 2| |       | |1 2| |
       A = |       | rhs = |       |
           | |0 1| |       | |1 2| | sol = | | 0.1428571428571428 0.2857142857142856 | |
           | |2 3| |       | |1 2| |       | | 0.3714285714285714 0.7428571428571429 | |
   */
    const int N = 2;
    const int M = 1;
    constexpr int blocksize = 2;
    using FMat = Dune::FieldMatrix<T, blocksize, blocksize>;
    using BVec = Dune::BlockVector<FMat>;
    using DMat = Dune::DynamicMatrix<FMat>;

    DMat A(N, M, 0);
    A[0][0][0][0] = 2.0;
    A[0][0][0][1] = 1.0;
    A[0][0][1][0] = 1.0;
    A[0][0][1][1] = 2.0;

    A[1][0][0][0] = 0.0;
    A[1][0][0][1] = 1.0;
    A[1][0][1][0] = 2.0;
    A[1][0][1][1] = 3.0;

    BVec rhs(N, 0);
    rhs[0][0][0] = 1.0;
    rhs[0][1][0] = 1.0;
    rhs[1][0][0] = 1.0;
    rhs[1][1][0] = 1.0;

    rhs[0][0][1] = 2.0;
    rhs[0][1][1] = 2.0;
    rhs[1][0][1] = 2.0;
    rhs[1][1][1] = 2.0;

    FMat exp_sol;
    exp_sol[0][0] = 0.1428571428571428;
    exp_sol[1][0] = 0.37142857142857144;
    exp_sol[0][1] = 0.2857142857142856;
    exp_sol[1][1] = 0.7428571428571429;

    auto sol = Opm::cuistl::solveWithScalarAndReturnBlocked(A, rhs);

    BOOST_CHECK_CLOSE(exp_sol[0][0], sol[0][0][0], 1e-7);
    BOOST_CHECK_CLOSE(exp_sol[0][1], sol[0][0][1], 1e-7);
    BOOST_CHECK_CLOSE(exp_sol[1][0], sol[0][1][0], 1e-7);
    BOOST_CHECK_CLOSE(exp_sol[1][1], sol[0][1][1], 1e-7);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CUSPAI_ON_DIAGONAL_MATRIX, T, NumericTypes)
{
    /*
       Test data to validate jacobi preconditioner, expected result is x_1, and relaxation factor is 0.5
           | |2 1| |0 0| |         | |2/3 -1/3| |0 0| |
           | |1 2| |0 0| |         | |-1/3 2/3| |0 0| |
       A = |             | A_inv = |                  |
           | |0 0| |0 1| |         | |0 0| |-3/2 1/2| |
           | |0 0| |2 3| |         | |0 0| |   1   0| |
   */
    const int N = 2;
    const int nonZeroes = 2;
    constexpr int blocksize = 2;
    using FMat = Dune::FieldMatrix<T, blocksize, blocksize>;
    using BCRSMat = Dune::BCRSMatrix<FMat>;
    using CuSPAI = Opm::cuistl::CuSPAI<BCRSMat, Opm::cuistl::CuVector<T>, Opm::cuistl::CuVector<T>>;

    BCRSMat A(N, N, nonZeroes, BCRSMat::row_wise);
    for (auto row = A.createbegin(); row != A.createend(); ++row) {
        row.insert(row.index());
    }
    A[0][0][0][0] = 2.0;
    A[0][0][0][1] = 1.0;
    A[0][0][1][0] = 1.0;
    A[0][0][1][1] = 2.0;

    A[1][1][0][0] = 0.0;
    A[1][1][0][1] = 1.0;
    A[1][1][1][0] = 2.0;
    A[1][1][1][1] = 3.0;

    CuSPAI cuspai(A, 1);
    std::vector<T> res = cuspai.getSpaiNnzValues();
    std::vector<T> exp_res = {2.0/3.0,-1.0/3.0,-1.0/3.0,2.0/3.0,-3.0/2.0,1.0/2.0,1.0,0.0};

    for (size_t i = 0; i < res.size(); i++){
        BOOST_CHECK_CLOSE(res[i], exp_res[i], 1e-7);
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CUSPAI_DENSE_MATRIX, T, NumericTypes)
{
    // This test case checks that the SPAI1 inverse matches the dune inverse when we have a dense matrix
    const int N = 10;
    const int nonZeroes = N*N;
    constexpr int blocksize = 3;
    using FMat = Dune::FieldMatrix<T, blocksize, blocksize>;
    using BCRSMat = Dune::BCRSMatrix<FMat>;
    using CuSPAI = Opm::cuistl::CuSPAI<BCRSMat, Opm::cuistl::CuVector<T>, Opm::cuistl::CuVector<T>>;

    std::mt19937 random_gen(42);
    std::uniform_real_distribution<double> distribution(1.0, 10);

    BCRSMat A(N, N, nonZeroes, BCRSMat::row_wise);
    Dune::DynamicMatrix<T> A_dyn_scalar(N*blocksize, N*blocksize, 0.0);

    for (auto row = A.createbegin(); row != A.createend(); ++row) {
        for (int col = 0; col < N; ++col){
            row.insert(col);
        }
    }

    // create A as a bcrs matrix that will be sent to the prec
    // create A as a dynamicmatrix that can easily be multiplied with the dynmatrix inverse
    for (auto row = A.begin(); row != A.end(); ++row){
        for (auto col = (*row).begin(); col != (*row).end(); ++col){
            for (auto brow = (*col).begin(); brow != (*col).end(); ++brow){
                for (auto bcol = (*brow).begin(); bcol != (*brow).end(); ++bcol){
                    T rnum = distribution(random_gen);
                    (*bcol) = rnum;
                    A_dyn_scalar[row.index()*blocksize + brow.index()][col.index()*blocksize + bcol.index()]= rnum;
                }
            }
        }
    }

    // The constructor of the cuspai will create the approximate inverse and store the inverse in the resNnz
    CuSPAI cuspai(A, 1);
    std::vector<T> resNnz = cuspai.getSpaiNnzValues();

    int idx = 0;
    Dune::DynamicMatrix<FMat> A_cuspai_inv(N,N,0.0);
    for (auto row = A_cuspai_inv.begin(); row != A_cuspai_inv.end(); ++row){
        for (auto col = (*row).begin(); col != (*row).end(); ++col){
            for (auto brow = (*col).begin(); brow != (*col).end(); ++brow){
                for (auto bcol = (*brow).begin(); bcol != (*brow).end(); ++bcol){
                    (*bcol) = resNnz[idx++];
                }
            }
        }
    }

    auto A_dune_inv_scalar = A_dyn_scalar;
    A_dune_inv_scalar.invert();

    Dune::DynamicMatrix<FMat> A_dune_inv_block(N,N,0.0);

    for (int i = 0; i < N*blocksize; ++i){
        for (int j = 0; j < N*blocksize; ++j){
            A_dune_inv_block[i/blocksize][j/blocksize][i%blocksize][j%blocksize] = A_dune_inv_scalar[i][j];
        }
    }

    for (auto row = 0; row < N; ++row){
        for (auto col = 0; col < N; ++col){
            for (auto brow = 0; brow < blocksize; ++brow){
                for (auto bcol = 0; bcol < blocksize; ++bcol){
                    auto cuspai_inv_scalar = A_cuspai_inv[row][col][brow][bcol];
                    auto dune_inv_scalar = A_dune_inv_block[row][col][brow][bcol];
                    BOOST_CHECK_CLOSE(cuspai_inv_scalar, dune_inv_scalar, 1e7);
                }
            }
        }
    }
}

template<class FMat>
double frob_of_identity_diff(Dune::DynamicMatrix<FMat> *mat){
    double ans = 0;
    for (auto row = mat->begin(); row != mat->end(); ++row){
        for (auto col = (*row).begin(); col != (*row).end(); ++col){
            for (auto brow = (*col).begin(); brow != (*col).end(); ++brow){
                for (auto bcol = (*brow).begin(); bcol != (*brow).end(); ++bcol){
                    double val = *bcol;
                    if (row.index() == col.index() && brow.index() == bcol.index()){
                        val -= 1;
                    }
                    ans += val*val;
                }
            }
        }
    }
    return ans;
}


BOOST_AUTO_TEST_CASE_TEMPLATE(CUSPAI_LEVELS, T, NumericTypes)
{
    // with this test case we want to verify that increasing the sparsity pattern for the spai
    // will create a better preconditioner until the spai is dense
    const int N = 10;
    const int nonZeroes = 3*N-2;
    constexpr int blocksize = 3;
    using FMat = Dune::FieldMatrix<T, blocksize, blocksize>;
    using BCRSMat = Dune::BCRSMatrix<FMat>;
    using CuSPAI = Opm::cuistl::CuSPAI<BCRSMat, Opm::cuistl::CuVector<T>, Opm::cuistl::CuVector<T>>;

    BCRSMat A(N, N, nonZeroes, BCRSMat::row_wise);
    Dune::DynamicMatrix<FMat> A_dyn_block(N, N, 0.0);

    for (auto row = A.createbegin(); row != A.createend(); ++row) {
        auto row_idx = row.index();
        row.insert(row_idx);
        if (row_idx > 0){
            row.insert(row_idx-1);
        }
        if (row_idx < N - 1){
            row.insert(row_idx+1);
        }
    }

   // create A as a bcrs matrix that will be sent to the prec
   // create A as a dynamicmatrix that can easily be multiplied with the dynmatrix inverse
    for (auto row = A.begin(); row != A.end(); ++row){
        for (auto col = (*row).begin(); col != (*row).end(); ++col){
            T scalar_val = (row.index() == col.index() ? 2.0 : 1.0);
            FMat block_val(0);
            for (auto i = 0; i < blocksize; ++i){
                block_val[i][i] = scalar_val;
            }
            (*col) = block_val;
            A_dyn_block[row.index()][col.index()] = block_val;
        }
    }

    std::vector<double> frobs = {frob_of_identity_diff<FMat>(&A_dyn_block)};
    for (int i = 1; i < 15; i++){
        CuSPAI cuspai(A, i);
        std::vector<T> spaiNnz = cuspai.getSpaiNnzValues();

        int idx = 0;
        Dune::DynamicMatrix<FMat> A_cuspai_inv(N,N,0.0);
        for (auto row = A_cuspai_inv.begin(); row != A_cuspai_inv.end(); ++row){
            for (auto col = (*row).begin(); col != (*row).end(); ++col){
                for (auto brow = (*col).begin(); brow != (*col).end(); ++brow){
                    for (auto bcol = (*brow).begin(); bcol != (*brow).end(); ++bcol){
                        (*bcol) = spaiNnz[idx++];
                    }
                }
            }
        }

        auto cuspai_id = A_dyn_block;
        cuspai_id.leftmultiply(A_cuspai_inv);

        frobs.push_back(frob_of_identity_diff<FMat>(&cuspai_id));
    }

    BOOST_CHECK(true);
}

