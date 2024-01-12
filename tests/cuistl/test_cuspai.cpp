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

using NumericTypes = boost::mpl::list<double>;

BOOST_AUTO_TEST_CASE_TEMPLATE(CUSPAI_TEST_LEAST_SQUARES, T, NumericTypes)
{
    /*
       Test data to validate jacobi preconditioner, expected result is x_1, and relaxation factor is 0.5
           | |2 1| |       | |1 2| |
           | |1 2| |       | |1 2| |
       A = |       | rhs = |       |
           | |0 1| |       | |1 2| |
           | |2 3| |       | |1 2| |
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
