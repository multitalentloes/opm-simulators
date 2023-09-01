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
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <fmt/core.h>
#include <opm/common/ErrorMacros.hpp>
#include <opm/simulators/linalg/cuistl/CuJac.hpp>
#include <opm/simulators/linalg/cuistl/detail/cusparse_constants.hpp>
#include <opm/simulators/linalg/cuistl/detail/cusparse_safe_call.hpp>
#include <opm/simulators/linalg/cuistl/detail/cusparse_wrapper.hpp>
#include <opm/simulators/linalg/cuistl/detail/fix_zero_diagonal.hpp>
#include <opm/simulators/linalg/cuistl/detail/safe_conversion.hpp>
#include <opm/simulators/linalg/matrixblock.hh>

// This file is based on the guide at https://docs.nvidia.com/cuda/cusparse/index.html#csrilu02_solve ,
// it highly recommended to read that before proceeding.


namespace Opm::cuistl
{

template <class M, class X, class Y, int l>
CuJac<M, X, Y, l>::CuJac(const M& A, field_type w)
    : m_underlyingMatrix(A)
    , m_w(w)
    , m(CuSparseMatrix<field_type>::fromMatrix(detail::makeMatrixWithNonzeroDiagonal(A)))
    , m_temporaryStorage(m.N() * m.blockSize())
    , m_description(detail::createMatrixDescription())
    , m_cuSparseHandle(detail::CuSparseHandle::getInstance())
{
    std::cout << "---- DEBUG ---- CUJAC FILE USED\n";
    // Some sanity check
    OPM_ERROR_IF(A.N() != m.N(),
                 fmt::format("CuSparse matrix not same size as DUNE matrix. {} vs {}.", m.N(), A.N()));
    OPM_ERROR_IF(
        A[0][0].N() != m.blockSize(),
        fmt::format("CuSparse matrix not same blocksize as DUNE matrix. {} vs {}.", m.blockSize(), A[0][0].N()));
    OPM_ERROR_IF(
        A.N() * A[0][0].N() != m.dim(),
        fmt::format("CuSparse matrix not same dimension as DUNE matrix. {} vs {}.", m.dim(), A.N() * A[0][0].N()));
    OPM_ERROR_IF(A.nonzeroes() != m.nonzeroes(),
                 fmt::format("CuSparse matrix not same number of non zeroes as DUNE matrix. {} vs {}. ",
                             m.nonzeroes(),
                             A.nonzeroes()));

    update();
}

template <class M, class X, class Y, int l>
void
CuJac<M, X, Y, l>::pre([[maybe_unused]] X& x, [[maybe_unused]] Y& b)
{
}

template <class M, class X, class Y, int l>
void
CuJac<M, X, Y, l>::apply(X& x, const Y& b)
{

    //TODO: Mimic Dunes performance of x_{n+1}=wD^-1(b-Ax_n)

    // OPM_CUSPARSE_SAFE_CALL(Cusparse);

    // x_{n+1} = x_n + w* (D^-1 * (b - Ax_n) )
    // cusparseDbsrmv computes -Ax_n + b

    // Maybe we can efficiently read the diagonal from the matrix
    // and use vector operations to invert it 

    // x_n + w*rhs only consists of dense vectors so use cublasDaxpy
      
    const float one = 1.0, neg_one = -1.0;

    const auto numberOfRows = detail::to_int(m.N());
    const auto numberOfNonzeroBlocks = detail::to_int(m.nonzeroes());
    const auto blockSize = detail::to_int(m.blockSize());

    auto nonZeroValues = m.getNonZeroValues().data();
    auto rowIndices = m.getRowIndices().data();
    auto columnIndices = m.getColumnIndices().data();

    // cusparseBsrmv(cusparseHandle_t handle,
    //           cusparseDirection_t dirA,
    //           cusparseOperation_t transA,
    //           int mb,
    //           int nb,
    //           int nnzb,
    //           const double* alpha,
    //           const cusparseMatDescr_t descrA,
    //           const double* bsrSortedValA,
    //           const int* bsrSortedRowPtrA,
    //           const int* bsrSortedColIndA,
    //           int blockDim,
    //           const double* x,
    //           const double* beta,
    //           double* y)

    // bsrmv:
    // alpha * op(A) * x + beta * y // alpha and beta are scalars, A is matrix, and x and y are vectors
    // we want to compute b - Ax, so set alpha to minus one, A is A, beta is one, x is x and y is b;
    OPM_CUSPARSE_SAFE_CALL(detail::cusparseBsrmv(m_cuSparseHandle.get(),
                                                 CUSPARSE_DIRECTION_ROW,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 numberOfRows,
                                                 numberOfNonzeroBlocks,
                                                 &neg_one,
                                                 m_description->get(),
                                                 nonZeroValues,
                                                 rowIndices,
                                                 columnIndices,
                                                 blockSize,
                                                 &one, // x
                                                 &one,
                                                 &one));// b


    /*
  122 |     OPM_CUSPARSE_SAFE_CALL(detail::cusparseBsrmv(m_cuSparseHandle.get(), cusparseBsrmv(cusparseHandle_t
      |                            ~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~
  123 |                                                  detail::CUSPARSE_MATRIX_ORDER,  const cusparseDirection_t
      |                                                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  124 |                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,  cusparseOperation_t
      |                                                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  125 |                                                  numberOfRows,  const int&
      |                                                  ~~~~~~~~~~~~~
  126 |                                                  numberOfNonzeroBlocks, const int&
      |                                                  ~~~~~~~~~~~~~~~~~~~~~~
  127 |                                                  &neg_one, const field_type*
      |                                                  ~~~~~~~~~
  128 |                                                  m_description->get(), cusparseMatDescr*
      |                                                  ~~~~~~~~~~~~~~~~~~~~~
  129 |                                                  nonZeroValues, float*&
      |                                                  ~~~~~~~~~~~~~~
  130 |                                                  rowIndices, int*&
      |                                                  ~~~~~~~~~~~
  131 |                                                  columnIndices, int*&
      |                                                  ~~~~~~~~~~~~~~
  132 |                                                  blockSize, const int&
      |                                                  ~~~~~~~~~~
  133 |                                                  &one, const field_type*
      |                                                  ~~~~~
  134 |                                                  &one, const field_type*
      |                                                  ~~~~~
  135 |                                                  &one)); const field_type*

    */
}

template <class M, class X, class Y, int l>
void
CuJac<M, X, Y, l>::post([[maybe_unused]] X& x)
{
}

template <class M, class X, class Y, int l>
Dune::SolverCategory::Category
CuJac<M, X, Y, l>::category() const
{
    return Dune::SolverCategory::sequential;
}

template <class M, class X, class Y, int l>
void
CuJac<M, X, Y, l>::update()
{
    m.updateNonzeroValues(m_underlyingMatrix);
}

} // namespace Opm::cuistl
#define INSTANTIATE_CUJAC_DUNE(realtype, blockdim)                                                                 \
    template class ::Opm::cuistl::CuJac<Dune::BCRSMatrix<Dune::FieldMatrix<realtype, blockdim, blockdim>>,         \
                                            ::Opm::cuistl::CuVector<realtype>,                                         \
                                            ::Opm::cuistl::CuVector<realtype>>;                                        \
    template class ::Opm::cuistl::CuJac<Dune::BCRSMatrix<Opm::MatrixBlock<realtype, blockdim, blockdim>>,          \
                                            ::Opm::cuistl::CuVector<realtype>,                                         \
                                            ::Opm::cuistl::CuVector<realtype>>


INSTANTIATE_CUJAC_DUNE(double, 1);
INSTANTIATE_CUJAC_DUNE(double, 2);
INSTANTIATE_CUJAC_DUNE(double, 3);
INSTANTIATE_CUJAC_DUNE(double, 4);
INSTANTIATE_CUJAC_DUNE(double, 5);
INSTANTIATE_CUJAC_DUNE(double, 6);

INSTANTIATE_CUJAC_DUNE(float, 1);
INSTANTIATE_CUJAC_DUNE(float, 2);
INSTANTIATE_CUJAC_DUNE(float, 3);
INSTANTIATE_CUJAC_DUNE(float, 4);
INSTANTIATE_CUJAC_DUNE(float, 5);
INSTANTIATE_CUJAC_DUNE(float, 6);
