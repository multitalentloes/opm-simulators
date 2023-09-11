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
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <opm/simulators/linalg/cuistl/detail/cusparse_constants.hpp>
#include <opm/simulators/linalg/cuistl/detail/cusparse_matrix_operations.hpp>
#include <opm/simulators/linalg/cuistl/detail/cusparse_safe_call.hpp>
#include <opm/simulators/linalg/cuistl/detail/cusparse_wrapper.hpp>
#include <opm/simulators/linalg/cuistl/detail/cublas_safe_call.hpp>
#include <opm/simulators/linalg/cuistl/detail/cublas_wrapper.hpp>
#include <opm/simulators/linalg/cuistl/detail/CuBlasHandle.hpp>
#include <opm/simulators/linalg/cuistl/detail/fix_zero_diagonal.hpp>
#include <opm/simulators/linalg/cuistl/detail/safe_conversion.hpp>
#include <opm/simulators/linalg/cuistl/detail/vector_operations.hpp>
#include <opm/simulators/linalg/cuistl/PreconditionerAdapter.hpp>
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
    , m_diagInvFlattened(m.N() * m.blockSize()*m.blockSize())
    , m_description(detail::createMatrixDescription())
    , m_cuSparseHandle(detail::CuSparseHandle::getInstance())
    , m_cuBlasHandle(detail::CuBlasHandle::getInstance())
    , d_resultBuffer(m.N() * m.blockSize())
{
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
    // x_{n+1} = x_n + w* (D^-1 * (b - Ax_n) )
    // cusparseDbsrmv computes -Ax_n + b
    // cuda kernel computes D^-1 * (b- Ax_n), call this rhs
    // use an axpy to compute x + w*rhs
      
    const field_type one = 1.0, neg_one = -1.0;

    const auto numberOfRows = detail::to_int(m.N());
    const auto numberOfColumns = detail::to_int(m.N());
    const auto numberOfNonzeroBlocks = detail::to_int(m.nonzeroes());
    const auto blockSize = detail::to_int(m.blockSize());

    auto nonZeroValues = m.getNonZeroValues().data();
    auto rowIndices = m.getRowIndices().data();
    auto columnIndices = m.getColumnIndices().data();

    // bsrmv computes -Ax + b
    // TODO: avoid making this copy, currently forced to because parameter is a const
    d_resultBuffer = b; 

    // allocate space for the inverted diagonal elements of m in a vector
    OPM_CUSPARSE_SAFE_CALL(detail::cusparseBsrmv(m_cuSparseHandle.get(),
                                                 detail::CUSPARSE_MATRIX_ORDER,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 numberOfRows,
                                                 numberOfColumns,
                                                 numberOfNonzeroBlocks,
                                                 &neg_one,
                                                 m_description->get(),
                                                 nonZeroValues,
                                                 rowIndices,
                                                 columnIndices,
                                                 blockSize,
                                                 x.data(),
                                                 &one,
                                                 d_resultBuffer.data()));

    // multiply D^-1 with (b-Ax)
    detail::blockVectorMultiplicationAtAllIndices(m_diagInvFlattened.data(), detail::to_size_t(numberOfRows), detail::to_size_t(blockSize), d_resultBuffer.data());

    // Finish adding w*rhs to x
    OPM_CUBLAS_SAFE_CALL(detail::cublasAxpy(m_cuBlasHandle.get(),
                                            numberOfRows*blockSize,
                                            &m_w,
                                            d_resultBuffer.data(),
                                            1,
                                            x.data(),
                                            1));
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
    //TODO: this update of m is done twice when it is initialized
    m.updateNonzeroValues(detail::makeMatrixWithNonzeroDiagonal(m_underlyingMatrix));
    // Compute the inverted diagonal of A and store it in a vector format in m_diagInvFlattened
    detail::invertDiagonalAndFlatten(m.getNonZeroValues().data(), m.getRowIndices().data(), m.getColumnIndices().data(), detail::to_int(m.N()), detail::to_int(m.blockSize()), m_diagInvFlattened.data());
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
