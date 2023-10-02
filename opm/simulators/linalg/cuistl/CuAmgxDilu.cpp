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
#include <opm/simulators/linalg/cuistl/CuAmgxDilu.hpp>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <opm/simulators/linalg/cuistl/detail/cusparse_constants.hpp>
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

#include <iostream>
#include <amgx_c.h>

class AmgxHelper
{
    public:
        static void ensureAmgxInitialized()
        {
            static AmgxHelper ah;
        }

    private:
        AmgxHelper()
        {
            AMGX_SAFE_CALL(AMGX_initialize());
        }

        ~AmgxHelper()
        {
            AMGX_SAFE_CALL(AMGX_finalize());
        }
};

namespace Opm::cuistl
{

template <class M, class X, class Y, int l>
CuDilu<M, X, Y, l>::CuDilu(const M& A, field_type w)
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

    AmgxHelper::ensureAmgxInitialized();
    // tested solvers: MULTICOLOR_GS   - works as expected, converges for spe1, seemingly converges on norne, run interrupted for being slow after some months
    //                 MULTICOLOR_DILU - does not converge on spe1
    // the config is taken from the test suite of amgx to ensure a valid set of parameters for DILU
    AMGX_SAFE_CALL(AMGX_config_create(&cfg, "config_version=2, solver(dilu_solv)=MULTICOLOR_DILU, dilu_solv:coloring_level=1, dilu_solv:matrix_coloring_scheme=MIN_MAX, dilu_solv:max_uncolored_percentage=0.15, dilu_solv:relaxation_factor=0.9, dilu_solv:max_iters=1"));
    AMGX_SAFE_CALL(AMGX_resources_create_simple(&rsrc, cfg));
    
    update();

}

template <class M, class X, class Y, int l>
CuDilu<M, X, Y, l>::~CuDilu(){
    AMGX_SAFE_CALL(AMGX_resources_destroy(rsrc));
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
}

template <class M, class X, class Y, int l>
void
CuDilu<M, X, Y, l>::pre([[maybe_unused]] X& x, [[maybe_unused]] Y& b)
{
}

template <class M, class X, class Y, int l>
void
CuDilu<M, X, Y, l>::apply(X& x, const Y& b)
{
    const auto numberOfRows = detail::to_int(m.N());
    const auto numberOfNonzeroBlocks = detail::to_int(m.nonzeroes());
    const auto blockSize = detail::to_int(m.blockSize());
    auto nonZeroValues = m.getNonZeroValues().data();
    auto rowIndices = m.getRowIndices().data();
    auto columnIndices = m.getColumnIndices().data();


    AMGX_SAFE_CALL(AMGX_vector_create(&amgx_x, rsrc, AMGX_mode_dDDI));
    AMGX_SAFE_CALL(AMGX_vector_create(&amgx_b, rsrc, AMGX_mode_dDDI));

    AMGX_SAFE_CALL(AMGX_solver_create(&amgx_solver, rsrc, AMGX_mode_dDDI, cfg));
    AMGX_SAFE_CALL(AMGX_matrix_create(&amgx_A, rsrc, AMGX_mode_dDDI));

    AMGX_SAFE_CALL(AMGX_matrix_upload_all(amgx_A, numberOfRows, numberOfNonzeroBlocks, blockSize, blockSize, rowIndices, columnIndices, nonZeroValues, NULL));
    AMGX_SAFE_CALL(AMGX_solver_setup(amgx_solver, amgx_A));
    
    AMGX_SAFE_CALL(AMGX_vector_upload(amgx_x, numberOfRows, blockSize, x.data()));
    AMGX_SAFE_CALL(AMGX_vector_upload(amgx_b, numberOfRows, blockSize, b.data()));

    AMGX_SAFE_CALL(AMGX_solver_solve(amgx_solver, amgx_b, amgx_x));
    AMGX_SAFE_CALL(AMGX_vector_download(amgx_x, x.data())); // put the data 

    AMGX_SAFE_CALL(AMGX_solver_destroy(amgx_solver)); // solver must be destroyed before matrix - amgx reference p.65
    AMGX_SAFE_CALL(AMGX_vector_destroy(amgx_x)); // recommended that vectors are destroyed before matrices - amgx reference p.112
    AMGX_SAFE_CALL(AMGX_vector_destroy(amgx_b));
    AMGX_SAFE_CALL(AMGX_matrix_destroy(amgx_A));
}

template <class M, class X, class Y, int l>
void
CuDilu<M, X, Y, l>::post([[maybe_unused]] X& x)
{
}

template <class M, class X, class Y, int l>
Dune::SolverCategory::Category
CuDilu<M, X, Y, l>::category() const
{
    return Dune::SolverCategory::sequential;
}

template <class M, class X, class Y, int l>
void
CuDilu<M, X, Y, l>::update()
{   
    m.updateNonzeroValues(detail::makeMatrixWithNonzeroDiagonal(m_underlyingMatrix));  

    // TODO: use AMGX_matrix_replace_coefficients here on A with the updated nnz values from m
}

} // namespace Opm::cuistl
#define INSTANTIATE_CuDilu_DUNE(realtype, blockdim)                                                                 \
    template class ::Opm::cuistl::CuDilu<Dune::BCRSMatrix<Dune::FieldMatrix<realtype, blockdim, blockdim>>,         \
                                            ::Opm::cuistl::CuVector<realtype>,                                         \
                                            ::Opm::cuistl::CuVector<realtype>>;                                        \
    template class ::Opm::cuistl::CuDilu<Dune::BCRSMatrix<Opm::MatrixBlock<realtype, blockdim, blockdim>>,          \
                                            ::Opm::cuistl::CuVector<realtype>,                                         \
                                            ::Opm::cuistl::CuVector<realtype>>



INSTANTIATE_CuDilu_DUNE(double, 1);
INSTANTIATE_CuDilu_DUNE(double, 2);
INSTANTIATE_CuDilu_DUNE(double, 3);
INSTANTIATE_CuDilu_DUNE(double, 4);
INSTANTIATE_CuDilu_DUNE(double, 5);
INSTANTIATE_CuDilu_DUNE(double, 6);

INSTANTIATE_CuDilu_DUNE(float, 1);
INSTANTIATE_CuDilu_DUNE(float, 2);
INSTANTIATE_CuDilu_DUNE(float, 3);
INSTANTIATE_CuDilu_DUNE(float, 4);
INSTANTIATE_CuDilu_DUNE(float, 5);
INSTANTIATE_CuDilu_DUNE(float, 6);