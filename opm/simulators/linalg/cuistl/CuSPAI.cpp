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
#include <opm/simulators/linalg/cuistl/CuSPAI.hpp>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <opm/simulators/linalg/cuistl/PreconditionerAdapter.hpp>
#include <opm/simulators/linalg/cuistl/detail/CuBlasHandle.hpp>
#include <opm/simulators/linalg/cuistl/detail/cublas_safe_call.hpp>
#include <opm/simulators/linalg/cuistl/detail/cublas_wrapper.hpp>
#include <opm/simulators/linalg/cuistl/detail/cusparse_constants.hpp>
#include <opm/simulators/linalg/cuistl/detail/cusparse_matrix_operations.hpp>
#include <opm/simulators/linalg/cuistl/detail/cusparse_safe_call.hpp>
#include <opm/simulators/linalg/cuistl/detail/cusparse_wrapper.hpp>
#include <opm/simulators/linalg/cuistl/detail/fix_zero_diagonal.hpp>
#include <opm/simulators/linalg/cuistl/detail/safe_conversion.hpp>
#include <opm/simulators/linalg/cuistl/detail/vector_operations.hpp>
#include <opm/simulators/linalg/matrixblock.hh>

namespace Opm::cuistl
{

template <class M, class X, class Y, int l>
CuSPAI<M, X, Y, l>::CuSPAI(const M& A, field_type w)
    : m_cpuMatrix(A)
    , m_relaxationFactor(w)
    , m_gpuMatrix(CuSparseMatrix<field_type>::fromMatrix(A))
    , m_diagInvFlattened(m_gpuMatrix.N() * m_gpuMatrix.blockSize() * m_gpuMatrix.blockSize())
{
    // Some sanity check
    OPM_ERROR_IF(A.N() != m_gpuMatrix.N(),
                 fmt::format("CuSparse matrix not same size as DUNE matrix. {} vs {}.", m_gpuMatrix.N(), A.N()));
    OPM_ERROR_IF(A[0][0].N() != m_gpuMatrix.blockSize(),
                 fmt::format("CuSparse matrix not same blocksize as DUNE matrix. {} vs {}.",
                             m_gpuMatrix.blockSize(),
                             A[0][0].N()));
    OPM_ERROR_IF(A.N() * A[0][0].N() != m_gpuMatrix.dim(),
                 fmt::format("CuSparse matrix not same dimension as DUNE matrix. {} vs {}.",
                             m_gpuMatrix.dim(),
                             A.N() * A[0][0].N()));
    OPM_ERROR_IF(A.nonzeroes() != m_gpuMatrix.nonzeroes(),
                 fmt::format("CuSparse matrix not same number of non zeroes as DUNE matrix. {} vs {}. ",
                             m_gpuMatrix.nonzeroes(),
                             A.nonzeroes()));

    const unsigned int bs = blocksize_;
    this->Nb = m_cpuMatrix.N();
    this->N = Nb * bs;
    this->nnzb = m_cpuMatrix.nonzeroes();
    this->nnz = nnzb * bs * bs;

    submat.resize(Nb);
    eyeBlockIndices.resize(Nb);
    submatValsPositions.resize(Nb);
    rowPointers.resize(Nb + 1);
    spaiColPointers.resize(Nb + 1);
    colIndices.resize(nnzb);

    // std::copy(m_cpuMatrix.getRowIndices().begin(), m_cpuMatrix.getRowIndices().begin() + Nb + 1, rowPointers.begin());
    // std::copy(m_cpuMatrix.getColumnIndices().begin(), m_cpuMatrix.getColumnIndices().begin() + nnzb, colIndices.begin());
    colIndices.reserve(nnzb);
    rowPointers.reserve(Nb + 1);
    for (auto& row : m_cpuMatrix) {
        for (auto columnIterator = row.begin(); columnIterator != row.end(); ++columnIterator) {
            colIndices.push_back(columnIterator.index());
        }
        rowPointers.push_back(detail::to_int(colIndices.size()));
    }

    // Dune::Timer t_analyze_matrix;

    for(int tcol = 0; tcol < Nb; tcol++){
        buildIJSets(tcol);

        submatIndices.clear();
        submatPointers.assign(iset.size() + 1, 0);

        unsigned int i = 1;
        for(auto rit = iset.begin(); rit != iset.end(); ++rit){
            auto fcol = colIndices.begin() + rowPointers[*rit];
            auto lcol = colIndices.begin() + rowPointers[*rit + 1];

            for(auto cit = fcol; cit != lcol; ++cit){
                if(jset.count(*cit)){
                    submatIndices.push_back(*cit);
                    submatValsPositions[tcol].resize(submatValsPositions[tcol].size() + bs * bs);
                    std::iota(submatValsPositions[tcol].end() - bs * bs, submatValsPositions[tcol].end(), (rowPointers[*rit] + cit - fcol) * bs * bs);
                }
            }

            submatPointers[i] = submatIndices.size();
            i++;
        }

        submat[tcol].setSize(iset.size(), jset.size(), submatPointers.back());
        submat[tcol].setBuildMode(DuneMat::row_wise);

        gatherSubmatIndices();

        for(typename DuneMat::CreateIterator row = submat[tcol].createbegin(); row != submat[tcol].createend(); ++row){
            for(int p = submatPointers[row.index()]; p < submatPointers[row.index() + 1]; ++p){
                row.insert(submatIndices[p]);
            }
        }

        spaiColPointers[tcol + 1] = spaiColPointers[tcol] + jset.size();
        spaiRowIndices.insert(spaiRowIndices.end(), jset.begin(), jset.end());
        eyeBlockIndices[tcol] = std::distance(iset.begin(), std::find(iset.begin(), iset.end(), tcol));
    }

    spaiNnzValues.resize(spaiColPointers.back() * bs * bs);

    /*
        TODO:
            move spaiNnzValues to the GPU
            move spaiColPointers to the GPU
            move spaiRowIndices to the GPU
    */
}

template <class M, class X, class Y, int l>
void
CuSPAI<M, X, Y, l>::pre([[maybe_unused]] X& x, [[maybe_unused]] Y& b)
{
}

template <class M, class X, class Y, int l>
void
CuSPAI<M, X, Y, l>::apply(X& v, const Y& d)
{
    m_gpuMatrix.mv(d, v);
}

template <class M, class X, class Y, int l>
void
CuSPAI<M, X, Y, l>::post([[maybe_unused]] X& x)
{
}

template <class M, class X, class Y, int l>
Dune::SolverCategory::Category
CuSPAI<M, X, Y, l>::category() const
{
    return Dune::SolverCategory::sequential;
}

template <class M, class X, class Y, int l>
void
CuSPAI<M, X, Y, l>::update()
{
    m_gpuMatrix.updateNonzeroValues(m_cpuMatrix);


    const unsigned int bs = blocksize_;
    int count;

    // solver.setBlocked(bs > 1);

    // for(int tcol = 0; tcol < Nb; tcol++){
    //     count = 0;
    //     for(auto row = submat[tcol].begin(); row != submat[tcol].end(); ++row){
    //         for(auto col = (*row).begin(); col != (*row).end(); ++col){
    //             for(auto br = (*col).begin(); br != (*col).end(); ++br){
    //                 for(auto bc = (*br).begin(); bc != (*br).end(); ++bc){
    //                     (*bc) = mat->nnzValues[submatValsPositions[tcol][count]];
    //                     ++count;
    //                 }
    //             }
    //         }
    //     }

    //     sol.resize(submat[tcol].M());
    //     rhs.resize(submat[tcol].N());
    //     rhs = 0;
    //     rhs[eyeBlockIndices[tcol]] = Dune::ScaledIdentityMatrix<double, bs>(1);

    //     solver.setMatrix(submat[tcol]);
    //     solver.apply(sol, rhs, res);

    //     for(unsigned int i = 0; i < submat[tcol].M(); i++){
    //         for(unsigned int j = 0; j < bs; j++){
    //             for(unsigned int k = 0; k < bs; k++){
    //                 spaiNnzValues[(spaiColPointers[tcol] + i) * bs * bs + j * bs + k] = sol[i][j][k];
    //             }
    //         }
    //     }
    // }
}

template <class M, class X, class Y, int l>
void
CuSPAI<M, X, Y, l>::buildIJSets(int tcol)
{
    jset.clear();

    auto fcol = colIndices.begin() + rowPointers[tcol];
    auto lcol = colIndices.begin() + rowPointers[tcol + 1];
    jset.insert(fcol, lcol);

    for(int f = 0; f <= fill_in; f++){
        iset.clear();

        for(auto it = jset.begin(); it != jset.end(); ++it){
            auto frow = colIndices.begin() + rowPointers[*it];
            auto lrow = colIndices.begin() + rowPointers[*it + 1];
            iset.insert(frow, lrow);
        }

        if(f < fill_in){
            jset = iset;
        }
    }
}

template <class M, class X, class Y, int l>
void
CuSPAI<M, X, Y, l>::gatherSubmatIndices()
{
    std::vector<int> tmp(submatIndices);
    std::transform(tmp.begin(), tmp.end(), submatIndices.begin(),
        [=](int i){return std::distance(jset.begin(), std::find(jset.begin(), jset.end(), i));});
}

} // namespace Opm::cuistl
#define INSTANTIATE_CUSPAI_DUNE(realtype, blockdim)                                                                     \
    template class ::Opm::cuistl::CuSPAI<Dune::BCRSMatrix<Dune::FieldMatrix<realtype, blockdim, blockdim>>,             \
                                        ::Opm::cuistl::CuVector<realtype>,                                             \
                                        ::Opm::cuistl::CuVector<realtype>>;                                            \
    template class ::Opm::cuistl::CuSPAI<Dune::BCRSMatrix<Opm::MatrixBlock<realtype, blockdim, blockdim>>,              \
                                        ::Opm::cuistl::CuVector<realtype>,                                             \
                                        ::Opm::cuistl::CuVector<realtype>>

INSTANTIATE_CUSPAI_DUNE(double, 1);
INSTANTIATE_CUSPAI_DUNE(double, 2);
INSTANTIATE_CUSPAI_DUNE(double, 3);
INSTANTIATE_CUSPAI_DUNE(double, 4);
INSTANTIATE_CUSPAI_DUNE(double, 5);
INSTANTIATE_CUSPAI_DUNE(double, 6);

INSTANTIATE_CUSPAI_DUNE(float, 1);
INSTANTIATE_CUSPAI_DUNE(float, 2);
INSTANTIATE_CUSPAI_DUNE(float, 3);
INSTANTIATE_CUSPAI_DUNE(float, 4);
INSTANTIATE_CUSPAI_DUNE(float, 5);
INSTANTIATE_CUSPAI_DUNE(float, 6);
