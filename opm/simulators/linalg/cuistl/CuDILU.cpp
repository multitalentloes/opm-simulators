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
#include <opm/simulators/linalg/cuistl/CuDILU.hpp>
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

#include <chrono>
#include <unistd.h>

#define CUTIME(var)  cudaDeviceSynchronize(); auto var = std::chrono::high_resolution_clock::now();

std::vector<int> createReorderedToNatural(int, Opm::SparseTable<size_t>);
std::vector<int> createNaturalToReordered(int, Opm::SparseTable<size_t>);

template <class M>
void
printMatrix(M mat)
{
    for (auto row = mat.begin(); row != mat.end(); ++row) {
        for (auto elem = row->begin(); elem != row->end(); elem++) {
            auto matBlock = *elem;
            for (int brow = 0; brow < M::block_type::cols; brow++) {
                for (int bcol = 0; bcol < M::block_type::cols; bcol++) {
                    printf("%lf ", matBlock[brow][bcol]);
                }
            }
            printf("\n");
        }
    }
}

template <class field_type>
void
printCuMatrix(Opm::cuistl::CuSparseMatrix<field_type>* m)
{
    std::vector<field_type> v = m->getNonZeroValues().asStdVector();
    for (int i = 0; i < m->getNonZeroValues().dim() / (m->blockSize() * m->blockSize()); i++) {
        for (int j = 0; j < m->blockSize() * m->blockSize(); j++) {
            printf("%lf ", v[i * m->blockSize() * m->blockSize() + j]);
        }
        printf("\n");
    }
}

template <class T>
void
printCuVec(Opm::cuistl::CuVector<T> vec)
{
    auto cpuVec = vec.asStdVector();
    for (auto e : cpuVec)
        printf("%lf\n", e);
}

template <class V>
void
printDuneVec(V v, int blocksize)
{
    for (int i = 0; i < v.N(); i++) {
        for (int j = 0; j < blocksize; j++) {
            printf("%lf\n", v[i][j]);
        }
    }
}

template <class V>
void
printDuneBlockVec(V v, int blocksize)
{
    for (int i = 0; i < v.size(); i++) {
        for (int j = 0; j < blocksize; j++) {
            for (int k = 0; k < blocksize; k++) {
                printf("%lf\n", v[i][j][k]);
            }
        }
    }
}

// TODO: I think sending size is excessive
std::vector<int>
createReorderedToNatural(int size, Opm::SparseTable<size_t> levelSets)
{
    auto res = std::vector<int>(size);
    int globCnt = 0;
    for (int i = 0; i < levelSets.size(); i++) {
        for (size_t j = 0; j < levelSets[i].size(); j++) {
            res[globCnt++] = (int)levelSets[i][j];
        }
    }
    return res;
}

std::vector<int>
createNaturalToReordered(int size, Opm::SparseTable<size_t> levelSets)
{
    auto res = std::vector<int>(size);
    int globCnt = 0;
    for (int i = 0; i < levelSets.size(); i++) {
        for (size_t j = 0; j < levelSets[i].size(); j++) {
            res[levelSets[i][j]] = globCnt++;
        }
    }
    return res;
}

// TODO: When this function is called we already have the natural ordered matrix on the GPU
// TODO: could it be possible to create the reordered one in a kernel to speed up the constructor?
template <class M, class field_type>
Opm::cuistl::CuSparseMatrix<field_type>
createReorderedMatrix(M naturalMatrix, std::vector<int> reorderedToNatural)
{
    M reorderedMatrix(naturalMatrix.N(), naturalMatrix.N(), naturalMatrix.nonzeroes(), M::row_wise);
    for (auto dstRowIt = reorderedMatrix.createbegin(); dstRowIt != reorderedMatrix.createend(); ++dstRowIt) {
        auto srcRow = naturalMatrix.begin() + reorderedToNatural[dstRowIt.index()];
        // For elements in A
        for (auto elem = srcRow->begin(); elem != srcRow->end(); elem++) {
            dstRowIt.insert(elem.index());
        }
    }

    // TODO: There is probably a faster way to copy by copying whole rows at a time
    for (auto dstRowIt = reorderedMatrix.begin(); dstRowIt != reorderedMatrix.end(); ++dstRowIt) {
        auto srcRow = naturalMatrix.begin() + reorderedToNatural[dstRowIt.index()];
        for (auto elem = srcRow->begin(); elem != srcRow->end(); elem++) {
            reorderedMatrix[dstRowIt.index()][elem.index()] = *elem;
        }
    }

    return Opm::cuistl::CuSparseMatrix<field_type>::fromMatrix(reorderedMatrix, true);
}

namespace Opm::cuistl
{

template <class M, class X, class Y, int l>
CuDILU<M, X, Y, l>::CuDILU(const M& A, field_type w)
    : m_cpuMatrix(A)
    , m_levelSets(Opm::getMatrixRowColoring(m_cpuMatrix, Opm::ColoringType::LOWER))
    , m_relaxationFactor(w)
    , m_reorderedToNatural(createReorderedToNatural(m_cpuMatrix.N(), m_levelSets))
    , m_naturalToReordered(createNaturalToReordered(m_cpuMatrix.N(), m_levelSets))
    , m_gpuMatrix(CuSparseMatrix<field_type>::fromMatrix(m_cpuMatrix, true))
    , m_gpuMatrixReordered(createReorderedMatrix<M, field_type>(m_cpuMatrix, m_reorderedToNatural))
    , m_gpuNaturalToReorder(m_naturalToReordered)
    , m_gpuReorderToNatural(m_reorderedToNatural)
    , m_gpuDInv(m_gpuMatrix.N() * m_gpuMatrix.blockSize() * m_gpuMatrix.blockSize())

{
    // TODO: verify that matrix is symmetric
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
    update();
}

template <class M, class X, class Y, int l>
void
CuDILU<M, X, Y, l>::pre([[maybe_unused]] X& x, [[maybe_unused]] Y& b)
{
}

template <class M, class X, class Y, int l>
void
CuDILU<M, X, Y, l>::apply(X& v, const Y& d)
{
    // CUTIME(start_time);

    OPM_TIMEBLOCK(prec_apply);
    {
        OPM_TIMEBLOCK(lower_solve);
        int levelStartIdx = 0;
        for (int level = 0; level < m_levelSets.size(); ++level) {
            const int numOfRowsInLevel = m_levelSets[level].size();
            detail::computeLowerSolveLevelSet<field_type, matrix_type::block_type::cols>(m_gpuMatrixReordered.getNonZeroValues().data(),
                                              m_gpuMatrixReordered.getRowIndices().data(),
                                              m_gpuMatrixReordered.getColumnIndices().data(),
                                              m_gpuMatrixReordered.N(),
                                              m_gpuReorderToNatural.data(),
                                              levelStartIdx,
                                              numOfRowsInLevel,
                                              m_gpuDInv.data(),
                                              d.data(),
                                              v.data());
            levelStartIdx += numOfRowsInLevel;
        }
    }
    {
        OPM_TIMEBLOCK(upper_solve);
        int levelStartIdx = m_cpuMatrix.N();
        //  upper triangular solve: (D + U_A) v = Dy
        for (int level = m_levelSets.size() - 1; level >= 0; --level) {
            const int numOfRowsInLevel = m_levelSets[level].size();
            levelStartIdx -= numOfRowsInLevel;
            detail::computeUpperSolveLevelSet<field_type, matrix_type::block_type::cols>(m_gpuMatrixReordered.getNonZeroValues().data(),
                                              m_gpuMatrixReordered.getRowIndices().data(),
                                              m_gpuMatrixReordered.getColumnIndices().data(),
                                              m_gpuMatrixReordered.N(),
                                              m_gpuReorderToNatural.data(),
                                              levelStartIdx,
                                              numOfRowsInLevel,
                                              m_gpuDInv.data(),
                                              d.data(),
                                              v.data());
        }
    }

    // CUTIME(end_time);
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    // std::cout << "Apply: " << duration.count() << " microseconds." << std::endl;
}

template <class M, class X, class Y, int l>
void
CuDILU<M, X, Y, l>::post([[maybe_unused]] X& x)
{
}

template <class M, class X, class Y, int l>
Dune::SolverCategory::Category
CuDILU<M, X, Y, l>::category() const
{
    return Dune::SolverCategory::sequential;
}


//! the moveDataToReordered typically takes 182ms, whereas the computediagonal takes less than 7ms
template <class M, class X, class Y, int l>
void
CuDILU<M, X, Y, l>::update()
{
    // CUTIME(start_time);

    m_gpuMatrix.updateNonzeroValues(m_cpuMatrix, true); // send updated matrix to the gpu

    detail::copyMatDataToReordered<field_type, matrix_type::block_type::cols>(m_gpuMatrix.getNonZeroValues().data(),
                                    m_gpuMatrix.getRowIndices().data(),
                                    m_gpuMatrix.getColumnIndices().data(),
                                    m_gpuMatrixReordered.getNonZeroValues().data(),
                                    m_gpuMatrixReordered.getRowIndices().data(),
                                    m_gpuMatrixReordered.getColumnIndices().data(),
                                    m_gpuNaturalToReorder.data(),
                                    m_gpuMatrixReordered.N());

    int levelStartIdx = 0;
    for (int level = 0; level < m_levelSets.size(); ++level) {
        const int numOfRowsInLevel = m_levelSets[level].size();

        detail::computeDiluDiagonal<field_type, matrix_type::block_type::cols>(m_gpuMatrixReordered.getNonZeroValues().data(),
                                    m_gpuMatrixReordered.getRowIndices().data(),
                                    m_gpuMatrixReordered.getColumnIndices().data(),
                                    m_gpuMatrixReordered.N(),
                                    m_gpuReorderToNatural.data(),
                                    m_gpuNaturalToReorder.data(),
                                    levelStartIdx,
                                    numOfRowsInLevel,
                                    m_gpuDInv.data());
        levelStartIdx += numOfRowsInLevel;
    }


    // CUTIME(end_time);
    // auto full = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    // std::cout << "Update: " << full.count() << "\n";

}

} // namespace Opm::cuistl
#define INSTANTIATE_CUDILU_DUNE(realtype, blockdim)                                                                    \
    template class ::Opm::cuistl::CuDILU<Dune::BCRSMatrix<Dune::FieldMatrix<realtype, blockdim, blockdim>>,            \
                                         ::Opm::cuistl::CuVector<realtype>,                                            \
                                         ::Opm::cuistl::CuVector<realtype>>;                                           \
    template class ::Opm::cuistl::CuDILU<Dune::BCRSMatrix<Opm::MatrixBlock<realtype, blockdim, blockdim>>,             \
                                         ::Opm::cuistl::CuVector<realtype>,                                            \
                                         ::Opm::cuistl::CuVector<realtype>>

INSTANTIATE_CUDILU_DUNE(double, 1);
INSTANTIATE_CUDILU_DUNE(double, 2);
INSTANTIATE_CUDILU_DUNE(double, 3);
INSTANTIATE_CUDILU_DUNE(double, 4);
INSTANTIATE_CUDILU_DUNE(double, 5);
INSTANTIATE_CUDILU_DUNE(double, 6);

INSTANTIATE_CUDILU_DUNE(float, 1);
INSTANTIATE_CUDILU_DUNE(float, 2);
INSTANTIATE_CUDILU_DUNE(float, 3);
INSTANTIATE_CUDILU_DUNE(float, 4);
INSTANTIATE_CUDILU_DUNE(float, 5);
INSTANTIATE_CUDILU_DUNE(float, 6);
