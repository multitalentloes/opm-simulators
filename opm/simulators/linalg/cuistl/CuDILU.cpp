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
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dune/common/fmatrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <fmt/core.h>
#include <opm/common/ErrorMacros.hpp>
#include <opm/simulators/linalg/cuistl/CuDILU.hpp>
#include <opm/simulators/linalg/cuistl/CuSparseMatrix.hpp>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <opm/simulators/linalg/cuistl/detail/cusparse_matrix_operations.hpp>
#include <opm/simulators/linalg/cuistl/detail/safe_conversion.hpp>
#include <opm/simulators/linalg/matrixblock.hh>
#include <vector>
#include <memory>
#include <iostream>

#include <cmath>
#include <cassert>
#include <chrono>
#include <string>
#define MEASURE_TIME_START(var_name) auto var_name##_start_time = std::chrono::high_resolution_clock::now();
#define MEASURE_TIME_END(var_name) auto var_name##_end_time = std::chrono::high_resolution_clock::now(); \
    auto var_name##_duration = std::chrono::duration_cast<std::chrono::milliseconds>(var_name##_end_time - var_name##_start_time); \
    std::cout << "Execution Time (" #var_name "): " << var_name##_duration.count() << " milliseconds" << std::endl;

template<class T>
void areVectorsEqual(const std::vector<T>& v1, const std::vector<T>& v2, T tolerance, std::string desc) {
    assert(v1.size() == v2.size());

    for (size_t i = 0; i < v1.size(); ++i) {
        if(std::abs(v1[i] - v2[i]) > tolerance){
            printf("%s: v[%ld / %ld] = (%lf != %lf)\n", desc, i, v1.size(), v1[i], v2[i]);
        }
    }
}

namespace
{
std::vector<int>
createReorderedToNatural(Opm::SparseTable<size_t> levelSets)
{
    auto res = std::vector<int>(Opm::cuistl::detail::to_size_t(levelSets.dataSize()));
    int globCnt = 0;
    for (auto row : levelSets) {
        for (auto col : row) {
            OPM_ERROR_IF(Opm::cuistl::detail::to_size_t(globCnt) >= res.size(),
                         fmt::format("Internal error. globCnt = {}, res.size() = {}", globCnt, res.size()));
            res[globCnt++] = static_cast<int>(col);
        }
    }
    return res;
}

std::vector<int>
createNaturalToReordered(Opm::SparseTable<size_t> levelSets)
{
    auto res = std::vector<int>(Opm::cuistl::detail::to_size_t(levelSets.dataSize()));
    int globCnt = 0;
    for (auto row : levelSets) {
        for (auto col : row) {
            OPM_ERROR_IF(Opm::cuistl::detail::to_size_t(globCnt) >= res.size(),
                         fmt::format("Internal error. globCnt = {}, res.size() = {}", globCnt, res.size()));
            res[col] = globCnt++;
        }
    }
    return res;
}

// TODO: When this function is called we already have the natural ordered matrix on the GPU
// TODO: could it be possible to create the reordered one in a kernel to speed up the constructor?
template <class M, class field_type>
Opm::cuistl::CuSparseMatrix<field_type>
createReorderedMatrix(const M& naturalMatrix, std::vector<int> reorderedToNatural)
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

/// @brief Naive conversion from bcrs on CPU to ELL on GPU to test apply performance
/// @tparam M 
/// @tparam field_type 
/// @param mat 
/// @return 
template <class M, class field_type>
void
BCSRtoELL(std::unique_ptr<Opm::cuistl::CuSparseMatrix<field_type>> &ptr, M mat, std::vector<int> naturalToReordered)
{

    /*
    given this blockmatrix on the CPU:
    +---+---+---+-+
    | 0  1 | 0  0 |
    | 2  3 | 0  0 |
    +---+---+---+-+
    | 8  9 |12 13 |
    |10 11 |14 15 |
    +---+---+---+-+
    |16 17 |20 21 |
    |18 19 |22 23 |
    +---+---+---+-+

    create a matrix on the GPU stored in this way:
    [0, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 19, NAN, 12, 20, NAN, 13, 21, NAN, 14, 22, NAN, 15, 23]
    This way, when a warp processes 32 consecutive rows, they will read consecutive
    values from memory when its reading out the matrix


        given this blockmatrix on the CPU:
    +---+---+---+-+
    | 0  1 | 0  0 |
    | 2  3 | 0  0 |
    +---+---+---+-+
    | 0  0 |12 13 |
    | 0  0 |14 15 |
    +---+---+---+-+
    |16 17 |20 21 |
    |18 19 |22 23 |
    +---+---+---+-+

    create a matrix on the GPU stored in this way:
    [0, 12, 16, 1, 13, 17, 2, 14, 18, 3, 15, 19, 0, 0, 20, 0, 0, 21, 0, 0, 22, 0, 0, 23]
    column pointers:
    [0, -1, 1, -1, 0, 1]
    */
    int ELLWidth = 0;
    size_t bs = M::block_type::cols;
    size_t n_rows = mat.N();
    for (auto row = mat.begin(); row != mat.end(); ++row) {
        int rowLength = 0;
        for (auto col = row->begin(); col != row->end(); ++col) {
            ++rowLength;
        }
        ELLWidth = std::max(ELLWidth, rowLength);
    }

    // allocate memory for the required datastructures of a cusparse matrix
    size_t n_ELLblocks = n_rows * ELLWidth;
    size_t n_ELLscalars = n_ELLblocks * bs * bs; // a dense ELLWidth * N matrix with blocks
    std::vector<field_type> nonZeroElements(n_ELLscalars);
    std::vector<int> rowIndices(n_rows+1); // is not actually used because ELLWidth is sufficient
    rowIndices[n_rows] = n_ELLblocks;
    std::vector<int> colIndices(n_ELLscalars, -1); // -1 represents invalid pointers

    // populate the allocated memory with the correct data
    for (auto row = mat.begin(); row != mat.end(); ++row) {
        size_t nrow = row.index();
        size_t column_filler = 0;
        for (auto col = row->begin(); col != row->end(); ++col, ++column_filler) {
            size_t ncol = col.index();
            colIndices[ELLWidth*nrow + column_filler] = ncol;
            for (size_t brow = 0; brow < bs; ++brow){
                for (size_t bcol = 0; bcol < bs; ++bcol){
                    // size_t idx = ncol*n_rows*bs*bs + (bcol + bs*brow)*n_rows + nrow;
                    size_t idx = column_filler*n_rows*bs*bs + (bcol + bs*brow)*n_rows + naturalToReordered[nrow];
                    nonZeroElements[idx] = (*col)[brow][bcol];
                }
            }
        }
    }

    // create the object
    ptr.reset(new Opm::cuistl::CuSparseMatrix<field_type>(nonZeroElements.data(), rowIndices.data(), colIndices.data(), n_ELLblocks, bs, n_rows));
}

} // NAMESPACE

namespace Opm::cuistl
{

template <class M, class X, class Y, int l>
CuDILU<M, X, Y, l>::CuDILU(const M& A)
    : m_cpuMatrix(A)
    , m_levelSets(Opm::getMatrixRowColoring(m_cpuMatrix, Opm::ColoringType::LOWER))
    , m_reorderedToNatural(createReorderedToNatural(m_levelSets))
    , m_naturalToReordered(createNaturalToReordered(m_levelSets))
    , m_gpuMatrix(CuSparseMatrix<field_type>::fromMatrix(m_cpuMatrix, true))
    , m_gpuMatrixReordered(createReorderedMatrix<M, field_type>(m_cpuMatrix, m_reorderedToNatural))
    , m_gpuNaturalToReorder(m_naturalToReordered)
    , m_gpuReorderToNatural(m_reorderedToNatural)
    , m_gpuDInv(m_gpuMatrix.N() * m_gpuMatrix.blockSize() * m_gpuMatrix.blockSize())

{
    // TODO: Should in some way verify that this matrix is symmetric, only do it debug mode?
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
    const int its = 10000;

    m_gpuMatrix.mv(d, v);
    MEASURE_TIME_START(cuda)
    for (int i = 0; i < its; ++i){
        m_gpuMatrix.mv(d, v);
    }
    MEASURE_TIME_END(cuda)
    X gpuCuRes(v);

    detail::ELLMV<field_type, blocksize_>(m_ELLGpuMatrix->getNonZeroValues().data(),
            m_ELLGpuMatrix->nonzeroes()/m_ELLGpuMatrix->N(),
            m_ELLGpuMatrix->N(),
            m_ELLGpuMatrix->getColumnIndices().data(),
            d.data(),
            v.data());
    MEASURE_TIME_START(ell)
    for (int i = 0; i < its; ++i){
        detail::ELLMV<field_type, blocksize_>(m_ELLGpuMatrix->getNonZeroValues().data(),
            m_ELLGpuMatrix->nonzeroes()/m_ELLGpuMatrix->N(),
            m_ELLGpuMatrix->N(),
            m_ELLGpuMatrix->getColumnIndices().data(),
            d.data(),
            v.data());
    }
    MEASURE_TIME_END(ell)
    X gpuELLRes(v);

    detail::bsrMV<field_type, blocksize_>(m_gpuMatrix.getNonZeroValues().data(),
                                    m_gpuMatrix.getRowIndices().data(),
                                    m_gpuMatrix.getColumnIndices().data(),
                                    m_gpuMatrix.N(),
                                    d.data(),
                                    v.data());
    MEASURE_TIME_START(bsr)
    for (int i = 0; i < its; ++i){
        detail::bsrMV<field_type, blocksize_>(m_gpuMatrix.getNonZeroValues().data(),
                                            m_gpuMatrix.getRowIndices().data(),
                                            m_gpuMatrix.getColumnIndices().data(),
                                            m_gpuMatrix.N(),
                                            d.data(),
                                            v.data());
    }
    MEASURE_TIME_END(bsr)
    X gpuBsrRes(v);

    std::vector<field_type> cpuCuRes(gpuCuRes.asStdVector());
    std::vector<field_type> cpuELLRes(gpuELLRes.asStdVector());
    std::vector<field_type> cpuBsrRes(gpuBsrRes.asStdVector());

    areVectorsEqual(cpuCuRes, cpuELLRes, (field_type)1e-4, "cuda vs ell");
    areVectorsEqual(cpuCuRes, cpuBsrRes, (field_type)1e-4, "cuda vs bsr");

    printf("BENCHMARK SUCCESSFULLY COMPLETED");
    // OPM_TIMEBLOCK(prec_apply);
    // int levelStartIdx = 0;
    // for (int level = 0; level < m_levelSets.size(); ++level) {
    //     const int numOfRowsInLevel = m_levelSets[level].size();
    //     detail::computeLowerSolveLevelSet<field_type, blocksize_>(m_gpuMatrixReordered.getNonZeroValues().data(),
    //                                                               m_gpuMatrixReordered.getRowIndices().data(),
    //                                                               m_gpuMatrixReordered.getColumnIndices().data(),
    //                                                               m_gpuReorderToNatural.data(),
    //                                                               levelStartIdx,
    //                                                               numOfRowsInLevel,
    //                                                               m_gpuDInv.data(),
    //                                                               d.data(),
    //                                                               v.data());

    //     // detail::computeELLLowerSolveLevelSet<field_type, blocksize_>(m_ELLGpuMatrix->getNonZeroValues().data(),
    //     //                                                           m_ELLGpuMatrix->nonzeroes()/m_ELLGpuMatrix->N(),
    //     //                                                           m_ELLGpuMatrix->N(),
    //     //                                                           m_ELLGpuMatrix->getColumnIndices().data(),
    //     //                                                           m_gpuReorderToNatural.data(),
    //     //                                                           levelStartIdx,
    //     //                                                           numOfRowsInLevel,
    //     //                                                           m_gpuDInv.data(),
    //     //                                                           d.data(),
    //     //                                                           v.data());
    //     levelStartIdx += numOfRowsInLevel;
    // }

    // levelStartIdx = m_cpuMatrix.N();
    // //  upper triangular solve: (D + U_A) v = Dy
    // for (int level = m_levelSets.size() - 1; level >= 0; --level) {
    //     const int numOfRowsInLevel = m_levelSets[level].size();
    //     levelStartIdx -= numOfRowsInLevel;
    //     detail::computeUpperSolveLevelSet<field_type, blocksize_>(m_gpuMatrixReordered.getNonZeroValues().data(),
    //                                                               m_gpuMatrixReordered.getRowIndices().data(),
    //                                                               m_gpuMatrixReordered.getColumnIndices().data(),
    //                                                               m_gpuReorderToNatural.data(),
    //                                                               levelStartIdx,
    //                                                               numOfRowsInLevel,
    //                                                               m_gpuDInv.data(),
    //                                                               v.data());
    // }
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

template <class M, class X, class Y, int l>
void
CuDILU<M, X, Y, l>::update()
{
    OPM_TIMEBLOCK(prec_update);

    // DEBUGGING TO TEST ELLPACK
    std::vector<int> noReorder(m_cpuMatrix.N());
    std::iota(noReorder.begin(), noReorder.end(), 0);
    BCSRtoELL<M, field_type>(m_ELLGpuMatrix, m_cpuMatrix, noReorder);
    // BCSRtoELL<M, field_type>(m_ELLGpuMatrix, m_cpuMatrix, m_naturalToReordered);
    // {
    //     if constexpr (std::is_same<field_type, double>::value && blocksize_ == 2) {
    //         const int N = 2, M_ = 3;
    //         constexpr int blocksize = 2;
    //         const int nonZeroes = 4;
    //         using MM = Dune::FieldMatrix<double, blocksize, blocksize>;
    //         using Vector = Dune::BlockVector<Dune::FieldVector<double, blocksize>>;
    //         using SpMatrix = Dune::BCRSMatrix<MM>;

    //         SpMatrix B(M_, N, nonZeroes, SpMatrix::row_wise);
    //         for (auto row = B.createbegin(); row != B.createend(); ++row) {
    //             if (row.index() == 0) {
    //                 row.insert(0);
    //             }
    //             else if (row.index() == 1){
    //                 // row.insert(0);
    //                 row.insert(1);
    //             }
    //             else if (row.index() == 2){
    //                 row.insert(0);
    //                 row.insert(1);
    //             }
    //         }

    //         B[0][0][0][0] = 0.0;
    //         B[0][0][0][1] = 1.0;
    //         B[0][0][1][0] = 2.0;
    //         B[0][0][1][1] = 3.0;

    //         B[1][1][0][0] = 12.0;
    //         B[1][1][0][1] = 13.0;
    //         B[1][1][1][0] = 14.0;
    //         B[1][1][1][1] = 15.0;

    //         B[2][0][0][0] = 16.0;
    //         B[2][0][0][1] = 17.0;
    //         B[2][0][1][0] = 18.0;
    //         B[2][0][1][1] = 19.0;

    //         B[2][1][0][0] = 20.0;
    //         B[2][1][0][1] = 21.0;
    //         B[2][1][1][0] = 22.0;
    //         B[2][1][1][1] = 23.0;
    //         BCSRtoELL(m_ELLGpuMatrix, B, {0, 1, 2});

    //         CuVector<double> v(6), d(std::vector<double>{1.0, 2.0, 3.0, 4.0});

    //         // std::vector<field_type> values(24);
    //         // std::vector<int> cols(6);
    //         // m_ELLGpuMatrix->getNonZeroValues().copyToHost(values);
    //         // m_ELLGpuMatrix->getColumnIndices().copyToHost(cols);

    //         // std::cout<<std::endl;
    //         // for (auto v : values){
    //         //     std::cout << v << " ";
    //         // }
    //         // std::cout<<std::endl;
    //         // for (auto v : cols){
    //         //     std::cout << v << " ";
    //         // }
    //         // std::cout<<std::endl;
    //         detail::ELLMV<double, 2>(m_ELLGpuMatrix->getNonZeroValues().data(), 2, 3, m_ELLGpuMatrix->getColumnIndices().data(), d.data(), v.data());
    //         Vector hv(3);
    //         v.copyToHost(hv);
    //         std::cout<<std::endl;
    //         for (auto val : hv){
    //             std::cout << val << " ";
    //         }
    //         exit(1);
    //     }
    // }

    m_gpuMatrix.updateNonzeroValues(m_cpuMatrix, true); // send updated matrix to the gpu

    detail::copyMatDataToReordered<field_type, blocksize_>(m_gpuMatrix.getNonZeroValues().data(),
                                                           m_gpuMatrix.getRowIndices().data(),
                                                           m_gpuMatrixReordered.getNonZeroValues().data(),
                                                           m_gpuMatrixReordered.getRowIndices().data(),
                                                           m_gpuNaturalToReorder.data(),
                                                           m_gpuMatrixReordered.N());

    int levelStartIdx = 0;
    for (int level = 0; level < m_levelSets.size(); ++level) {
        const int numOfRowsInLevel = m_levelSets[level].size();

        detail::computeDiluDiagonal<field_type, blocksize_>(m_gpuMatrixReordered.getNonZeroValues().data(),
                                                            m_gpuMatrixReordered.getRowIndices().data(),
                                                            m_gpuMatrixReordered.getColumnIndices().data(),
                                                            m_gpuReorderToNatural.data(),
                                                            m_gpuNaturalToReorder.data(),
                                                            levelStartIdx,
                                                            numOfRowsInLevel,
                                                            m_gpuDInv.data());
        levelStartIdx += numOfRowsInLevel;
    }
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
