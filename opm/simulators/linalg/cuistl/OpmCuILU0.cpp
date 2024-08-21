/*
  Copyright 2024 SINTEF AS

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
#include <chrono>
#include <config.h>
#include <dune/common/fmatrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <fmt/core.h>
#include <limits>
#include <opm/common/ErrorMacros.hpp>
#include <opm/common/TimingMacros.hpp>
#include <opm/simulators/linalg/GraphColoring.hpp>
#include <opm/simulators/linalg/cuistl/CuSparseMatrix.hpp>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <opm/simulators/linalg/cuistl/OpmCuILU0.hpp>
#include <opm/simulators/linalg/cuistl/detail/coloringAndReorderingUtils.hpp>
#include <opm/simulators/linalg/cuistl/detail/cusparse_matrix_operations.hpp>
#include <opm/simulators/linalg/cuistl/detail/preconditionerKernels/ILU0Kernels.hpp>
#include <opm/simulators/linalg/matrixblock.hh>
#include <tuple>
#include <type_traits>

class CumulativeScopeTimer {
public:
    // Constructor starts the timer
    CumulativeScopeTimer() : start_time(std::chrono::high_resolution_clock::now()) {
        ++instance_count_2;
    }

    // Destructor stops the timer and accumulates the time
    ~CumulativeScopeTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        total_time_spent_2 += duration;

        std::cout << "Average: " << total_time_spent_2 / instance_count_2 << "us/apply. This apply: " << duration << "us. Total time spent: " << total_time_spent_2 / 1000.0 << " ms. " << " Applies: " << instance_count_2 << std::endl;
    }

    // Static method to report the cumulative time and instance count
    static void report() {
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;  // Time when the timer started
    static long long total_time_spent_2;  // Cumulative time spent in all instances
    static int instance_count_2;  // Number of times the timer has been instantiated
};

// Static member variables need to be defined outside the class
long long CumulativeScopeTimer::total_time_spent_2 = 0;
int CumulativeScopeTimer::instance_count_2 = 0;

class CumulativeScopeTimer2 {
public:
    // Constructor starts the timer
    CumulativeScopeTimer2() : start_time(std::chrono::high_resolution_clock::now()) {
        ++instance_count_3;
    }

    // Destructor stops the timer and accumulates the time
    ~CumulativeScopeTimer2() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        total_time_spent_3 += duration;

        std::cout << "Average: " << total_time_spent_3 / instance_count_3 << "us/update. This update: " << duration << "us. Total time spent: " << total_time_spent_3 / 1000.0 << " ms. " << " Applies: " << instance_count_3 << std::endl;
    }

    // Static method to report the cumulative time and instance count
    static void report() {
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;  // Time when the timer started
    static long long total_time_spent_3;  // Cumulative time spent in all instances
    static int instance_count_3;  // Number of times the timer has been instantiated
};

// Static member variables need to be defined outside the class
long long CumulativeScopeTimer2::total_time_spent_3 = 0;
int CumulativeScopeTimer2::instance_count_3 = 0;
namespace Opm::cuistl
{

template <class M, class X, class Y, int l>
OpmCuILU0<M, X, Y, l>::OpmCuILU0(const M& A, bool splitMatrix, bool tuneKernels, bool float_ILU, bool float_ILU_off_diags, bool float_ILU_float_compute)
    : m_cpuMatrix(A)
    , m_levelSets(Opm::getMatrixRowColoring(m_cpuMatrix, Opm::ColoringType::LOWER))
    , m_reorderedToNatural(detail::createReorderedToNatural(m_levelSets))
    , m_naturalToReordered(detail::createNaturalToReordered(m_levelSets))
    , m_gpuMatrix(CuSparseMatrix<field_type>::fromMatrix(m_cpuMatrix, true))
    , m_gpuMatrixReorderedLower(nullptr)
    , m_gpuMatrixReorderedUpper(nullptr)
    , m_gpuMatrixReorderedLowerfloat(nullptr)
    , m_gpuMatrixReorderedUpperfloat(nullptr)
    , m_gpuNaturalToReorder(m_naturalToReordered)
    , m_gpuReorderToNatural(m_reorderedToNatural)
    , m_gpuDInv(m_gpuMatrix.N() * m_gpuMatrix.blockSize() * m_gpuMatrix.blockSize())
    , m_splitMatrix(splitMatrix)
    , m_tuneThreadBlockSizes(tuneKernels)
    , m_float_ILU(float_ILU)
    , m_float_ILU_off_diags(float_ILU_off_diags)
    , m_float_ILU_float_compute(float_ILU_float_compute)
    , m_gpuMatrixReorderedDiagfloat(m_gpuMatrix.N() * m_gpuMatrix.blockSize() * m_gpuMatrix.blockSize())
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
    if (m_splitMatrix) {
        m_gpuMatrixReorderedDiag.emplace(CuVector<field_type>(blocksize_ * blocksize_ * m_cpuMatrix.N()));
        std::tie(m_gpuMatrixReorderedLower, m_gpuMatrixReorderedUpper)
            = detail::extractLowerAndUpperMatrices<M, field_type, CuSparseMatrix<field_type>>(m_cpuMatrix,
                                                                                              m_reorderedToNatural);
    } else {
        m_gpuReorderedLU = detail::createReorderedMatrix<M, field_type, CuSparseMatrix<field_type>>(
            m_cpuMatrix, m_reorderedToNatural);
    }

    // only one of these may be selected at a time
    assert(!(m_float_ILU && m_float_ILU_off_diags));
    bool using_mixed = (m_float_ILU || m_float_ILU_off_diags || m_float_ILU_float_compute);
    if (using_mixed){
        assert(m_splitMatrix);
        m_gpuMatrixReorderedLowerfloat = std::unique_ptr<floatMat>(new auto(floatMat(m_gpuMatrixReorderedLower->getRowIndices(), m_gpuMatrixReorderedLower->getColumnIndices(), blocksize_)));
        m_gpuMatrixReorderedUpperfloat = std::unique_ptr<floatMat>(new auto(floatMat(m_gpuMatrixReorderedUpper->getRowIndices(), m_gpuMatrixReorderedUpper->getColumnIndices(), blocksize_)));
        m_gpuMatrixReorderedDiagfloat.emplace(CuVector<float>(m_gpuMatrix.N() * m_gpuMatrix.blockSize() * m_gpuMatrix.blockSize()));
    }

    LUFactorizeAndMoveData();

#ifdef USE_HIP
    if (m_tuneThreadBlockSizes) {
        tuneThreadBlockSizes();
    }
#endif
}

template <class M, class X, class Y, int l>
void
OpmCuILU0<M, X, Y, l>::pre([[maybe_unused]] X& x, [[maybe_unused]] Y& b)
{
}

template <class M, class X, class Y, int l>
void
OpmCuILU0<M, X, Y, l>::apply(X& v, const Y& d)
{
    cudaDeviceSynchronize();
    CumulativeScopeTimer timer1;
    OPM_TIMEBLOCK(prec_apply);
    {
        int levelStartIdx = 0;
        for (int level = 0; level < m_levelSets.size(); ++level) {
            const int numOfRowsInLevel = m_levelSets[level].size();
            if (m_splitMatrix || m_float_ILU_off_diags || m_float_ILU_float_compute) {
                if (m_float_ILU) { // use float for all ILU things
                    detail::ILU0::solveLowerLevelSetSplitMixed<field_type, blocksize_>(
                        m_gpuMatrixReorderedLowerfloat->getNonZeroValues().data(),
                        m_gpuMatrixReorderedLowerfloat->getRowIndices().data(),
                        m_gpuMatrixReorderedLowerfloat->getColumnIndices().data(),
                        m_gpuReorderToNatural.data(),
                        levelStartIdx,
                        numOfRowsInLevel,
                        d.data(),
                        v.data(),
                        m_applyThreadBlockSize);
                }
                else if (m_float_ILU_float_compute){
                    detail::ILU0::solveLowerLevelSetSplitMixedFloatCompute<field_type, blocksize_>(
                        m_gpuMatrixReorderedLowerfloat->getNonZeroValues().data(),
                        m_gpuMatrixReorderedLowerfloat->getRowIndices().data(),
                        m_gpuMatrixReorderedLowerfloat->getColumnIndices().data(),
                        m_gpuReorderToNatural.data(),
                        levelStartIdx,
                        numOfRowsInLevel,
                        d.data(),
                        v.data(),
                        m_applyThreadBlockSize);
                }
                else{
                    detail::ILU0::solveLowerLevelSetSplit<field_type, blocksize_>(
                        m_gpuMatrixReorderedLower->getNonZeroValues().data(),
                        m_gpuMatrixReorderedLower->getRowIndices().data(),
                        m_gpuMatrixReorderedLower->getColumnIndices().data(),
                        m_gpuReorderToNatural.data(),
                        levelStartIdx,
                        numOfRowsInLevel,
                        d.data(),
                        v.data(),
                        m_applyThreadBlockSize);
                }
            } else {
                detail::ILU0::solveLowerLevelSet<field_type, blocksize_>(m_gpuReorderedLU->getNonZeroValues().data(),
                                                                         m_gpuReorderedLU->getRowIndices().data(),
                                                                         m_gpuReorderedLU->getColumnIndices().data(),
                                                                         m_gpuReorderToNatural.data(),
                                                                         levelStartIdx,
                                                                         numOfRowsInLevel,
                                                                         d.data(),
                                                                         v.data(),
                                                                         m_applyThreadBlockSize);
            }
            levelStartIdx += numOfRowsInLevel;
        }

        levelStartIdx = m_cpuMatrix.N();
        for (int level = m_levelSets.size() - 1; level >= 0; --level) {
            const int numOfRowsInLevel = m_levelSets[level].size();
            levelStartIdx -= numOfRowsInLevel;
            if (m_splitMatrix) {
                if (m_float_ILU){
                    detail::ILU0::solveUpperLevelSetSplitFloatILU<field_type, blocksize_>(
                        m_gpuMatrixReorderedUpperfloat->getNonZeroValues().data(),
                        m_gpuMatrixReorderedUpperfloat->getRowIndices().data(),
                        m_gpuMatrixReorderedUpperfloat->getColumnIndices().data(),
                        m_gpuReorderToNatural.data(),
                        levelStartIdx,
                        numOfRowsInLevel,
                        m_gpuMatrixReorderedDiagfloat.value().data(),
                        v.data(),
                        m_applyThreadBlockSize);
                }
                else if (m_float_ILU_off_diags){
                    detail::ILU0::solveUpperLevelSetSplitFloatOffDiag<field_type, blocksize_>(
                        m_gpuMatrixReorderedUpperfloat->getNonZeroValues().data(),
                        m_gpuMatrixReorderedUpperfloat->getRowIndices().data(),
                        m_gpuMatrixReorderedUpperfloat->getColumnIndices().data(),
                        m_gpuReorderToNatural.data(),
                        levelStartIdx,
                        numOfRowsInLevel,
                        m_gpuMatrixReorderedDiag.value().data(),
                        v.data(),
                        m_applyThreadBlockSize);
                }
                else if (m_float_ILU_float_compute) {
                    detail::ILU0::solveUpperLevelSetSplitFloatILUFloatCompute<field_type, blocksize_>(
                        m_gpuMatrixReorderedUpperfloat->getNonZeroValues().data(),
                        m_gpuMatrixReorderedUpperfloat->getRowIndices().data(),
                        m_gpuMatrixReorderedUpperfloat->getColumnIndices().data(),
                        m_gpuReorderToNatural.data(),
                        levelStartIdx,
                        numOfRowsInLevel,
                        m_gpuMatrixReorderedDiagfloat.value().data(),
                        v.data(),
                        m_applyThreadBlockSize);
                }
                else{
                    detail::ILU0::solveUpperLevelSetSplit<field_type, blocksize_>(
                        m_gpuMatrixReorderedUpper->getNonZeroValues().data(),
                        m_gpuMatrixReorderedUpper->getRowIndices().data(),
                        m_gpuMatrixReorderedUpper->getColumnIndices().data(),
                        m_gpuReorderToNatural.data(),
                        levelStartIdx,
                        numOfRowsInLevel,
                        m_gpuMatrixReorderedDiag.value().data(),
                        v.data(),
                        m_applyThreadBlockSize);
                }
            } else {
                detail::ILU0::solveUpperLevelSet<field_type, blocksize_>(m_gpuReorderedLU->getNonZeroValues().data(),
                                                                         m_gpuReorderedLU->getRowIndices().data(),
                                                                         m_gpuReorderedLU->getColumnIndices().data(),
                                                                         m_gpuReorderToNatural.data(),
                                                                         levelStartIdx,
                                                                         numOfRowsInLevel,
                                                                         v.data(),
                                                                         m_applyThreadBlockSize);
            }
        }
    }
    cudaDeviceSynchronize();
}

template <class M, class X, class Y, int l>
void
OpmCuILU0<M, X, Y, l>::post([[maybe_unused]] X& x)
{
}

template <class M, class X, class Y, int l>
Dune::SolverCategory::Category
OpmCuILU0<M, X, Y, l>::category() const
{
    return Dune::SolverCategory::sequential;
}

template <class M, class X, class Y, int l>
void
OpmCuILU0<M, X, Y, l>::update()
{
    OPM_TIMEBLOCK(prec_update);
    {
        cudaDeviceSynchronize();
        CumulativeScopeTimer2 timer2;
        m_gpuMatrix.updateNonzeroValues(m_cpuMatrix, true); // send updated matrix to the gpu
        LUFactorizeAndMoveData();
        cudaDeviceSynchronize();
    }
}

template <class M, class X, class Y, int l>
void
OpmCuILU0<M, X, Y, l>::LUFactorizeAndMoveData()
{
    OPM_TIMEBLOCK(prec_update);
    {
        if (m_splitMatrix) {
            detail::copyMatDataToReorderedSplit<field_type, blocksize_>(
                m_gpuMatrix.getNonZeroValues().data(),
                m_gpuMatrix.getRowIndices().data(),
                m_gpuMatrix.getColumnIndices().data(),
                m_gpuMatrixReorderedLower->getNonZeroValues().data(),
                m_gpuMatrixReorderedLower->getRowIndices().data(),
                m_gpuMatrixReorderedUpper->getNonZeroValues().data(),
                m_gpuMatrixReorderedUpper->getRowIndices().data(),
                m_gpuMatrixReorderedDiag.value().data(),
                m_gpuNaturalToReorder.data(),
                m_gpuMatrixReorderedLower->N(),
                m_updateThreadBlockSize);
        } else {
            detail::copyMatDataToReordered<field_type, blocksize_>(m_gpuMatrix.getNonZeroValues().data(),
                                                                   m_gpuMatrix.getRowIndices().data(),
                                                                   m_gpuReorderedLU->getNonZeroValues().data(),
                                                                   m_gpuReorderedLU->getRowIndices().data(),
                                                                   m_gpuNaturalToReorder.data(),
                                                                   m_gpuReorderedLU->N(),
                                                                   m_updateThreadBlockSize);
        }
        int levelStartIdx = 0;
        for (int level = 0; level < m_levelSets.size(); ++level) {
            const int numOfRowsInLevel = m_levelSets[level].size();
            bool is_mixed_precision = (m_float_ILU || m_float_ILU_off_diags || m_float_ILU_float_compute);
            if (m_splitMatrix) {
                detail::ILU0::LUFactorizationSplit<blocksize_, field_type, float>(
                    m_gpuMatrixReorderedLower->getNonZeroValues().data(),
                    m_gpuMatrixReorderedLower->getRowIndices().data(),
                    m_gpuMatrixReorderedLower->getColumnIndices().data(),
                    m_gpuMatrixReorderedUpper->getNonZeroValues().data(),
                    m_gpuMatrixReorderedUpper->getRowIndices().data(),
                    m_gpuMatrixReorderedUpper->getColumnIndices().data(),
                    m_gpuMatrixReorderedDiag.value().data(),
                    is_mixed_precision ? m_gpuMatrixReorderedLowerfloat->getNonZeroValues().data() : nullptr,
                    is_mixed_precision ? m_gpuMatrixReorderedUpperfloat->getNonZeroValues().data() : nullptr,
                    is_mixed_precision ? m_gpuMatrixReorderedDiagfloat.value().data() : nullptr,
                    m_gpuReorderToNatural.data(),
                    m_gpuNaturalToReorder.data(),
                    levelStartIdx,
                    numOfRowsInLevel,
                    m_updateThreadBlockSize,
                    is_mixed_precision);

            } else {
                detail::ILU0::LUFactorization<field_type, blocksize_>(m_gpuReorderedLU->getNonZeroValues().data(),
                                                                      m_gpuReorderedLU->getRowIndices().data(),
                                                                      m_gpuReorderedLU->getColumnIndices().data(),
                                                                      m_gpuNaturalToReorder.data(),
                                                                      m_gpuReorderToNatural.data(),
                                                                      numOfRowsInLevel,
                                                                      levelStartIdx,
                                                                      m_updateThreadBlockSize);
            }
            levelStartIdx += numOfRowsInLevel;
        }
    }

    // // mixed precision only makes sense if this is instantiated on a double
    // if constexpr(std::is_same_v<double, field_type>){
    //     // cast off-diagonals to float
    //     if (m_float_ILU_off_diags || m_float_ILU || m_float_ILU_float_compute){
    //         {
    //             auto elements = m_gpuMatrixReorderedLower->getNonZeroValues().asStdVector();
    //             auto rows = m_gpuMatrixReorderedLower->getRowIndices().asStdVector();
    //             auto cols = m_gpuMatrixReorderedLower->getColumnIndices().asStdVector();

    //             size_t idx = 0;
    //             std::vector<float> floatElements(elements.size());
    //             for (auto v : elements) {
    //                 floatElements[idx++] = float(v);
    //             }
    //             m_gpuMatrixReorderedLowerfloat = std::unique_ptr<floatMat>(new auto(floatMat(floatElements.data(), rows.data(), cols.data(), m_gpuMatrixReorderedLower->nonzeroes(), blocksize_, m_gpuMatrix.N())));
    //         }
    //         {
    //         auto elements = m_gpuMatrixReorderedUpper->getNonZeroValues().asStdVector();
    //         auto rows = m_gpuMatrixReorderedUpper->getRowIndices().asStdVector();
    //         auto cols = m_gpuMatrixReorderedUpper->getColumnIndices().asStdVector();

    //         size_t idx = 0;
    //         std::vector<float> floatElements(elements.size());
    //         for (auto v : elements) {
    //             floatElements[idx++] = float(v);
    //         }
    //         m_gpuMatrixReorderedUpperfloat = std::unique_ptr<floatMat>(new auto(floatMat(floatElements.data(), rows.data(), cols.data(), m_gpuMatrixReorderedUpper->nonzeroes(), blocksize_, m_gpuMatrix.N())));
    //         }
    //     }
    //     // cast diagonal to float
    //     if (m_float_ILU || m_float_ILU_float_compute){
    //         auto elements = m_gpuMatrixReorderedDiag.value().asStdVector();

    //         size_t idx = 0;
    //         std::vector<float> floatElements(elements.size());
    //         for (auto v : elements) {
    //             floatElements[idx++] = float(v);
    //         }

    //         m_gpuMatrixReorderedDiagfloat.emplace(CuVector<float>(floatElements));
    //     }
    // }
}

template <class M, class X, class Y, int l>
void
OpmCuILU0<M, X, Y, l>::tuneThreadBlockSizes()
{
    // TODO generalize this tuning process in a function separate of the class
    long long bestApplyTime = std::numeric_limits<long long>::max();
    long long bestUpdateTime = std::numeric_limits<long long>::max();
    int bestApplyBlockSize = -1;
    int bestUpdateBlockSize = -1;
    int interval = 64;

    // temporary buffers for the apply
    CuVector<field_type> tmpV(m_gpuMatrix.N() * m_gpuMatrix.blockSize());
    CuVector<field_type> tmpD(m_gpuMatrix.N() * m_gpuMatrix.blockSize());
    tmpD = 1;

    for (int thrBlockSize = interval; thrBlockSize <= 1024; thrBlockSize += interval) {
        // sometimes the first kernel launch kan be slower, so take the time twice
        for (int i = 0; i < 2; ++i) {

            auto beforeUpdate = std::chrono::high_resolution_clock::now();
            m_updateThreadBlockSize = thrBlockSize;
            update();
            std::ignore = cudaDeviceSynchronize();
            auto afterUpdate = std::chrono::high_resolution_clock::now();
            if (cudaSuccess == cudaGetLastError()) { // kernel launch was valid
                long long durationInMicroSec
                    = std::chrono::duration_cast<std::chrono::microseconds>(afterUpdate - beforeUpdate).count();
                if (durationInMicroSec < bestUpdateTime) {
                    bestUpdateTime = durationInMicroSec;
                    bestUpdateBlockSize = thrBlockSize;
                }
            }

            auto beforeApply = std::chrono::high_resolution_clock::now();
            m_applyThreadBlockSize = thrBlockSize;
            apply(tmpV, tmpD);
            std::ignore = cudaDeviceSynchronize();
            auto afterApply = std::chrono::high_resolution_clock::now();
            if (cudaSuccess == cudaGetLastError()) { // kernel launch was valid
                long long durationInMicroSec
                    = std::chrono::duration_cast<std::chrono::microseconds>(afterApply - beforeApply).count();
                if (durationInMicroSec < bestApplyTime) {
                    bestApplyTime = durationInMicroSec;
                    bestApplyBlockSize = thrBlockSize;
                }
            }
        }
    }

    printf("Apply Size: %d, Update Size: %d\n", bestApplyBlockSize, bestUpdateBlockSize);
    m_applyThreadBlockSize = bestApplyBlockSize;
    m_updateThreadBlockSize = bestUpdateBlockSize;
}

} // namespace Opm::cuistl
#define INSTANTIATE_CUDILU_DUNE(realtype, blockdim)                                                                    \
    template class ::Opm::cuistl::OpmCuILU0<Dune::BCRSMatrix<Dune::FieldMatrix<realtype, blockdim, blockdim>>,         \
                                            ::Opm::cuistl::CuVector<realtype>,                                         \
                                            ::Opm::cuistl::CuVector<realtype>>;                                        \
    template class ::Opm::cuistl::OpmCuILU0<Dune::BCRSMatrix<Opm::MatrixBlock<realtype, blockdim, blockdim>>,          \
                                            ::Opm::cuistl::CuVector<realtype>,                                         \
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
