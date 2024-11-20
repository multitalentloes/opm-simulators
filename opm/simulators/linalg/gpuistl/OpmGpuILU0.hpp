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
#ifndef OPM_GPUILU0_OPM_Impl_HPP
#define OPM_GPUILU0_OPM_Impl_HPP

#include <memory>
#include <opm/grid/utility/SparseTable.hpp>
#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>
#include <opm/simulators/linalg/gpuistl/GpuSparseMatrix.hpp>
#include <opm/simulators/linalg/gpuistl/GpuVector.hpp>
#include <optional>
#include <type_traits>
#include <vector>


namespace Opm::gpuistl
{
//! \brief ILU0 preconditioner on the GPU.
//!
//! \tparam M The matrix type to operate on
//! \tparam X Type of the update
//! \tparam Y Type of the defect
//! \tparam l Ignored. Just there to have the same number of template arguments
//!    as other preconditioners.
//!
//! \note We assume X and Y are both GpuVector<real_type>, but we leave them as template
//! arguments in case of future additions.
template <class M, class X, class Y, int l = 1>
class OpmGpuILU0 : public Dune::PreconditionerWithUpdate<X, Y>
{
public:
    //! \brief The matrix type the preconditioner is for.
    using matrix_type = typename std::remove_const<M>::type;
    //! \brief The domain type of the preconditioner.
    using domain_type = X;
    //! \brief The range type of the preconditioner.
    using range_type = Y;
    //! \brief The field type of the preconditioner.
    using field_type = typename X::field_type;
    //! \brief The GPU matrix type
    using GpuMat = GpuSparseMatrix<field_type>;
    //! \brief The Float matrix type for mixed precision
    using FloatMat = GpuSparseMatrix<float>;

    //! \brief Constructor.
    //!
    //!  Constructor gets all parameters to operate the prec.
    //! \param A The matrix to operate on.
    //! \param w The relaxation factor.
    //!
    explicit OpmGpuILU0(const M& A, bool splitMatrix, bool tuneKernels, bool storeFactorizationAsFloat);

    //! \brief Prepare the preconditioner.
    //! \note Does nothing at the time being.
    void pre(X& x, Y& b) override;

    //! \brief Apply the preconditoner.
    void apply(X& v, const Y& d) override;

    //! \brief Post processing
    //! \note Does nothing at the moment
    void post(X& x) override;

    //! Category of the preconditioner (see SolverCategory::Category)
    Dune::SolverCategory::Category category() const override;

    //! \brief Updates the matrix data.
    void update() final;

    //! \brief Compute LU factorization, and update the data of the reordered matrix
    void LUFactorizeAndMoveData(int moveThreadBlockSize, int factorizationThreadBlockSize);

    //! \brief function that will experimentally tune the thread block sizes of the important cuda kernels
    void tuneThreadBlockSizes();

    //! \returns false
    static constexpr bool shouldCallPre()
    {
        return false;
    }

    //! \returns false
    static constexpr bool shouldCallPost()
    {
        return false;
    }

    virtual bool hasPerfectUpdate() const override {
        return true;
    }


private:
    //! \brief Apply the preconditoner.
    void apply(X& v, const Y& d, int lowerSolveThreadBlockSize, int upperSolveThreadBlockSize);
    //! \brief Updates the matrix data.
    void update(int moveThreadBlockSize, int factorizationThreadBlockSize);
    //! \brief Reference to the underlying matrix
    const M& m_cpuMatrix;
    //! \brief size_t describing the dimensions of the square block elements
    static constexpr const size_t blocksize_ = matrix_type::block_type::cols;
    //! \brief SparseTable storing each row by level
    Opm::SparseTable<size_t> m_levelSets;
    //! \brief converts from index in reordered structure to index natural ordered structure
    std::vector<int> m_reorderedToNatural;
    //! \brief converts from index in natural ordered structure to index reordered strucutre
    std::vector<int> m_naturalToReordered;
    //! \brief The A matrix stored on the gpu, and its reordred version
    GpuMat m_gpuMatrix;
    std::unique_ptr<GpuMat> m_gpuReorderedLU;
    //! \brief If matrix splitting is enabled, then we store the lower and upper part separately
    std::unique_ptr<GpuMat> m_gpuMatrixReorderedLower;
    std::unique_ptr<GpuMat> m_gpuMatrixReorderedUpper;
    //! \brief If mixed precision is enabled, store a float matrix
    std::unique_ptr<FloatMat> m_gpuMatrixReorderedLowerFloat;
    std::unique_ptr<FloatMat> m_gpuMatrixReorderedUpperFloat;
    std::optional<GpuVector<float>> m_gpuMatrixReorderedDiagFloat;
    //! \brief If matrix splitting is enabled, we also store the diagonal separately
    std::optional<GpuVector<field_type>> m_gpuMatrixReorderedDiag;
    //! row conversion from natural to reordered matrix indices stored on the GPU
    GpuVector<int> m_gpuNaturalToReorder;
    //! row conversion from reordered to natural matrix indices stored on the GPU
    GpuVector<int> m_gpuReorderToNatural;
    //! \brief Stores the inverted diagonal that we use in ILU0
    GpuVector<field_type> m_gpuDInv;
    //! \brief Bool storing whether or not we should store matrices in a split format
    bool m_splitMatrix;
    //! \brief Bool storing whether or not we will tune the threadblock sizes. Only used for AMD cards
    bool m_tuneThreadBlockSizes;
    //! \brief Bool storing whether or not we should store the ILU factorization in a float datastructure.
    //! This uses a mixed precision preconditioner to trade numerical accuracy for memory transfer speed.
    bool m_storeFactorizationAsFloat;
    //! \brief variables storing the threadblocksizes to use if using the tuned sizes and AMD cards
    //! The default value of -1 indicates that we have not calibrated and selected a value yet
    int m_upperSolveThreadBlockSize = -1;
    int m_lowerSolveThreadBlockSize = -1;
    int m_moveThreadBlockSize = -1;
    int m_ILU0FactorizationThreadBlockSize = -1;

    GpuVector<int> m_gpuLevelSetSizes;
    int* m_gpuLargestLevelSetSize;
    int* m_gpuNLevels;
    int m_cpuLargestLevelSetSize;
};
} // end namespace Opm::gpuistl

#endif
