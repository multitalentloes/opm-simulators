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
#ifndef OPM_CUSPAI_HPP
#define OPM_CUSPAI_HPP

#include <dune/istl/preconditioner.hh>
#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>
#include <opm/simulators/linalg/cuistl/CuSparseMatrix.hpp>
#include <opm/simulators/linalg/cuistl/detail/CuMatrixDescription.hpp>
#include <opm/simulators/linalg/cuistl/detail/CuSparseHandle.hpp>
#include <opm/simulators/linalg/cuistl/detail/CuSparseResource.hpp>

#include <set>
#include <algorithm>
#include <dune/common/dynmatrix.hh>

namespace Opm::cuistl
{

template<class BlockMatrixClass, class BlockVectorClass> BlockVectorClass solveWithScalarAndReturnBlocked(const BlockMatrixClass&, const BlockVectorClass&);
//! \brief Jacobi preconditioner on the GPU.
//!
//! \note This is a fast but weak preconditioner
//!
//! \tparam M The matrix type to operate on
//! \tparam X Type of the update
//! \tparam Y Type of the defect
//! \tparam l Ignored. Just there to have the same number of template arguments
//!    as other preconditioners.
//!
//! \note We assume X and Y are both CuVector<real_type>, but we leave them as template
//! arguments in case of future additions.
template <class M, class X, class Y, int l = 1>
class CuSPAI : public Dune::PreconditionerWithUpdate<X, Y>
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

    using FieldMat = Dune::FieldMatrix<field_type, matrix_type::block_type::cols, matrix_type::block_type::cols>;
    using DuneMat = Dune::BCRSMatrix<FieldMat>;
    using DuneVec = Dune::BlockVector<FieldMat>;
    using DuneDynMat = Dune::DynamicMatrix<FieldMat>;

    //! \brief Constructor.
    //!
    //!  Constructor gets all parameters to operate the prec.
    //! \param A The matrix to operate on.
    //! \param spai_level decides which power of A's sparsity pattern to ues
    //!
    CuSPAI(const M& A, const int spai_level);

    //! \brief Prepare the preconditioner.
    //! \note Does nothing at the time being.
    virtual void pre(X& x, Y& b) override;

    //! \brief Apply the preconditoner.
    virtual void apply(X& v, const Y& d) override;

    //! \brief Post processing
    //! \note Does nothing at the moment
    virtual void post(X& x) override;

    //! Category of the preconditioner (see SolverCategory::Category)
    virtual Dune::SolverCategory::Category category() const override;

    //! \brief Updates the matrix data.
    virtual void update() override;

    virtual void buildIJSets(int);
    virtual void gatherSubmatIndices();

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

    //! ONLY FOR DEBUGGING
    std::vector<double> getSpaiNnzValues(){
        return spaiNnzValues;
    }

private:
    static constexpr const size_t blocksize_ = matrix_type::block_type::cols;
    //! \brief Reference to the underlying matrix
    const M& m_cpuMatrix;
    //! \brief The relaxation factor to use.
    const int m_SPAI_level;
    //! \brief The A matrix stored on the gpu
    std::unique_ptr<CuSparseMatrix<field_type>> m_gpuMatrix;

    std::set<int> iset, jset;
    int fill_in;

    int N, Nb, nnz, nnzb;

    DuneVec rhs, sol;
    std::vector<DuneMat> submat;
    std::vector<int> submatPointers;
    std::vector<int> submatIndices;
    std::vector<std::vector<int> > submatValsPositions;
    std::vector<int> eyeBlockIndices;
    std::vector<int> rowPointers, colIndices;
    std::vector<int> spaiColPointers, spaiRowIndices;
    std::vector<double> nnzValues, spaiNnzValues;

    DuneDynMat tmp_mat;

};
} // end namespace Opm::cuistl

#endif
