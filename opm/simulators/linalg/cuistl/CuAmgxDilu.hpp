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
#ifndef OPM_CUAMGXDILU_HPP
#define OPM_CUAMGXDILU_HPP

#include <dune/istl/preconditioner.hh>
#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>
#include <opm/simulators/linalg/cuistl/CuSparseMatrix.hpp>
#include <opm/simulators/linalg/cuistl/detail/CuMatrixDescription.hpp>
#include <opm/simulators/linalg/cuistl/detail/CuSparseHandle.hpp>
#include <opm/simulators/linalg/cuistl/detail/CuSparseResource.hpp>

#include <amgx_c.h>

namespace Opm::cuistl
{
//! \brief Sequential Jacobi preconditioner on the GPU through the CuSparse library.
//!
//! This implementation calls the CuSparse functions, which in turn essentially
//! does a level decomposition to get some parallelism.
//!
//! \note This is not expected to be a fast preconditioner.
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
class CuDilu : public Dune::PreconditionerWithUpdate<X, Y>
{
public:
    //! \brief The matrix type the preconditioner is for.
    using matrix_type = typename std::remove_const<M>::type ;
    //! \brief The domain type of the preconditioner.
    using domain_type = X;
    //! \brief The range type of the preconditioner.
    using range_type = Y;
    //! \brief The field type of the preconditioner.
    using field_type = typename X::field_type;

    //! \brief Constructor.
    //!
    //!  Constructor gets all parameters to operate the prec.
    //! \param A The matrix to operate on.
    //! \param w The relaxation factor.
    //!
    CuDilu(const M& A, field_type w);
    ~CuDilu();

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


private:
    //! \brief Reference to the underlying matrix
    const M& m_underlyingMatrix;
    //! \brief The relaxation factor to use.
    field_type m_w;

    CuSparseMatrix<field_type> m;
    CuVector<field_type> m_diagInvFlattened;
    CuVector<field_type> d_resultBuffer;
    detail::CuSparseMatrixDescriptionPtr m_description;

    detail::CuSparseHandle& m_cuSparseHandle;
    detail::CuBlasHandle& m_cuBlasHandle;

    size_t findBufferSize();

    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_solver_handle amgx_solver;
    AMGX_vector_handle amgx_x; 
    AMGX_vector_handle amgx_b;
    AMGX_matrix_handle amgx_A;
};
} // end namespace Opm::cuistl

#endif