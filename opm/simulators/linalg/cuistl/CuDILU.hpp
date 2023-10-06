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

#ifndef OPM_DILU_HEADER_INCLUDED
#define OPM_DILU_HEADER_INCLUDED

#include <config.h>
#include <opm/common/ErrorMacros.hpp>
#include <opm/common/TimingMacros.hpp>
#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>

#include <dune/common/fmatrix.hh>
#include <dune/common/version.hh>
#include <dune/common/unused.hh>
#include <dune/istl/bcrsmatrix.hh>


namespace Dune
{

/*! \brief The sequential DILU preconditioner.

   \tparam M The matrix type to operate on
   \tparam X Type of the update
   \tparam Y Type of the defect
 */
template <class M, class X, class Y>
class SeqDilu : public PreconditionerWithUpdate<X, Y>
{
public:
    //! \brief The matrix type the preconditioner is for.
    typedef M matrix_type;
    //! \brief The domain type of the preconditioner.
    typedef X domain_type;
    //! \brief The range type of the preconditioner.
    typedef Y range_type;
    //! \brief The field type of the preconditioner.
    typedef typename X::field_type field_type;
    //! \brief scalar type underlying the field_type
#if DUNE_VERSION_NEWER(DUNE_ISTL, 2, 7)
    typedef Simd::Scalar<field_type> scalar_field_type;
#else
    typedef SimdScalar<field_type> scalar_field_type;
#endif

    /*! \brief Constructor.
       Constructor gets all parameters to operate the prec.
       \param A The matrix to operate on.
     */
    SeqDilu(const M& A);

    virtual void update() override;

    /*!
       \brief Prepare the preconditioner.
       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre(X& v, Y& d) override;

  /*!
       \brief Apply the preconditioner.
       \copydoc Preconditioner::apply(X&,const Y&)
     */
 virtual void apply(X& v, const Y& d) override;

    /*!
       \brief Clean up.
       \copydoc Preconditioner::post(X&)
     */
    virtual void post(X& x) override;
    
    std::vector<typename M::block_type> getDiagonal();

    //! Category of the preconditioner (see SolverCategory::Category)
    virtual SolverCategory::Category category() const override;

private:
    //! \brief The matrix we operate on.
    const M& A_;
    //! \brief The inverse of the diagnal matrix
    typedef typename M::block_type matrix_block_type;
    std::vector<matrix_block_type> Dinv_;
};

} // namespace Dune

#endif