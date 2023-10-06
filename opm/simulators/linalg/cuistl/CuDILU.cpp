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

#include <config.h>
#include <opm/common/ErrorMacros.hpp>
#include <opm/common/TimingMacros.hpp>
#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>

#include <dune/common/fmatrix.hh>
#include <dune/common/version.hh>
#include <dune/common/unused.hh>
#include <dune/istl/bcrsmatrix.hh>

#include <opm/simulators/linalg/cuistl/CuDILU.hpp>


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
    SeqDilu(const M& A)
        : A_(A)
    {
        CheckIfDiagonalPresent<M, 1>::check(A_);
        // we build the inverseD matrix
        Dinv_.resize(A_.N());

        // is this the correct usage?
        update();
    }

    virtual void update() override {

        using block = typename M::block_type;
        
        for ( auto i = A_.begin(); i != A_.end(); ++i)
        {
            block Dinv_temp;
            for (auto a_ij=i->begin(); a_ij != i->end(); ++a_ij) {
                auto a_ji = A_[a_ij.index()].find(i.index());
                
                if (a_ij.index() == i.index()) {
                    Dinv_temp += A_[i.index()][i.index()];
                }

                // if A[i, j] != 0 and A[j, i] != 0:
                else if (a_ji != A_[a_ij.index()].end() && (a_ij.index() < i.index())) {

                    auto d_i = Dinv_[a_ij.index()];
                    d_i.rightmultiply(*a_ji);
                    d_i.leftmultiply(*a_ij);

                    // d[j] -= A[j, i] * d[i] * A[i, j]
                    Dinv_temp -= d_i;
                }
            }
            Dinv_temp.invert();
            Dinv_[i.index()] = Dinv_temp;
        }
    }

    /*!
       \brief Prepare the preconditioner.
       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre(X& v, Y& d) override
    {
        DUNE_UNUSED_PARAMETER(v);
        DUNE_UNUSED_PARAMETER(d);
    }


  /*!
       \brief Apply the preconditioner.
       \copydoc Preconditioner::apply(X&,const Y&)
     */
 virtual void apply(X& v, const Y& d) override
    {
        
    // M = (D + L_A) D^-1 (D + U_A)   (a LU decomposition of M)
    // where L_A and U_A are the strictly lower and upper parts of A and M has the properties:
    // diag(A) = diag(M)
    // solving the product M^-1(b-Ax) using upper and lower triangular solve
    // z = x_k+1 - x_k = M^-1(b - Ax) = (D + L_A)^-1 D (D + U_A)^-1 (b - Ax)

    typedef typename Y::block_type dblock;
    
    // copy current v
    X x(v);
    X z;
    Y y;
    z.resize(A_.N());
    y.resize(A_.N());
    
    // lower triangular solve: (D + L) y = b - Ax
    auto endi=A_.end();
    for (auto i=A_.begin(); i != endi; ++i)
      {
        dblock rhsValue(d[i.index()]);
        auto&& rhs = Impl::asVector(rhsValue);
        for (auto j=(*i).begin(); j != i->end(); ++j) {
            // if  A[i][j] != 0
            // rhs -= A[i][j]* x[j];
            Impl::asMatrix(*j).mmv(Impl::asVector(x[j.index()]), rhs);
                
            // rhs -= A[i][j]* y[j];
            if (j.index() < i.index()) {
                Impl::asMatrix(*j).mmv(Impl::asVector(y[j.index()]), rhs);
            }
        }
        // y = Dinv * rhs
        Impl::asMatrix(Dinv_[i.index()]).mv(rhs, y[i.index()]);
    }


    // upper triangular solve: (D + U) z = Dy 
    auto rendi=A_.beforeBegin();
    for (auto i=A_.beforeEnd(); i!=rendi; --i)
    {
        // rhs = 0
        dblock rhs;

        for (auto j=(*i).beforeEnd(); j.index()>i.index(); --j) {
            // if A [i][j] != 0
            //rhs += A[i][j]*z[j]
            Impl::asMatrix(*j).umv(Impl::asVector(z[j.index()]), rhs);

        }
        // calculate update z = M^-1(b - Ax)
        // z_i = y_i - Dinv*temp
        dblock temp;
        Impl::asMatrix(Dinv_[i.index()]).mv(rhs, temp); 
        z[i.index()] = y[i.index()] - temp;
    }
    // update v
    v += z;
    }

    /*!
       \brief Clean up.
       \copydoc Preconditioner::post(X&)
     */
    virtual void post(X& x) override
    {
        DUNE_UNUSED_PARAMETER(x);
    }
    
    std::vector<typename M::block_type> getDiagonal() {
        return Dinv_;
    }

    //! Category of the preconditioner (see SolverCategory::Category)
    virtual SolverCategory::Category category() const override
    {
        return SolverCategory::sequential;
    }

private:
    //! \brief The matrix we operate on.
    const M& A_;
    //! \brief The inverse of the diagnal matrix
    typedef typename M::block_type matrix_block_type;
    std::vector<matrix_block_type> Dinv_;
};

} // namespace Dune
