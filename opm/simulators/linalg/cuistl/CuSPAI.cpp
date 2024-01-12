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
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <fmt/core.h>
#include <opm/common/ErrorMacros.hpp>
#include <opm/simulators/linalg/cuistl/CuSPAI.hpp>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <opm/simulators/linalg/cuistl/CuSparseMatrix.hpp>
#include <opm/simulators/linalg/cuistl/detail/safe_conversion.hpp>
#include <opm/simulators/linalg/matrixblock.hh>

#include <opm/simulators/linalg/bda/BlockedMatrix.hpp>
#include <opm/simulators/linalg/bda/BdaBridge.hpp>
#include <vector>
#include <dune/common/dynvector.hh>
#include <dune/common/dynmatrix.hh>

namespace Opm::cuistl
{
// Given Dune BCSR matrix, create a bda blocked matrix.
template<class DuneMatrixClass>
Opm::Accelerator::BlockedMatrix convertToBlockedMatrix(DuneMatrixClass& matrix){
    const int Nb = matrix.N(); // number of rows
    const int nnzb = matrix.nonzeroes();
    std::vector<int> h_rows;
    std::vector<int> h_cols;
    // convert colIndices and rowPointers
    h_rows.emplace_back(0);
    for (typename DuneMatrixClass::const_iterator r = matrix.begin(); r != matrix.end(); ++r) {
        for (auto c = r->begin(); c != r->end(); ++c) {
            h_cols.emplace_back(c.index());
        }
        h_rows.emplace_back(h_cols.size());
    }

    // Opm::checkMemoryContiguous(matrix);
    return Opm::Accelerator::BlockedMatrix(Nb, nnzb, (matrix[0][0]).size(), const_cast<typename DuneMatrixClass::field_type*>(&(((matrix)[0][0][0][0]))), h_cols.data(), h_rows.data());
}

template<class BlockMatrixClass, class BlockVectorClass>
BlockVectorClass solveWithScalarAndReturnBlocked(const BlockMatrixClass& blockMatrix, const BlockVectorClass& blockRhs) {
    using real_type = typename BlockMatrixClass::value_type::field_type;

    //! set the correct M to be matrix.M(), and change the name of the current M variable to something else
    const size_t N = blockMatrix.N(); // number of rows
    const size_t M = blockMatrix.M(); // number of columns
    static constexpr int blocksize = BlockMatrixClass::value_type::rows;
    const size_t N_scalar = N * blocksize;
    const size_t M_scalar = M * blocksize;

    Dune::DynamicMatrix<real_type> scalarMatrix(N_scalar, M_scalar, 0.0);

    for (size_t row = 0; row < N; ++row) {
        for (size_t column = 0; column < M; ++column) {
            for (int i = 0; i < blocksize; ++i) {
                for (int j = 0; j < blocksize; ++j) {
                    scalarMatrix[row * blocksize + i][column * blocksize + j] = blockMatrix[row][column][i][j];
                }
            }
        }
    }

    BlockVectorClass blockSolution(M); //TODO CHECK DIMS PLEASE

    // create the solution column by column to utilize the solve method of scalar matrices
    for (int col = 0; col < blocksize; ++col){
        Dune::DynamicVector<real_type> scalarRhsBeforeMTV(N_scalar, 0.0); //TODO CHECK DIMS PLEASE
        Dune::DynamicVector<real_type> scalarRhsAfterMTV(M_scalar, 0.0); //TODO CHECK DIMS PLEASE
        for (size_t row = 0; row < N; ++row) {
            for (int i = 0; i < blocksize; ++i) {
                scalarRhsBeforeMTV[row * blocksize + i] = blockRhs[row][i][col];
            }
        }


        Dune::DynamicVector<real_type> scalarSolution(M_scalar, 0); //TODO CHECK DIMS PLEASE

        //! go from Ax=b to A'Ax=A'b to find least squares solution
        //TODO verify that the vector read from and written to can be the same!
        scalarMatrix.mtv(scalarRhsBeforeMTV, scalarRhsAfterMTV); // scalarRhs = scalarMatrix'*scalarRhs

        //TODO transpose the matrix properly
        Dune::DynamicMatrix<real_type> transposedScalarMatrix(M_scalar, N_scalar, 0.0);
        for (auto loc_row = scalarMatrix.begin(); loc_row != scalarMatrix.end(); ++loc_row){
            for (auto loc_col = loc_row->begin(); loc_col != loc_row->end(); ++loc_col){
                transposedScalarMatrix[loc_col.index()][loc_row.index()] = *loc_col;
            }
        }

        Dune::DynamicMatrix<real_type> aTa(transposedScalarMatrix.N(), scalarMatrix.M(), 0.0);
        for (size_t i = 0; i < transposedScalarMatrix.N(); ++i){
            for (size_t j = 0; j < scalarMatrix.M(); ++j){
                aTa[i][j] = 0;
                for (size_t k = 0; k < transposedScalarMatrix.M(); ++k){
                    aTa[i][j] += transposedScalarMatrix[i][k] * scalarMatrix[k][j];
                }
            }
        }
        aTa.solve(scalarSolution, scalarRhsAfterMTV);

        for (size_t row = 0; row < M; ++row) {
            for (int i = 0; i < blocksize; ++i) {
                blockSolution[row][i][col] = scalarSolution[row * blocksize + i];
            }
        }
    }

    return blockSolution;
}

// added for the test to compile
template Dune::BlockVector<Dune::FieldMatrix<double, 2, 2>> solveWithScalarAndReturnBlocked(const Dune::DynamicMatrix<Dune::FieldMatrix<double, 2, 2>>&, const Dune::BlockVector<Dune::FieldMatrix<double, 2, 2>>&);

template <class M, class X, class Y, int l>
CuSPAI<M, X, Y, l>::CuSPAI(const M& A, int spai_level)
    : m_cpuMatrix(A)
    , m_SPAI_level(spai_level)
    , fill_in(spai_level-1)
    , m_gpuMatrix(nullptr)
{
    assert(m_SPAI_level>0); // SPAI0 is currently not supported
    const unsigned int bs = blocksize_;
    this->Nb = m_cpuMatrix.N();
    this->N = Nb * bs;
    this->nnzb = m_cpuMatrix.nonzeroes();
    this->nnz = nnzb * bs * bs;

    submat.resize(Nb);
    eyeBlockIndices.resize(Nb);
    submatValsPositions.resize(Nb);
    rowPointers.reserve(Nb + 1);
    colIndices.reserve(nnzb);
    spaiColPointers.resize(Nb + 1);

    // std::copy(m_cpuMatrix.getRowIndices().begin(), m_cpuMatrix.getRowIndices().begin() + Nb + 1, rowPointers.begin());
    // std::copy(m_cpuMatrix.getColumnIndices().begin(), m_cpuMatrix.getColumnIndices().begin() + nnzb, colIndices.begin());
    // colIndices.reserve(nnzb);
    rowPointers.push_back(0); //! I added this to have n+1 items in list
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

    update();
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
    m_gpuMatrix->mv(d, v);
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
    const unsigned int bs = blocksize_;
    int count;

    //! new matrix for easing indexing during porting
    Opm::Accelerator::BlockedMatrix bda_matrix = convertToBlockedMatrix(m_cpuMatrix);

    for(int tcol = 0; tcol < Nb; tcol++){
        count = 0;
        for(auto row = submat[tcol].begin(); row != submat[tcol].end(); ++row){
            for(auto col = (*row).begin(); col != (*row).end(); ++col){
                for(auto br = (*col).begin(); br != (*col).end(); ++br){
                    for(auto bc = (*br).begin(); bc != (*br).end(); ++bc){
                        (*bc) = bda_matrix.nnzValues[submatValsPositions[tcol][count]];
                        ++count;
                    }
                }
            }
        }

        sol.resize(submat[tcol].M());
        rhs.resize(submat[tcol].N());
        rhs = 0;
        rhs[eyeBlockIndices[tcol]] = Dune::ScaledIdentityMatrix<double, bs>(1);

        //! create a dense matrix M_d from the submat[tcol], and call M_d.solve(sol, rhs)
        tmp_mat = DuneDynMat(submat[tcol].N(), submat[tcol].M(), 0.0);
        for(auto row = submat[tcol].begin(); row != submat[tcol].end(); ++row){
            for(auto col = (*row).begin(); col != (*row).end(); ++col){
                for(auto br = (*col).begin(); br != (*col).end(); ++br){
                    for(auto bc = (*br).begin(); bc != (*br).end(); ++bc){
                        tmp_mat[row.index()][col.index()][br.index()][bc.index()] = *bc;
                    }
                }
            }
        }

        sol = solveWithScalarAndReturnBlocked(tmp_mat, rhs);

        for(unsigned int i = 0; i < submat[tcol].M(); i++){
            for(unsigned int j = 0; j < bs; j++){
                for(unsigned int k = 0; k < bs; k++){
                    spaiNnzValues[(spaiColPointers[tcol] + i) * bs * bs + j * bs + k] = sol[i][j][k];
                }
            }
        }
    }
   //TODO: the Bueno code has flipped the naming convention of rows/columns in the csr format spaiColPointers point to the start of the rows, and spaiRowIndices give the column index of the items...
   //TODO: The flipping of the names might be due to some implicit transposition?
    m_gpuMatrix.reset(new CuSparseMatrix<field_type>(spaiNnzValues.data(), spaiColPointers.data(), spaiRowIndices.data(), spaiNnzValues.size()/(bs*bs), bs, Nb));
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

//! commented out to make sure the conversion from bcsr work with the bda bridge
//! the bda bridge converter only works for doubles
// INSTANTIATE_CUSPAI_DUNE(float, 1);
// INSTANTIATE_CUSPAI_DUNE(float, 2);
// INSTANTIATE_CUSPAI_DUNE(float, 3);
// INSTANTIATE_CUSPAI_DUNE(float, 4);
// INSTANTIATE_CUSPAI_DUNE(float, 5);
// INSTANTIATE_CUSPAI_DUNE(float, 6);
