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
#include <dune/common/fmatrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <fmt/core.h>
#include <cuda_fp16.h>
#include <opm/common/ErrorMacros.hpp>
#include <opm/simulators/linalg/cuistl/CuJac.hpp>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <opm/simulators/linalg/cuistl/detail/preconditionerKernels/JacKernels.hpp>
#include <opm/simulators/linalg/cuistl/detail/vector_operations.hpp>
#include <opm/simulators/linalg/matrixblock.hh>

#include <chrono>

class CumulativeScopeTimer {
public:
    // Constructor starts the timer
    CumulativeScopeTimer() : start_time(std::chrono::high_resolution_clock::now()) {
        ++instance_count;
    }

    // Destructor stops the timer and accumulates the time
    ~CumulativeScopeTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        total_time_spent += duration;

        std::cout << "Average: " << total_time_spent / instance_count << "us/apply. This apply: " << duration << "us. Total time spent: " << total_time_spent / 1000.0 << " ms. " << " Applies: " << instance_count << std::endl;
    }

    // Static method to report the cumulative time and instance count
    static void report() {
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;  // Time when the timer started
    static long long total_time_spent;  // Cumulative time spent in all instances
    static int instance_count;  // Number of times the timer has been instantiated
};

// Static member variables need to be defined outside the class
long long CumulativeScopeTimer::total_time_spent = 0;
int CumulativeScopeTimer::instance_count = 0;

namespace Opm::cuistl
{

template <class M, class X, class Y, int l>
CuJac<M, X, Y, l>::CuJac(const M& A, field_type w)
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

    // Compute the inverted diagonal of A and store it in a vector format in m_diagInvFlattened
    invertDiagonalAndFlatten();
}

template <class M, class X, class Y, int l>
void
CuJac<M, X, Y, l>::pre([[maybe_unused]] X& x, [[maybe_unused]] Y& b)
{
}

template <class M, class X, class Y, int l>
void
CuJac<M, X, Y, l>::apply(X& v, const Y& d)
{
    cudaDeviceSynchronize();
    CumulativeScopeTimer timer1;
    // Jacobi preconditioner: x_{n+1} = x_n + w * (D^-1 * (b - Ax_n) )
    // Working with defect d and update v it we only need to set v = w*(D^-1)*d

    // Compute the MV product where the matrix is diagonal and therefore stored as a vector.
    // The product is thus computed as a hadamard product.
    detail::weightedDiagMV(
        m_diagInvFlattened.data(), m_gpuMatrix.N(), m_gpuMatrix.blockSize(), m_relaxationFactor, d.data(), v.data());
    cudaDeviceSynchronize();
}

template <class M, class X, class Y, int l>
void
CuJac<M, X, Y, l>::post([[maybe_unused]] X& x)
{
}

template <class M, class X, class Y, int l>
Dune::SolverCategory::Category
CuJac<M, X, Y, l>::category() const
{
    return Dune::SolverCategory::sequential;
}

template <class M, class X, class Y, int l>
void
CuJac<M, X, Y, l>::update()
{
    m_gpuMatrix.updateNonzeroValues(m_cpuMatrix);
    invertDiagonalAndFlatten();
}

template <class M, class X, class Y, int l>
void
CuJac<M, X, Y, l>::invertDiagonalAndFlatten()
{
    detail::JAC::invertDiagonalAndFlatten<field_type, matrix_type::block_type::cols>(
        m_gpuMatrix.getNonZeroValues().data(),
        m_gpuMatrix.getRowIndices().data(),
        m_gpuMatrix.getColumnIndices().data(),
        m_gpuMatrix.N(),
        m_diagInvFlattened.data());
}

} // namespace Opm::cuistl
#define INSTANTIATE_CUJAC_DUNE(realtype, blockdim)                                                                     \
    template class ::Opm::cuistl::CuJac<Dune::BCRSMatrix<Dune::FieldMatrix<realtype, blockdim, blockdim>>,             \
                                        ::Opm::cuistl::CuVector<realtype>,                                             \
                                        ::Opm::cuistl::CuVector<realtype>>;                                            \
    template class ::Opm::cuistl::CuJac<Dune::BCRSMatrix<Opm::MatrixBlock<realtype, blockdim, blockdim>>,              \
                                        ::Opm::cuistl::CuVector<realtype>,                                             \
                                        ::Opm::cuistl::CuVector<realtype>>

INSTANTIATE_CUJAC_DUNE(double, 1);
INSTANTIATE_CUJAC_DUNE(double, 2);
INSTANTIATE_CUJAC_DUNE(double, 3);
INSTANTIATE_CUJAC_DUNE(double, 4);
INSTANTIATE_CUJAC_DUNE(double, 5);
INSTANTIATE_CUJAC_DUNE(double, 6);

INSTANTIATE_CUJAC_DUNE(float, 1);
INSTANTIATE_CUJAC_DUNE(float, 2);
INSTANTIATE_CUJAC_DUNE(float, 3);
INSTANTIATE_CUJAC_DUNE(float, 4);
INSTANTIATE_CUJAC_DUNE(float, 5);
INSTANTIATE_CUJAC_DUNE(float, 6);

INSTANTIATE_CUJAC_DUNE(__half, 1);
INSTANTIATE_CUJAC_DUNE(__half, 2);
INSTANTIATE_CUJAC_DUNE(__half, 3);
INSTANTIATE_CUJAC_DUNE(__half, 4);
INSTANTIATE_CUJAC_DUNE(__half, 5);
INSTANTIATE_CUJAC_DUNE(__half, 6);
