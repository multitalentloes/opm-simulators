/*
  Copyright 2025 Equinor ASA

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
#include "opm/simulators/linalg/PropertyTree.hpp"
#include <boost/test/tools/old/interface.hpp>
#include <config.h>
#include <cstddef>
#include <stdexcept>

#define BOOST_TEST_MODULE TestPreconditionerFactoryGPU

#include <boost/test/unit_test.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include <opm/simulators/linalg/PreconditionerFactory.hpp>
#include <opm/simulators/linalg/PreconditionerFactory_impl.hpp>
#include <opm/simulators/linalg/gpuistl/PreconditionerAdapter.hpp>
#include <opm/simulators/linalg/gpuistl/PreconditionerCPUMatrixToGPUMatrix.hpp>

using ScalarT = double;
constexpr static int Dim = 1;

using CommSeq = Dune::Amg::SequentialInformation;

using MatrixTypeCPU = Dune::BCRSMatrix<Dune::FieldMatrix<ScalarT, Dim, Dim>>;
using VectorCPU = Dune::BlockVector<Dune::FieldVector<ScalarT, 1>>;
using CpuOperatorType = Dune::MatrixAdapter<MatrixTypeCPU, VectorCPU, VectorCPU>;
using FactoryTypeCpu = Opm::PreconditionerFactory<CpuOperatorType, CommSeq>;

using GpuMatrixType = Opm::gpuistl::GpuSparseMatrixWrapper<ScalarT>;
using GpuVectorType = Opm::gpuistl::GpuVector<ScalarT>;
using GpuOperatorType = Dune::MatrixAdapter<GpuMatrixType, GpuVectorType, GpuVectorType>;
using FactoryTypeGpu = Opm::PreconditionerFactory<GpuOperatorType, CommSeq>;

BOOST_AUTO_TEST_CASE(TestMatrixAdapter)
{
    const std::size_t N = 10;
    const std::size_t nonZeroes = N * 3 - 2;

    // The name of the preconditioner on the GPU directly (key) and
    // the name of the wrapped preconditioner (value).
    auto preconditionerPairs = std::map<std::string, std::string> {
        {"ilu0", "gpuilu0"},
        {"jac", "gpujac"},
        {"opmilu0", "opmgpuilu0"},
        {"dilu", "gpudilu"},
    };

    for (const auto& [gpuPreconditionerName, wrappedPreconditionerName] : preconditionerPairs) {
        MatrixTypeCPU matrix(N, N, nonZeroes, MatrixTypeCPU::row_wise);
        for (auto row = matrix.createbegin(); row != matrix.createend(); ++row) {
            // Add nonzeros for left neighbour, diagonal and right neighbour
            if (row.index() > 0) {
                row.insert(row.index() - 1);
            }
            row.insert(row.index());
            if (row.index() < matrix.N() - 1) {
                row.insert(row.index() + 1);
            }
        }

        for (std::size_t i = 0; i < N; ++i) {
            matrix[i][i] = -2;
            if (i < int(N) - 1) {
                matrix[i][i + 1] = 1;
            }

            if (i > 0) {
                matrix[i][i - 1] = 1;
            }
        }
        GpuMatrixType gpuMatrix = GpuMatrixType::fromMatrix(matrix);
        GpuOperatorType gpuOp(gpuMatrix);

        CpuOperatorType cpuOp(matrix);



        Opm::PropertyTree prm;
        prm.put(std::string("type"), gpuPreconditionerName);
        Opm::PropertyTree prmCPU;
        prmCPU.put(std::string("type"), wrappedPreconditionerName);


        auto preconditionerGPU = FactoryTypeGpu::create(gpuOp, prm);
        auto preconditionerWrapped = FactoryTypeCpu::create(cpuOp, prmCPU);


        // check for the standard basis {e_i}
        // (e_i=(0,...,0, 1 (i-th place), 0, ..., 0))
        for (std::size_t i = 0; i < N; ++i) {
            VectorCPU inputVector(N);
            inputVector[i][0] = 1.0;
            VectorCPU outputVectorWrapped(N);
            auto outputVectorGPU = Opm::gpuistl::GpuVector<ScalarT>(N);
            auto inputVectorGPU = Opm::gpuistl::GpuVector<ScalarT>(inputVector);
            preconditionerGPU->apply(outputVectorGPU, inputVectorGPU);
            preconditionerWrapped->apply(outputVectorWrapped, inputVector);

            auto outputVectorgGPUonCPU = outputVectorGPU.asDuneBlockVector<Dim>();

            for (std::size_t component = 0; component < N; ++component) {
                BOOST_TEST(outputVectorWrapped[component][0] == outputVectorgGPUonCPU[component][0],
                           "Component " << component << " of the output vector is not equal to the expected value."
                                        << "\nExpected: " << outputVectorWrapped[component][0]
                                        << ",\nbut got: " << outputVectorgGPUonCPU[component][0]
                                        << "\nfor preconditioner: " << gpuPreconditionerName
                                        << " and wrapped preconditioner: " << wrappedPreconditionerName);
            }
        }
    }
}

namespace {
template <bool Pre, bool Post>
struct LifecycleTraits
{
    static constexpr bool shouldCallPre() { return Pre; }
    static constexpr bool shouldCallPost() { return Post; }
};

template <bool Pre, bool Post>
void checkAdapterLifecycleTraits()
{
    using Traits = LifecycleTraits<Pre, Post>;
    using VectorAdapter = Opm::gpuistl::PreconditionerAdapter<VectorCPU, VectorCPU, Traits>;
    using MatrixAdapter = Opm::gpuistl::PreconditionerCPUMatrixToGPUMatrix<
        GpuVectorType, GpuVectorType, Traits, MatrixTypeCPU>;
    BOOST_CHECK_EQUAL(VectorAdapter::shouldCallPre(), Pre);
    BOOST_CHECK_EQUAL(VectorAdapter::shouldCallPost(), Post);
    BOOST_CHECK_EQUAL(MatrixAdapter::shouldCallPre(), Pre);
    BOOST_CHECK_EQUAL(MatrixAdapter::shouldCallPost(), Post);
}
} // namespace

BOOST_AUTO_TEST_CASE(TestAdapterLifecycleTraits)
{
    checkAdapterLifecycleTraits<false, false>();
    checkAdapterLifecycleTraits<true, false>();
    checkAdapterLifecycleTraits<false, true>();
    checkAdapterLifecycleTraits<true, true>();
}

BOOST_AUTO_TEST_CASE(TestLevelScheduledReplayAndUpdate)
{
    // ILU0 and DILU are exact for this scalar tridiagonal matrix. Exercise
    // capture and replay with two vector pairs, changing RHS and matrix values.
    const int n = 17;
    for (const std::string type : {"opmilu0", "dilu"}) {
        for (int mode = 0; mode < (type == "dilu" ? 6 : 5); ++mode) {
            BOOST_TEST_CONTEXT("type=" << type << ", mode=" << mode) {
                const bool split = mode < 4;
                const int precision = mode < 3 ? mode : 0;
                const bool tune = mode == 3;
                MatrixTypeCPU matrix(n, n, 3 * n - 2, MatrixTypeCPU::row_wise);
                for (auto row = matrix.createbegin(); row != matrix.createend(); ++row) {
                    if (row.index() > 0) row.insert(row.index() - 1);
                    row.insert(row.index());
                    if (row.index() + 1 < matrix.N()) row.insert(row.index() + 1);
                }
                for (int i = 0; i < n; ++i) {
                    matrix[i][i] = 6.0 + 0.01 * i;
                    if (i > 0) matrix[i][i - 1] = -0.5;
                    if (i + 1 < n) matrix[i][i + 1] = -0.75;
                }
                auto gpuMatrix = GpuMatrixType::fromMatrix(matrix);
                GpuOperatorType op(gpuMatrix);
                Opm::PropertyTree prm;
                prm.put("type", type);
                prm.put("split_matrix", split);
                prm.put("mixed_precision_scheme", precision);
                prm.put("tune_gpu_kernels", tune);
                prm.put("reorder", mode != 5);
                auto prec = FactoryTypeGpu::create(op, prm);
                GpuVectorType inputA(n), outputA(n), inputB(n), outputB(n);
                VectorCPU expected(n), rhs(n);
                for (int epoch = 0; epoch < 3; ++epoch) {
                    if (epoch) {
                        matrix *= 1.1;
                        gpuMatrix.updateNonzeroValues(matrix, true);
                        prec->update();
                    }
                    for (int replay = 0; replay < 6; ++replay) {
                        for (int i = 0; i < n; ++i) expected[i] = 1.0 + 0.2 * i + 0.3 * replay;
                        matrix.mv(expected, rhs);
                        auto& input = replay % 2 ? inputA : inputB;
                        auto& output = replay % 2 ? outputA : outputB;
                        input.copyFromHost(rhs);
                        output = 0;
                        prec->apply(output, input);
                        // A default-stream consumer must observe the finished solve.
                        output *= 2.0;
                        const auto actual = output.asStdVector();
                        const double tolerance = precision ? 2e-6 : 1e-11;
                        for (int i = 0; i < n; ++i) {
                            BOOST_CHECK_SMALL(actual[i] - 2.0 * expected[i][0], tolerance);
                        }
                    }
                }
            }
        }
    }
}
