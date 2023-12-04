/*
  Copyright SINTEF AS 2022

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



#include <cuda_runtime.h>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/matrixmarket.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solvers.hh>
#include <opm/simulators/linalg/ParallelOverlappingILU0.hpp>
#include <opm/simulators/linalg/cuistl/CuSeqILU0.hpp>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <opm/simulators/linalg/cuistl/PreconditionerAdapter.hpp>
#include <opm/simulators/linalg/cuistl/detail/cuda_safe_call.hpp>

#include <limits>
#include <memory>
#include <random>

template <int dim, class T = double>
void
readAndSolve(const auto xFilename, const auto matrixFilename, const size_t iterationMax, const auto rhsFilename)
{
    using M = Dune::FieldMatrix<T, dim, dim>;
    using SpMatrix = Dune::BCRSMatrix<M>;
    using Vector = Dune::BlockVector<Dune::FieldVector<T, dim>>;
    using CuILU0 = Opm::cuistl::CuSeqILU0<SpMatrix, Opm::cuistl::CuVector<T>, Opm::cuistl::CuVector<T>>;
    using GPUVector = Opm::cuistl::CuVector<T>;
    SpMatrix B;
    Vector x, rhs;

    Dune::loadMatrixMarket(B, matrixFilename);
    Dune::loadMatrixMarket(x, xFilename);
    Dune::loadMatrixMarket(rhs, rhsFilename);

    const size_t N = B.N();
    auto BonGPU = Opm::cuistl::CuSparseMatrix<T>::fromMatrix(B);//std::make_shared<Opm::cuistl::CuSparseMatrix<T>>(Opm::cuistl::CuSparseMatrix<T>::fromMatrix(B));
    auto BOperator = std::make_shared<
        Dune::MatrixAdapter<Opm::cuistl::CuSparseMatrix<T>, Opm::cuistl::CuVector<T>, Opm::cuistl::CuVector<T>>>(
        BonGPU);
    auto cuILU = std::make_shared<CuILU0>(B, 1.0);
    auto scalarProduct = std::make_shared<Dune::SeqScalarProduct<Opm::cuistl::CuVector<T>>>();
    const auto tolerance = 1e-2;
    auto solver
        = Dune::BiCGSTABSolver<Opm::cuistl::CuVector<T>>(BOperator, scalarProduct, cuILU, tolerance, iterationMax, 0);


    auto BCPUOperator = std::make_shared<Dune::MatrixAdapter<SpMatrix, Vector, Vector>>(B);
    auto scalarProductCPU = std::make_shared<Dune::SeqScalarProduct<Vector>>();
    auto ILUCPU
        = std::make_shared<Opm::ParallelOverlappingILU0<SpMatrix, Vector, Vector, Dune::Amg::SequentialInformation>>(
            B, 0, 1.0, Opm::MILU_VARIANT::ILU);
    auto solverCPU = Dune::BiCGSTABSolver<Vector>(BCPUOperator, scalarProductCPU, ILUCPU, tolerance, iterationMax, 0);

    Dune::InverseOperatorResult result, resultCPU;

    auto xOnGPU = GPUVector(x.dim());
    xOnGPU.copyFromHost(x);

    bool gpufailed = false;
    bool cpufailed = false;
    auto rhsOnGPU = GPUVector(rhs.dim());
    rhsOnGPU.copyFromHost(rhs);
    auto gpuStart = std::chrono::high_resolution_clock::now();
    try {
        solver.apply(xOnGPU, rhsOnGPU, result);
    } catch (const std::logic_error& e) {
        gpufailed = true;
    }

    OPM_CUDA_SAFE_CALL(cudaDeviceSynchronize());
    auto gpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuStart = std::chrono::high_resolution_clock::now();
    solverCPU.apply(x, rhs, resultCPU);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    std::cout << "{\n";
    if (result.iterations == iterationMax || gpufailed) {
        std::cout << "    \"GPU\": -1," << std::endl;
    } else {
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(gpuEnd - gpuStart);
        std::cout << "    \"GPU\": " << double(duration.count()) / 1e6 << "," << std::endl;
    }
    if (resultCPU.iterations == iterationMax) {
        std::cout << "    \"CPU\": -1" << std::endl;
    } else {
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart);
        std::cout << "    \"CPU\": " << double(duration.count()) / 1e6 << std::endl;
    }

    std::cout << "}\n";
}

int
main(int argc, char** argv)
{


    [[maybe_unused]] const auto& helper = Dune::MPIHelper::instance(argc, argv);

    const auto matrixFilename = std::string(argv[1]);
    const auto xFilename = std::string(argv[2]);
    const auto rhsFilename = std::string(argv[3]);
    const size_t iterationMax = 200;

    size_t dim = 2;

    {
        std::ifstream matrixfile(matrixFilename);
        std::string line;
        const std::string lineToFind = "% ISTL_STRUCT blocked";
        while (std::getline(matrixfile, line)) {
            if (line.substr(0, lineToFind.size()) == lineToFind) {
                dim = std::atoi(line.substr(lineToFind.size() + 2).c_str()); // Yeah, max 9
            }
        }
    }

    switch (dim) {
    case 1:
        readAndSolve<1>(xFilename, matrixFilename, iterationMax, rhsFilename);
        break;
    case 2:
        readAndSolve<2>(xFilename, matrixFilename, iterationMax, rhsFilename);
        break;
    case 3:
        readAndSolve<3>(xFilename, matrixFilename, iterationMax, rhsFilename);
        break;
    case 4:
        readAndSolve<4>(xFilename, matrixFilename, iterationMax, rhsFilename);
        break;
    default:
        throw std::runtime_error("Unresolved matrix dimension " + std::to_string(dim));
    }
}