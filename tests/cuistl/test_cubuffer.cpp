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

#define BOOST_TEST_MODULE TestCuBuffer

#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <dune/common/fvector.hh>
#include <dune/istl/bvector.hh>
#include <opm/simulators/linalg/cuistl/CuBuffer.hpp>
#include <opm/simulators/linalg/cuistl/detail/cuda_safe_call.hpp>
#include <random>
#include <array>

BOOST_AUTO_TEST_CASE(TestDocumentedUsage)
{
    auto someDataOnCPU = std::vector<double>({1.0, 2.0, 42.0, 59.9451743, 10.7132692});

    auto dataOnGPU = ::Opm::cuistl::CuBuffer<double>(someDataOnCPU);

    // Get data back on CPU in another vector:
    auto stdVectorOnCPU = dataOnGPU.asStdVector();

    BOOST_CHECK_EQUAL_COLLECTIONS(
        stdVectorOnCPU.begin(), stdVectorOnCPU.end(), someDataOnCPU.begin(), someDataOnCPU.end());
}

BOOST_AUTO_TEST_CASE(TestFrontAndBack)
{
    size_t bytes = sizeof(double) * 2;

    auto someDataOnCPU = std::vector<double>({1.0, 2.0, 42.0, 59.9451743, 10.7132692});
    auto dataOnGPU = ::Opm::cuistl::CuBuffer<double>(someDataOnCPU);

    std::array<double, 2> cpuResults;

    double* gpuResults; // store the result of front and back here
    OPM_CUDA_SAFE_CALL(cudaMalloc(&gpuResults, bytes));

    gpuResults[0] = dataOnGPU.front();
    gpuResults[1] = dataOnGPU.back();

    OPM_CUDA_SAFE_CALL(cudaMemcpy(cpuResults.data(), gpuResults, bytes, cudaMemcpyDeviceToHost));

    BOOST_CHECK(cpuResults[0] == someDataOnCPU.front());
    BOOST_CHECK(cpuResults[1] == someDataOnCPU.back());
}
