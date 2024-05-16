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
#include <algorithm>

//TODO: clang format this file when sketch is done

BOOST_AUTO_TEST_CASE(TestDocumentedUsage)
{
    // A simple test to check that we can move data to and from the GPU
    auto someDataOnCPU = std::vector<double>({1.0, 2.0, 42.0, 59.9451743, 10.7132692});

    auto dataOnGPU = ::Opm::cuistl::CuBuffer<double>(someDataOnCPU);
    auto checkThatDefaulConstructorCompiles = ::Opm::cuistl::CuBuffer<double>();

    auto stdVectorOnCPU = dataOnGPU.asStdVector();

    BOOST_CHECK_EQUAL_COLLECTIONS(
        stdVectorOnCPU.begin(), stdVectorOnCPU.end(), someDataOnCPU.begin(), someDataOnCPU.end());
}

BOOST_AUTO_TEST_CASE(TestFrontAndBack)
{
    // This test checks that the results of front() and back() matches that of a regular vector
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

BOOST_AUTO_TEST_CASE(TestConstAndSwap)
{
    std::vector<double> avec = {1, 2, 3, 4, 5, 6, 7};
    using CuBuffer = ::Opm::cuistl::CuBuffer<double>;
    const CuBuffer a(avec);

    CuBuffer b;
    b.resize(a.size());

    std::move(a.begin(), a.end(), b.begin());

    auto bvec = b.asStdVector();

    BOOST_CHECK_EQUAL_COLLECTIONS(avec.begin(), avec.end(), bvec.begin(), bvec.end());
}

BOOST_AUTO_TEST_CASE(TestSquareBracketOperator)
{
    // test for mutable cubuffers
    std::vector<double> cpuv = {1.0, 0.0};
    std::vector<double> expected = {1.0, 1.0};

    using CuBuffer = ::Opm::cuistl::CuBuffer<double>;

    CuBuffer a(cpuv);
    a[1] = a[0]; // checks both read and write for mutable cubuffers

    auto gpuv = a.asStdVector();

    BOOST_CHECK(gpuv[0] == gpuv[1]);
    BOOST_CHECK_EQUAL_COLLECTIONS(expected.begin(), expected.end(), gpuv.begin(), gpuv.end());

    // test functionality of const cubuffers
    const CuBuffer b(cpuv);
    double* gpuDouble;
    OPM_CUDA_SAFE_CALL(cudaMalloc(&gpuDouble, sizeof(double)));
    gpuDouble[0] = b[0];

    double cpuDouble[1];

    OPM_CUDA_SAFE_CALL(cudaMemcpy(cpuDouble, gpuDouble, sizeof(double), cudaMemcpyDeviceToHost));
    auto gpuv2 = b.asStdVector();

    BOOST_CHECK(cpuDouble[0] == cpuv[0]);
    BOOST_CHECK_EQUAL_COLLECTIONS(cpuv.begin(), cpuv.end(), gpuv2.begin(), gpuv2.end());
}

BOOST_AUTO_TEST_CASE(TestSTLSort)
{
    auto someDataOnCPU = std::vector<double>({1.0, 2.0, 42.0, 59.9451743, 10.7132692, -100, 20});
    auto expectedWhenSorted = std::vector<double>({{-100, 1.0, 2.0, 10.7132692, 20, 42.0, 59.9451743}});
    auto dataOnGPU = ::Opm::cuistl::CuBuffer<double>(someDataOnCPU);

    std::sort(someDataOnCPU.begin(), someDataOnCPU.end());
    std::sort(dataOnGPU.begin(), dataOnGPU.end());

    auto gpuResults = dataOnGPU.asStdVector();


    BOOST_CHECK_EQUAL_COLLECTIONS(
        gpuResults.begin(), gpuResults.end(), expectedWhenSorted.begin(), expectedWhenSorted.end());

    BOOST_CHECK_EQUAL_COLLECTIONS(
        gpuResults.begin(), gpuResults.end(), someDataOnCPU.begin(), someDataOnCPU.end());
}
