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

#define BOOST_TEST_MODULE TestCuView

#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <dune/common/fvector.hh>
#include <dune/istl/bvector.hh>
#include <opm/simulators/linalg/cuistl/CuView.hpp>
#include <opm/simulators/linalg/cuistl/CuBuffer.hpp>
#include <opm/simulators/linalg/cuistl/detail/cuda_safe_call.hpp>
#include <random>
#include <array>
#include <algorithm>
#include <type_traits>

#include <iostream>

template<class integer_t>
struct mystruct
{
    integer_t cnt = 0;

    __device__ __host__ void inc(){
        ++cnt;
    }
};

__global__ void structtest(mystruct<int> s){
    printf("\nSTRUCT VAL: %d\n", s.cnt);
    s.inc();
    printf("\nSTRUCT VAL: %d\n", s.cnt);
}

template<class dataT>
struct simpleView
{
    dataT* dataPtr;

    __device__ __host__ dataT& front(){
        return dataPtr[0];
    }
};

__global__ void structtest(simpleView<float> v, float val){
    v.front() = val;
}

__global__ void useCuViewOnGPU(::Opm::cuistl::CuView<double> view){
    view.front();
    // view.back();

    // view.begin();
    // view.end();

    // view[0] = view[1];
}

//TODO: clang format this file when sketch is done

// BOOST_AUTO_TEST_CASE(TestCreationAndIndexing)
// {
//     // A simple test to check that we can move data to and from the GPU
//     auto cpubuffer = std::vector<double>({1.0, 2.0, 42.0, 59.9451743, 10.7132692});
//     auto cubuffer = ::Opm::cuistl::CuBuffer<double>(cpubuffer);
//     auto cuview = ::Opm::cuistl::CuView<double>(cubuffer.data(), cubuffer.size());
//     const auto const_cuview = ::Opm::cuistl::CuView<double>(cubuffer.data(), cubuffer.size());

//     auto stdVecOfCuView = cuview.asStdVector();
//     auto const_stdVecOfCuView = cuview.asStdVector();

//     BOOST_CHECK_EQUAL_COLLECTIONS(
//         stdVecOfCuView.begin(), stdVecOfCuView.end(), cpubuffer.begin(), cpubuffer.end());
//     BOOST_CHECK_EQUAL_COLLECTIONS(
//         stdVecOfCuView.begin(), stdVecOfCuView.end(), const_stdVecOfCuView.begin(), const_stdVecOfCuView.end());
// }

// BOOST_AUTO_TEST_CASE(TestCuViewOnCPUTypes)
// {
//     auto buf = std::vector<double>({1.0, 2.0, 42.0, 59.9451743, 10.7132692});
//     auto cpuview = ::Opm::cuistl::CuView<double>(buf.data(), buf.size());
//     const auto const_cpuview = ::Opm::cuistl::CuView<double>(buf.data(), buf.size());

//     // check that indexing a mutable view gives references when indexing it
//     bool cpu_front = std::is_same<double&, decltype(cpuview.front())>::value;
//     bool cpu_back = std::is_same<double&, decltype(cpuview.back())>::value;
//     bool const_cpu_front = std::is_same<double, decltype(const_cpuview.front())>::value;
//     bool const_cpu_back = std::is_same<double, decltype(const_cpuview.back())>::value;

//     BOOST_CHECK(cpu_front);
//     BOOST_CHECK(cpu_back);
//     BOOST_CHECK(const_cpu_front);
//     BOOST_CHECK(const_cpu_back);

//     // just checking that these functions exist
//     cpuview.begin();
//     cpuview.end();
//     const_cpuview.begin();
//     const_cpuview.end();
// }

// BOOST_AUTO_TEST_CASE(TestCuViewOnCPUWithSTLIteratorAlgorithm)
// {
//     auto buf = std::vector<double>({1.0, 2.0, 42.0, 59.9451743, 10.7132692});
//     auto cpuview = ::Opm::cuistl::CuView<double>(buf.data(), buf.size());
//     std::sort(buf.begin(), buf.end());
//     BOOST_CHECK(42.0 == cpuview[3]);
// }

// BOOST_AUTO_TEST_CASE(TestStructsOnGPU)
// {
//     mystruct<int> s;
//     s.inc();
//     structtest<<<1,1>>>(s);
//     BOOST_CHECK(true);
// }

// BOOST_AUTO_TEST_CASE(TestSimpleViewOnGPU)
// {
//     float value = 99.0f;
//     simpleView<float> v;
//     float* dev_ptr;
//     OPM_CUDA_SAFE_CALL(cudaMalloc(&dev_ptr, 4*sizeof(float)));
//     v.dataPtr = dev_ptr;
//     structtest<<<1,1>>>(v, value);
//     float cpu_ptr[4];
//     OPM_CUDA_SAFE_CALL(cudaMemcpy(cpu_ptr, v.dataPtr, 4*sizeof(float), cudaMemcpyDeviceToHost));
//     BOOST_CHECK(cpu_ptr[0] == value);
// }

BOOST_AUTO_TEST_CASE(TestCuViewOnGPU)
{
    auto buf = std::vector<double>({1.0, 2.0, 42.0, 59.9451743, 10.7132692});
    auto cubuf = ::Opm::cuistl::CuBuffer<double>(buf);
    auto cuview = ::Opm::cuistl::CuView<double>(cubuf.data(), cubuf.size());
    useCuViewOnGPU<<<1,1>>>(cuview);

    auto vec = cuview.asStdVector();
    BOOST_CHECK(42.0 == cuview[3]);
}
