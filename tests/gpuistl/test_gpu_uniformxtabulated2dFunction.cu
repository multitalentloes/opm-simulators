/*
  Copyright 2024 SINTEF AS
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

#define BOOST_TEST_MODULE TestGpuUniformXTabulated2DFunction

#include <boost/test/unit_test.hpp>
#include <opm/material/densead/Evaluation.hpp>
#include <opm/simulators/linalg/gpuistl/detail/gpu_safe_call.hpp>
#include <opm/material/common/UniformXTabulated2DFunction.hpp>
#include <opm/simulators/linalg/gpuistl/GpuBuffer.hpp>
#include <opm/simulators/linalg/gpuistl/GpuView.hpp>
#include <cuda_runtime.h>

namespace{
__global__ void instansiate_ad_object(Opm::DenseAd::Evaluation<float, 3>* adObj, double value){
    *adObj = Opm::DenseAd::Evaluation<float, 3>(value, 0);
}

//TODO add more comprenehsive AD tests

} // END EMPTY NAMESPACE


BOOST_AUTO_TEST_CASE(TestInstansiateADObject)
{
    using CpuTable = Opm::UniformXTabulated2DFunction<double>;
    using GpuTable = Opm::UniformXTabulated2DFunction<double, Opm::gpuistl::GpuBuffer>;

    Opm::UniformXTabulated2DFunction<double> cpuTable;
    cpuTable.appendXPos(0.1);
    cpuTable.appendXPos(0.15);
    cpuTable.appendXPos(0.5);
    cpuTable.appendXPos(0.9);
    cpuTable.appendSamplePoint(0, 0.3, 2.0);
    cpuTable.appendSamplePoint(1, 0.5, 5.0);
    cpuTable.appendSamplePoint(2, 0.8, 20.0);
    cpuTable.appendSamplePoint(3, 0.9, 3.0);

    GpuTable gpuTable = Opm::gpuistl::moveToGpu<double, CpuTable, GpuTable, Opm::gpuistl::GpuBuffer>(cpuTable);

    // using Evaluation = Opm::DenseAd::Evaluation<float, 3>;
    // double testValue = 123.456;
    // Evaluation cpuMadeAd = Evaluation(testValue, 0);

    // Evaluation gpuMadeAd[1]; // allocate space for one more AD object on the CPU
    // Evaluation *d_ad;

    // // allocate space on GPU, run kernel, and move results back to the CPU
    // OPM_GPU_SAFE_CALL(cudaMalloc(&d_ad, sizeof(Evaluation)));
    // instansiate_ad_object<<<1,1>>>(d_ad, testValue);
    // OPM_GPU_SAFE_CALL(cudaDeviceSynchronize());
    // OPM_GPU_SAFE_CALL(cudaMemcpy(&gpuMadeAd, d_ad, sizeof(Evaluation), cudaMemcpyDeviceToHost));
    // OPM_GPU_SAFE_CALL(cudaFree(d_ad));

    // // Check that the object made in a GPU kernel is equivalent to that made on the CPU
    // BOOST_CHECK(cpuMadeAd == gpuMadeAd[0]);
}
