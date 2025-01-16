/*
  Copyright 2024 Equinor AS
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

#define BOOST_TEST_MODULE TestGpuBlackOilFluidState

#include <boost/test/unit_test.hpp>
#include <opm/material/fluidstates/BlackOilFluidState.hpp>
#include <opm/material/densead/Evaluation.hpp>
#include <opm/simulators/linalg/gpuistl/detail/gpu_safe_call.hpp>
#include <cuda_runtime.h>

namespace{
__global__ void instantiate_ad_object(Opm::DenseAd::Evaluation<float, 3>* adObj, double value){
    *adObj = Opm::DenseAd::Evaluation<float, 3>(value, 0);
}

//TODO add more comprenehsive AD tests

} // END EMPTY NAMESPACE


BOOST_AUTO_TEST_CASE(TestInstantiateADObject)
{
    using Evaluation = Opm::DenseAd::Evaluation<float, 3>;

    Opm::BlackOilFluidState cpuFluidState;
    cpuFluidState.setPressure(1.0);
    cpuFluidState.setSaturation(0.5);
    cpuFluidState.setTemperature(300.0);

    BOOST_CHECK_EQUAL(cpuFluidState.pressure(), 1.0);
}
