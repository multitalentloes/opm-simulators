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
#include <config.h>

#define BOOST_TEST_MODULE TestBlackOilFluidStateGPU

#include <cuda.h>
#include <cuda_runtime.h>
#include <boost/test/unit_test.hpp>
#include <opm/material/fluidstates/BlackOilFluidState.hpp>
#include <opm/material/fluidsystems/BlackOilFluidSystem.hpp>
#include <opm/material/fluidsystems/BlackOilFluidSystemNonStatic.hpp>
#include <opm/simulators/linalg/gpuistl/detail/gpu_safe_call.hpp>
#include <opm/material/common/HasMemberGeneratorMacros.hpp>
#include <opm/simulators/linalg/gpuistl/gpu_smart_pointer.hpp>

#include <opm/material/fluidsystems/BlackOilFluidSystem.hpp>
#include <opm/material/fluidsystems/BlackOilFluidSystemNonStatic.hpp>
#include <opm/material/fluidstates/BlackOilFluidState.hpp>
#include <opm/material/fluidsystems/blackoilpvt/GasPvtMultiplexer.hpp>
#include <opm/material/fluidsystems/blackoilpvt/WaterPvtMultiplexer.hpp>
#include <opm/material/components/CO2Tables.hpp>
#include <opm/material/densead/Evaluation.hpp>

#include <opm/material/common/Valgrind.hpp>

#include <opm/input/eclipse/Parser/Parser.hpp>
#include <opm/input/eclipse/Deck/Deck.hpp>
#include <opm/input/eclipse/EclipseState/EclipseState.hpp>
#include <opm/input/eclipse/Python/Python.hpp>
#include <opm/input/eclipse/Schedule/Schedule.hpp>

#include <type_traits>
#include <cmath>

#include <opm/simulators/linalg/gpuistl/GpuView.hpp>
#include <opm/simulators/linalg/gpuistl/GpuBuffer.hpp>
#include <opm/simulators/linalg/gpuistl/gpu_smart_pointer.hpp>
#include <opm/simulators/linalg/gpuistl/detail/gpu_safe_call.hpp>

static constexpr const char* deckString1 =
"-- =============== RUNSPEC\n"
"RUNSPEC\n"
"DIMENS\n"
"3 3 3 /\n"
"EQLDIMS\n"
"/\n"
"TABDIMS\n"
"/\n"
"WATER\n"
"GAS\n"
"CO2STORE\n"
"METRIC\n"
"-- =============== GRID\n"
"GRID\n"
"GRIDFILE\n"
"0 0 /\n"
"DX\n"
"27*1 /\n"
"DY\n"
"27*1 /\n"
"DZ\n"
"27*1 /\n"
"TOPS\n"
"9*0 /\n"
"PERMX\n"
"27*1013.25 /\n"
"PORO\n"
"27*0.25 /\n"
"COPY\n"
"PERMX PERMY /\n"
"PERMX PERMZ /\n"
"/\n"
"-- =============== PROPS\n"
"PROPS\n"
"SGWFN\n"
"0.000000E+00 0.000000E+00 1.000000E+00 3.060000E-02\n"
"1.000000E+00 1.000000E+00 0.000000E+00 3.060000E-01 /\n"
"-- =============== SOLUTION\n"
"SOLUTION\n"
"RPTRST\n"
"'BASIC=0' /\n"
"EQUIL\n"
"0 300 100 0 0 0 1 1 0 /\n"
"-- =============== SCHEDULE\n"
"SCHEDULE\n"
"RPTRST\n"
"'BASIC=0' /\n"
"TSTEP\n"
"1 /";

using ScalarT = double;
using Evaluation = Opm::DenseAd::Evaluation<double,2>;

using FluidSystemCPU = Opm::BlackOilFluidSystem<double>;
using FluidSystemNonStatic = Opm::BlackOilFluidSystemNonStatic<double>;

using FluidState = Opm::BlackOilFluidState<ScalarT, FluidSystemCPU>;
using FluidStateDynamic = Opm::BlackOilFluidState<ScalarT, FluidSystemNonStatic>;
using FluidStateDynamicGPU = Opm::BlackOilFluidState<ScalarT, Opm::BlackOilFluidSystemNonStatic<ScalarT, Opm::BlackOilDefaultIndexTraits, Opm::gpuistl::GpuView, Opm::gpuistl::ValueAsPointer>>;
namespace
{
template <class FluidState>
__global__ void kernelCreatingBlackoilFluidState() {
    FluidState state;
}

template<class FluidState>
__global__ void kernelSetAndGetTotalSaturation(double saturation, double* readSaturation) {
    FluidState state;
    state.setTotalSaturation(saturation);
    *readSaturation = state.totalSaturation();
}

template<class FluidState>
__global__ void getPressure(Opm::gpuistl::PointerView<FluidState> input, std::array<double, 3>* output) {
    for (int i = 0; i < 3; ++i) {
        (*output)[i] = input->pressure(i);
    }
}

template<class FluidState, class FluidSystem>
__global__ void getViscosity(FluidSystem input, double* output) {
    FluidState state(input);
    *output = state.viscosity(0);
}

} // namespace

BOOST_AUTO_TEST_CASE(TestCreation)
{
    kernelCreatingBlackoilFluidState<FluidState><<<1, 1>>>();
    OPM_GPU_SAFE_CALL(cudaDeviceSynchronize());
    OPM_GPU_SAFE_CALL(cudaGetLastError());
}

BOOST_AUTO_TEST_CASE(TestSaturation)
{
    const double saturation = 0.5;
    auto saturationRead = Opm::gpuistl::make_gpu_unique_ptr<double>(0.0);
    kernelSetAndGetTotalSaturation<FluidState><<<1, 1>>>(saturation, saturationRead.get());
    auto saturationFromGPU = Opm::gpuistl::copyFromGPU(saturationRead);
    BOOST_CHECK_EQUAL(saturationFromGPU, saturation);
    OPM_GPU_SAFE_CALL(cudaDeviceSynchronize());
    OPM_GPU_SAFE_CALL(cudaGetLastError());
}

BOOST_AUTO_TEST_CASE(TestPressure)
{
    FluidState state;
    state.setPressure(0, 1.0);
    state.setPressure(1, 2.0);
    state.setPressure(2, 3.0);

    auto stateGPU = Opm::gpuistl::make_gpu_unique_ptr<FluidState>(state);
    auto output = Opm::gpuistl::make_gpu_unique_ptr<std::array<double, 3>>();

    getPressure<<<1, 1>>>(Opm::gpuistl::make_view(stateGPU), output.get());
    auto outputCPU = Opm::gpuistl::copyFromGPU(output);
    BOOST_CHECK_EQUAL(1.0, outputCPU[0]);
    BOOST_CHECK_EQUAL(2.0, outputCPU[1]);
    BOOST_CHECK_EQUAL(3.0, outputCPU[2]);
}

BOOST_AUTO_TEST_CASE(TestPassByValueToGPUDynamic)
{
    Opm::Parser parser;

    auto deck = parser.parseString(deckString1);
    auto python = std::make_shared<Opm::Python>();
    Opm::EclipseState eclState(deck);
    Opm::Schedule schedule(deck, eclState, python);

    FluidSystemCPU::initFromState(eclState, schedule);

    auto& dynamicFluidSystem = FluidSystemCPU::getNonStaticInstance();

    auto dynamicGpuFluidSystemBuffer = ::Opm::gpuistl::copy_to_gpu<::Opm::gpuistl::GpuBuffer, double>(dynamicFluidSystem);
    auto system = ::Opm::gpuistl::make_view<::Opm::gpuistl::GpuView, ::Opm::gpuistl::ValueAsPointer>(dynamicGpuFluidSystemBuffer);

    static_assert(
        std::is_same_v<
            decltype(system),
            ::Opm::BlackOilFluidSystemNonStatic
            <
                double,
                ::Opm::BlackOilDefaultIndexTraits,
                ::Opm::gpuistl::GpuView,
                ::Opm::gpuistl::ValueAsPointer
            >
        >
    );

    auto output = Opm::gpuistl::make_gpu_unique_ptr<double>();
    getViscosity<FluidStateDynamicGPU, decltype(system)><<<1, 1>>>(system, output.get());
    auto outputCPU = Opm::gpuistl::copyFromGPU(output);

    FluidState state(FluidSystemCPU{});

    BOOST_CHECK_CLOSE(state.viscosity(0), outputCPU, 1e-10);
}
