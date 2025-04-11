// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  This file is part of the Open Porous Media project (OPM).
  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.
  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/

/*
    This file is based on a copy of the regular blackoilfluidsystem test
    This file contributes extra assertions that the values match on GPU and CPU
*/


/*!
 * \file
 *
 * \brief This is the unit test for the black oil fluid system
 *
 * This test requires the presence of opm-parser.
 */
#include "config.h"
#include <fmt/format.h>
#include <iostream>

#if !HAVE_ECL_INPUT
#error "The test for the black oil fluid system classes requires ecl input support in opm-common"
#endif

#include <boost/mpl/list.hpp>

#define BOOST_TEST_MODULE EclBlackOilFluidSystem
#include <boost/test/unit_test.hpp>

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

using GpuB = Opm::gpuistl::GpuBuffer<double>;
using GpuV = Opm::gpuistl::GpuView<double>;
using GpuBufCo2Tables = Opm::CO2Tables<double, GpuB>;
using GpuBufBrineCo2Pvt = Opm::BrineCo2Pvt<double, GpuBufCo2Tables, GpuB>;
using FluidSystem = Opm::BlackOilFluidSystem<double>;
using Evaluation = Opm::DenseAd::Evaluation<double,2>;
using Scalar = typename Opm::MathToolbox<Evaluation>::Scalar;

// checks that we can access value stored as scalar
template <class IndexTraits>
__global__ void getReservoirTemperature(Opm::BlackOilFluidSystemNonStatic<double, IndexTraits, Opm::gpuistl::GpuView, Opm::gpuistl::ValueAsPointer> fs, double* res)
{
  *res = fs.reservoirTemperature();
}

// checks that we can access value stored in vectors/buffer/views
template <class IndexTraits>
__global__ void getReferenceDensity(Opm::BlackOilFluidSystemNonStatic<double, IndexTraits, Opm::gpuistl::GpuView, Opm::gpuistl::ValueAsPointer> fs, double* res)
{
  *res = fs.referenceDensity(0, 0);
}

// check that we can correctly compute values that require using pvt multiplexers
template <class IndexTraits>
__global__ void getReferenceDensityFromGasPvt(Opm::BlackOilFluidSystemNonStatic<double, IndexTraits, Opm::gpuistl::GpuView, Opm::gpuistl::ValueAsPointer> fs, double* res)
{
  *res = fs.gasPvt().gasReferenceDensity(0);
}

BOOST_AUTO_TEST_CASE(BlackOilFluidSystemOnGpu)
{
    Opm::Parser parser;

    auto deck = parser.parseString(deckString1);
    auto python = std::make_shared<Opm::Python>();
    Opm::EclipseState eclState(deck);
    Opm::Schedule schedule(deck, eclState, python);

    FluidSystem::initFromState(eclState, schedule);

    auto& dynamicFluidSystem = FluidSystem::getNonStaticInstance();

    auto dynamicGpuFluidSystemBuffer = ::Opm::gpuistl::copy_to_gpu<::Opm::gpuistl::GpuBuffer, double>(dynamicFluidSystem);
    auto dynamicGpuFluidSystemView = ::Opm::gpuistl::make_view<::Opm::gpuistl::GpuView, ::Opm::gpuistl::ValueAsPointer>(dynamicGpuFluidSystemBuffer);

    // create a parameter cache
    using ParamCache = typename FluidSystem::template ParameterCache<Scalar>;
    ParamCache paramCache(/*maxOilSat=*/0.5, /*regionIdx=*/1);
    BOOST_CHECK_EQUAL(paramCache.regionIndex(), 1);
    BOOST_CHECK_EQUAL(FluidSystem::numRegions(), 1);
    BOOST_CHECK_EQUAL(FluidSystem::numActivePhases(), 2);

    double GpuComputedVal = 0.0;
    double* gpuComputedValPtr = nullptr;
    OPM_GPU_SAFE_CALL(cudaMalloc(&gpuComputedValPtr, sizeof(double)));
    getReservoirTemperature<<<1, 1>>>(dynamicGpuFluidSystemView, gpuComputedValPtr);
    OPM_GPU_SAFE_CALL(cudaMemcpy(&GpuComputedVal, gpuComputedValPtr, sizeof(double), cudaMemcpyDeviceToHost));
    BOOST_CHECK_CLOSE(FluidSystem::reservoirTemperature(), GpuComputedVal, 1e-10);

    getReferenceDensityFromGasPvt<<<1, 1>>>(dynamicGpuFluidSystemView, gpuComputedValPtr);
    OPM_GPU_SAFE_CALL(cudaMemcpy(&GpuComputedVal, gpuComputedValPtr, sizeof(double), cudaMemcpyDeviceToHost));
    BOOST_CHECK_CLOSE(FluidSystem::gasPvt().gasReferenceDensity(0), GpuComputedVal, 1e-10);

    getReferenceDensity<<<1, 1>>>(dynamicGpuFluidSystemView, gpuComputedValPtr);
    OPM_GPU_SAFE_CALL(cudaMemcpy(&GpuComputedVal, gpuComputedValPtr, sizeof(double), cudaMemcpyDeviceToHost));
    BOOST_CHECK_CLOSE(FluidSystem::referenceDensity(0, 0), GpuComputedVal, 1e-10);


    OPM_GPU_SAFE_CALL(cudaFree(gpuComputedValPtr));
}

__global__ void useGasPvtMultiplexer(Opm::GasPvtMultiplexer<double, true, GpuV, GpuV, Opm::gpuistl::PointerView> gasMultiplexer, double* refTemp)
{
  *refTemp = gasMultiplexer.gasReferenceDensity(0);
}

__global__ void useWaterPvtMultiplexer(Opm::WaterPvtMultiplexer<double, true, true, GpuV, GpuV, Opm::gpuistl::PointerView> waterMultiplexer, double* refTemp)
{
  *refTemp = waterMultiplexer.waterReferenceDensity(0);
}

BOOST_AUTO_TEST_CASE(GasPvtMultiplexer)
{
    Opm::Parser parser;

    auto deck = parser.parseString(deckString1);
    auto python = std::make_shared<Opm::Python>();
    Opm::EclipseState eclState(deck);
    Opm::Schedule schedule(deck, eclState, python);

    FluidSystem::initFromState(eclState, schedule);

    // get and compute CPU pvts
    auto gaspvt = FluidSystem::gasPvt();
    auto cpuGasRefDensity = gaspvt.gasReferenceDensity(0);
    auto waterpvt = FluidSystem::waterPvt();
    auto cpuWaterRefDensity = waterpvt.waterReferenceDensity(0);

    // move pvts to gpu
    auto gpuGasPvtBuf = ::Opm::gpuistl::copy_to_gpu<GpuB, GpuB>(gaspvt);
    auto gpuGasPvtView = ::Opm::gpuistl::make_view<::Opm::gpuistl::PointerView, GpuV, GpuV>(gpuGasPvtBuf);
    auto gpuWaterPvtBuf = ::Opm::gpuistl::copy_to_gpu<GpuB, GpuB>(waterpvt);
    auto gpuWaterPvtView = ::Opm::gpuistl::make_view<::Opm::gpuistl::PointerView, GpuV, GpuV>(gpuWaterPvtBuf);

    double gpuRefDensity = 0.0;
    double* gpuRefDensityPtr = nullptr;
    OPM_GPU_SAFE_CALL(cudaMalloc(&gpuRefDensityPtr, sizeof(double)));

    // check that the GPU computed GAS reference density is correct
    useGasPvtMultiplexer<<<1, 1>>>(gpuGasPvtView, gpuRefDensityPtr);
    OPM_GPU_SAFE_CALL(cudaMemcpy(&gpuRefDensity, gpuRefDensityPtr, sizeof(double), cudaMemcpyDeviceToHost));
    BOOST_CHECK_CLOSE(cpuGasRefDensity, gpuRefDensity, 1e-10);

    // check that the GPU computed WATER reference density is correct
    useWaterPvtMultiplexer<<<1, 1>>>(gpuWaterPvtView, gpuRefDensityPtr);
    OPM_GPU_SAFE_CALL(cudaMemcpy(&gpuRefDensity, gpuRefDensityPtr, sizeof(double), cudaMemcpyDeviceToHost));
    BOOST_CHECK_CLOSE(cpuWaterRefDensity, gpuRefDensity, 1e-10);

    OPM_GPU_SAFE_CALL(cudaFree(gpuRefDensityPtr));
}


#include <opm/material/fluidmatrixinteractions/EclTwoPhaseMaterialParams.hpp>
#include <opm/material/fluidmatrixinteractions/MaterialTraits.hpp>
#include <opm/material/fluidmatrixinteractions/PiecewiseLinearTwoPhaseMaterial.hpp>
#include <opm/material/densead/Evaluation.hpp>
#include <opm/material/fluidmatrixinteractions/EclMaterialLawManagerSimple.hpp>

#include <opm/models/blackoil/blackoilmodel.hh>
#include <opm/models/discretization/common/tpfalinearizer.hh>
#include <opm/models/utils/simulator.hh>

#include <opm/simulators/utils/moduleVersion.hpp>
#include <opm/simulators/flow/FlowProblemBlackoilGpu.hpp>
#include <opm/simulators/flow/FlowProblemBlackoil.hpp>
#include <opm/simulators/flow/FlowProblemBlackoilProperties.hpp>

#include <opm/simulators/flow/BlackoilModelParameters.hpp>
#include <opm/simulators/flow/FlowGenericVanguard.hpp>
#include <opm/simulators/flow/FlowProblemBlackoil.hpp>
#include <opm/simulators/flow/FlowProblemBlackoilProperties.hpp>
#include <opm/simulators/flow/equil/EquilibrationHelpers.hpp>
#include <opm/simulators/linalg/parallelbicgstabbackend.hh>
#include <opm/simulators/linalg/gpuistl/GpuBuffer.hpp>
#include <opm/simulators/linalg/gpuistl/GpuView.hpp>
#include <opm/simulators/linalg/gpuistl/gpu_smart_pointer.hpp>
#include <opm/simulators/linalg/gpuistl/detail/gpu_safe_call.hpp>
#include <opm/simulators/wells/BlackoilWellModel.hpp>

#include <utility>

#include <cuda_runtime.h>

namespace Opm {
  namespace Properties {
      namespace TTag {
          struct FlowSimpleProblem {
              using InheritsFrom = std::tuple<FlowProblem>;
          };
      }

      // Indices for two-phase gas-water.
      template<class TypeTag>
      struct Indices<TypeTag, TTag::FlowSimpleProblem>
      {
      private:
          // it is unfortunately not possible to simply use 'TypeTag' here because this leads
          // to cyclic definitions of some properties. if this happens the compiler error
          // messages unfortunately are *really* confusing and not really helpful.
          using BaseTypeTag = TTag::FlowProblem;
          using FluidSystem = GetPropType<BaseTypeTag, Properties::FluidSystem>;

      public:
          using type = BlackOilTwoPhaseIndices<getPropValue<TypeTag, Properties::EnableSolvent>(),
                                              getPropValue<TypeTag, Properties::EnableExtbo>(),
                                              getPropValue<TypeTag, Properties::EnablePolymer>(),
                                              getPropValue<TypeTag, Properties::EnableEnergy>(),
                                              getPropValue<TypeTag, Properties::EnableFoam>(),
                                              getPropValue<TypeTag, Properties::EnableBrine>(),
                                              /*PVOffset=*/0,
                                              /*disabledCompIdx=*/FluidSystem::oilCompIdx,
                                              getPropValue<TypeTag, Properties::EnableMICP>()>;
      };

      // SPE11C requires thermal/energy
      // template<class TypeTag>
      // struct EnableEnergy<TypeTag, TTag::FlowSimpleProblem> {
      //     static constexpr bool value = true;
      // };

      // SPE11C requires dispersion
      template<class TypeTag>
      struct EnableDispersion<TypeTag, TTag::FlowSimpleProblem> {
          static constexpr bool value = true;
      };

      // Use the simple material law.
      template<class TypeTag>
      struct MaterialLaw<TypeTag, TTag::FlowSimpleProblem>
      {
      private:
          using Scalar = GetPropType<TypeTag, Properties::Scalar>;
          using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;

          using Traits = ThreePhaseMaterialTraits<Scalar,
                                                  /*wettingPhaseIdx=*/FluidSystem::waterPhaseIdx,
                                                  /*nonWettingPhaseIdx=*/FluidSystem::oilPhaseIdx,
                                                  /*gasPhaseIdx=*/FluidSystem::gasPhaseIdx>;
      public:
          using EclMaterialLawManager = ::Opm::EclMaterialLawManagerSimple<Traits>;
          using type = typename EclMaterialLawManager::MaterialLaw;
      };

      // Use the TPFA linearizer.
      template<class TypeTag>
      struct Linearizer<TypeTag, TTag::FlowSimpleProblem> { using type = TpfaLinearizer<TypeTag>; };

      template<class TypeTag>
      struct LocalResidual<TypeTag, TTag::FlowSimpleProblem> { using type = BlackOilLocalResidualTPFA<TypeTag>; };

      // Diffusion.
      template<class TypeTag>
      struct EnableDiffusion<TypeTag, TTag::FlowSimpleProblem> { static constexpr bool value = true; };

      template<class TypeTag>
      struct EnableDisgasInWater<TypeTag, TTag::FlowSimpleProblem> { static constexpr bool value = true; };

      template<class TypeTag>
      struct EnableVapwat<TypeTag, TTag::FlowSimpleProblem> { static constexpr bool value = true; };
      // template<class TypeTag>
      // struct PrimaryVariables<TypeTag, TTag::FlowSimpleProblem> { using type = BlackOilPrimaryVariables<TypeTag, Opm::gpuistl::dense::FieldVector>; };
  };

}

  // these types are taken from Norne
  using ValueVector = std::vector<Scalar>;
  using GPUBuffer = Opm::gpuistl::GpuBuffer<Scalar>;
  using GPUView = Opm::gpuistl::GpuView<Scalar>;

  using TraitsT = Opm::TwoPhaseMaterialTraits<Scalar, 1, 2>;
  using CPUParams = Opm::PiecewiseLinearTwoPhaseMaterialParams<TraitsT>;
  using GPUBufferParams = Opm::PiecewiseLinearTwoPhaseMaterialParams<TraitsT, GPUBuffer>;
  using GPUViewParams = Opm::PiecewiseLinearTwoPhaseMaterialParams<TraitsT, GPUView>;

  using CPUTwoPhaseMaterialLaw = Opm::PiecewiseLinearTwoPhaseMaterial<TraitsT, CPUParams>;
  using GPUTwoPhaseViewMaterialLaw = Opm::PiecewiseLinearTwoPhaseMaterial<TraitsT, GPUViewParams>;
  using NorneEvaluation = Opm::DenseAd::Evaluation<Scalar, 3, 0u>;

__global__ void gpuTwoPhaseSatPcnwWrapper(GPUTwoPhaseViewMaterialLaw::Params params, NorneEvaluation* Sw, NorneEvaluation* res){
    *res = GPUTwoPhaseViewMaterialLaw::twoPhaseSatPcnw(params, *Sw);
}

// using OPMFS = Opm::Properties::FluidSystem;
using ProblemType = Opm::Properties::TTag::FlowSimpleProblem;
using namespace Opm;

using Traits = ThreePhaseMaterialTraits<Scalar,
/*wettingPhaseIdx=*/GetPropType<ProblemType, Opm::Properties::FluidSystem>::waterPhaseIdx,
/*nonWettingPhaseIdx=*/GetPropType<ProblemType, Opm::Properties::FluidSystem>::oilPhaseIdx,
/*gasPhaseIdx=*/GetPropType<ProblemType, Opm::Properties::FluidSystem>::gasPhaseIdx>;

BOOST_AUTO_TEST_CASE(TestSimpleInterpolation)
{

    ValueVector cx = {0.0, 0.5, 1.0};
    ValueVector cy = {0.0, 0.9, 1.0};
    const GPUBuffer gx(cx);
    const GPUBuffer gy(cy);

    CPUParams cpuParams;
    cpuParams.setPcnwSamples(cx, cy);
    cpuParams.setKrwSamples(cx, cy);
    cpuParams.setKrnSamples(cx, cy);
    cpuParams.finalize();

    std::shared_ptr<CPUParams> cpuParamsPtr = std::make_shared<CPUParams>(cpuParams);

    GPUBufferParams gpuBufferParams = gpuistl::copy_to_gpu<GPUBuffer>(cpuParams);
    GPUViewParams gpuViewParams = gpuistl::make_view<GPUView>(gpuBufferParams);

    EclTwoPhaseMaterialParams<Traits, CPUParams, CPUParams, CPUParams> cpuTwoPhaseParams;
    cpuTwoPhaseParams.setApproach(EclTwoPhaseApproach::GasWater);
    cpuTwoPhaseParams.setGasWaterParams(cpuParamsPtr);

    auto gpuTwoPhaseParamsBuffer = gpuistl::copy_to_gpu<GPUBuffer, GPUBufferParams, GPUBufferParams, GPUBufferParams>(cpuTwoPhaseParams);
    auto gpuTwoPhaseParamsView = gpuistl::make_view<GPUView, GPUViewParams, GPUViewParams, GPUViewParams, gpuistl::PointerView>(gpuTwoPhaseParamsBuffer);

    BOOST_CHECK(true);

    Opm::Parser parser;

    auto deck = parser.parseString(deckString1);
    auto python = std::make_shared<Opm::Python>();
    Opm::EclipseState eclState(deck);
    Opm::Schedule schedule(deck, eclState, python);

    FluidSystem::initFromState(eclState, schedule);

    auto& dynamicFluidSystem = FluidSystem::getNonStaticInstance();

    auto dynamicGpuFluidSystemBuffer = ::Opm::gpuistl::copy_to_gpu<::Opm::gpuistl::GpuBuffer, double>(dynamicFluidSystem);
    auto dynamicGpuFluidSystemView = ::Opm::gpuistl::make_view<::Opm::gpuistl::GpuView, ::Opm::gpuistl::ValueAsPointer>(dynamicGpuFluidSystemBuffer);
    BOOST_CHECK(true);
}
