/*
  Copyright 2025 SINTEF AS
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

#define BOOST_TEST_MODULE TestGpuMaterialLaw

#include <boost/test/unit_test.hpp>

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


#include <iostream>
#include <type_traits>
#include <memory>
// #include <dune/common/mpihelper.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <opm/models/utils/start.hh>

  // these types are taken from Norne
  using Scalar = float;
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

    GPUBufferParams gpuBufferParams = Opm::gpuistl::copy_to_gpu<GPUBuffer>(cpuParams);

    GPUViewParams gpuViewParams = Opm::gpuistl::make_view<GPUView>(gpuBufferParams);

    ValueVector testXs = {-1.0, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0, 1.1};

    BOOST_CHECK(true);
}
