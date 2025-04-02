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

#define BOOST_TEST_MODULE TestFlowProblemGpu

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

#include <cuda_runtime.h>

/*
Functionality tested:
[X] - problem.model().linearizer().getLinearizationType()
[X] - problem.satnumRegionIndex(globalSpaceIdx)
[ ] - problem.materialLawParams(globalSpaceIdx)
[X] - problem.rockCompressibility(globalSpaceIdx)
[ ] - problem.rockReferencePressure(globalSpaceIdx)
[ ] - problem.porosity(globalSpaceIdx, timeIdx)
[ ] - problem.maxOilVaporizationFactor(timeIdx, globalSpaceIdx)
[ ] - problem.maxGasDissolutionFactor(timeIdx, globalSpaceIdx)
[ ] - problem.maxOilSaturation(globalSpaceIdx)
[ ] - problem.template rockCompPoroMultiplier<Evaluation>(*this, globalSpaceIdx)
[ ] - problem.updateRelperms(mobility_, dirMob_, fluidState_, globalSpaceIdx)
[ ] - problem.template rockCompTransMultiplier<Evaluation>(*this, globalSpaceIdx)

*/

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

template<class ProblemView>
__global__ void satnumFromFlowProblemBlackoilGpu(ProblemView prob, unsigned short* res)
{
  *res = prob.satnumRegionIndex(0);
}

template<class ProblemView>
__global__ void linTypeFromFlowProblemBlackoilGpu(ProblemView prob, Opm::LinearizationType* res)
{
  *res = prob.model().linearizer().getLinearizationType();
}

template<class ProblemView>
__global__ void rockCompressibilityFromFlowProblemBlackoilGpu(ProblemView prob, double* res)
{
  *res = prob.rockCompressibility(0);
}

BOOST_AUTO_TEST_CASE(TestInstantiateGpuFlowProblem)
{
  using TypeTag = Opm::Properties::TTag::FlowSimpleProblem;
  // FIXTURE FROM TEST EQUIL
  int argc1 = boost::unit_test::framework::master_test_suite().argc;
  char** argv1 = boost::unit_test::framework::master_test_suite().argv;

#if HAVE_DUNE_FEM
  Dune::Fem::MPIManager::initialize(argc1, argv1);
#else
  Dune::MPIHelper::instance(argc1, argv1);
#endif

  using namespace Opm;
  FlowGenericVanguard::setCommunication(std::make_unique<Opm::Parallel::Communication>());
  Opm::ThreadManager::registerParameters();
  BlackoilModelParameters<double>::registerParameters();
  AdaptiveTimeStepping<TypeTag>::registerParameters();
  Parameters::Register<Parameters::EnableTerminalOutput>("Dummy added for the well model to compile.");
  registerAllParameters_<TypeTag>();

  // END OF FIXTURE FROM TEST EQUIL

  using Simulator = Opm::GetPropType<TypeTag, Opm::Properties::Simulator>;

  // TODO: will this actually refer to the very_simple_deck.DATA inside the gpuistl folder,
  // TODO: do we need to keep track of the path since it can be hipified?
  const char* filename = "very_simple_deck.DATA";
  const auto filenameArg = std::string {"--ecl-deck-file-name="} + filename;

  const char* argv2[] = {
      "test_gpuflowproblem",
      filenameArg.c_str(),
      "--check-satfunc-consistency=false",
  };

  Opm::setupParameters_<TypeTag>(/*argc=*/sizeof(argv2)/sizeof(argv2[0]), argv2, /*registerParams=*/false);

  Opm::FlowGenericVanguard::readDeck(filename);

  auto sim = std::make_unique<Simulator>();

  auto problemGpuBuf = Opm::gpuistl::copy_to_gpu<double, Opm::gpuistl::GpuBuffer, TypeTag, TypeTag>(sim->problem());
  auto problemGpuView = Opm::gpuistl::make_view<Opm::gpuistl::GpuView>(problemGpuBuf);

  unsigned short satNumOnCpu;
  unsigned short* satNumOnGpu;
  cudaMalloc(&satNumOnGpu, sizeof(unsigned short));
  satnumFromFlowProblemBlackoilGpu<<<1, 1>>>(problemGpuView, satNumOnGpu);
  cudaMemcpy(&satNumOnCpu, satNumOnGpu, sizeof(unsigned short), cudaMemcpyDeviceToHost);

  BOOST_CHECK_EQUAL(satNumOnCpu, sim->problem().satnumRegionIndex(0));
  cudaFree(satNumOnGpu);

  Opm::LinearizationType linTypeOnCpu;
  Opm::LinearizationType* linTypeOnGpu;
  cudaMalloc(&linTypeOnGpu, sizeof(Opm::LinearizationType));
  linTypeFromFlowProblemBlackoilGpu<<<1, 1>>>(problemGpuView, linTypeOnGpu);
  cudaMemcpy(&linTypeOnCpu, linTypeOnGpu, sizeof(Opm::LinearizationType), cudaMemcpyDeviceToHost);

  auto linTypeFromCPUSimulator = sim->problem().model().linearizer().getLinearizationType();
  BOOST_CHECK_EQUAL(linTypeOnCpu.type, linTypeFromCPUSimulator.type);
  cudaFree(linTypeOnGpu);

  double rocmCompressibilityOnCpu;
  double* rockCompressibilityOnGpu;
  cudaMalloc(&rockCompressibilityOnGpu, sizeof(double));
  rockCompressibilityFromFlowProblemBlackoilGpu<<<1, 1>>>(problemGpuView, rockCompressibilityOnGpu);
  cudaMemcpy(&rocmCompressibilityOnCpu, rockCompressibilityOnGpu, sizeof(double), cudaMemcpyDeviceToHost);

  BOOST_CHECK_EQUAL(rocmCompressibilityOnCpu, sim->problem().rockCompressibility(0));
  cudaFree(rockCompressibilityOnGpu);


}
