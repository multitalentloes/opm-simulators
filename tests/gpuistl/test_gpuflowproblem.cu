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
#if 0
#include "config.h"

#include <opm/simulators/flow/Main.hpp>
#include <opm/simulators/flow/FlowMain.hpp>
//#define BOOST_TEST_MODULE TestFlowProblemGpu

//#include <boost/test/unit_test.hpp>

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
#include <opm/simulators/linalg/gpuistl/DualBuffer.hpp>
#include <opm/simulators/linalg/gpuistl/GpuView.hpp>
#include <opm/simulators/linalg/gpuistl/gpu_smart_pointer.hpp>
#include <opm/simulators/linalg/gpuistl/detail/gpu_safe_call.hpp>
#include <opm/simulators/wells/BlackoilWellModel.hpp>

#include <utility>

#include <cuda_runtime.h>


#include <fmt/core.h>

/*
Functionality requested for the blackoil flow problem on gpu:
[X] indicates that the functionality is added and verified with unit test
[-] indicates that the functionality does not seem to be used in spe11
[ ] indicates that the functionality is not added yet

[X] - problem.model().linearizer().getLinearizationType()
[X] - problem.satnumRegionIndex(globalSpaceIdx)
[X] - problem.materialLawParams(globalSpaceIdx)
[X] - problem.rockCompressibility(globalSpaceIdx)
[X] - problem.rockReferencePressure(globalSpaceIdx)
[X] - problem.porosity(globalSpaceIdx, timeIdx)
[-] - problem.maxOilVaporizationFactor(timeIdx, globalSpaceIdx)
[-] - problem.maxGasDissolutionFactor(timeIdx, globalSpaceIdx)
[-] - problem.maxOilSaturation(globalSpaceIdx)
[-] - problem.template rockCompPoroMultiplier<Evaluation>(*this, globalSpaceIdx)
[X] - problem.updateRelperms(mobility_, dirMob_, fluidState_, globalSpaceIdx)
[X] - problem.template rockCompTransMultiplier<Evaluation>(*this, globalSpaceIdx)

*/

#include <opm/material/fluidsystems/BlackOilFluidSystem.hpp>
#include <opm/material/fluidsystems/BlackOilFluidSystemNonStatic.hpp>
#include <opm/material/fluidstates/BlackOilFluidState.hpp>

#include <opm/input/eclipse/Parser/Parser.hpp>
#include <opm/input/eclipse/Deck/Deck.hpp>
#include <opm/input/eclipse/EclipseState/EclipseState.hpp>
#include <opm/input/eclipse/Python/Python.hpp>
#include <opm/input/eclipse/Schedule/Schedule.hpp>

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
      // READD THIS ARGUMENT
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

  };

}

// namespace Opm {
//   namespace Properties {
//       namespace TTag {
//           struct FlowSimpleProblem {
//               using InheritsFrom = std::tuple<FlowProblem>;
//           };
//       }

//       // Indices for two-phase gas-water.
//       template<class TypeTag>
//       struct Indices<TypeTag, TTag::FlowSimpleProblem>
//       {
//       private:
//           // it is unfortunately not possible to simply use 'TypeTag' here because this leads
//           // to cyclic definitions of some properties. if this happens the compiler error
//           // messages unfortunately are *really* confusing and not really helpful.
//           using BaseTypeTag = TTag::FlowProblem;
//           using FluidSystem = GetPropType<BaseTypeTag, Properties::FluidSystem>;

//       public:
//           using type = BlackOilTwoPhaseIndices<getPropValue<TypeTag, Properties::EnableSolvent>(),
//                                               getPropValue<TypeTag, Properties::EnableExtbo>(),
//                                               getPropValue<TypeTag, Properties::EnablePolymer>(),
//                                               getPropValue<TypeTag, Properties::EnableEnergy>(),
//                                               getPropValue<TypeTag, Properties::EnableFoam>(),
//                                               getPropValue<TypeTag, Properties::EnableBrine>(),
//                                               /*PVOffset=*/0,
//                                               /*disabledCompIdx=*/FluidSystem::oilCompIdx,
//                                               getPropValue<TypeTag, Properties::EnableMICP>()>;
//       };

//       // SPE11C requires thermal/energy
//       // template<class TypeTag>
//       // struct EnableEnergy<TypeTag, TTag::FlowSimpleProblem> {
//       //     static constexpr bool value = true;
//       // };

//       // SPE11C requires dispersion
//       template<class TypeTag>
//       struct EnableDispersion<TypeTag, TTag::FlowSimpleProblem> {
//           static constexpr bool value = true;
//       };

//       // Use the simple material law.
//       template<class TypeTag>
//       struct MaterialLaw<TypeTag, TTag::FlowSimpleProblem>
//       {
//       private:
//           using Scalar = GetPropType<TypeTag, Properties::Scalar>;
//           using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;
//       public:
//           using Traits = ThreePhaseMaterialTraits<Scalar,
//                                                   /*wettingPhaseIdx=*/FluidSystem::waterPhaseIdx,
//                                                   /*nonWettingPhaseIdx=*/FluidSystem::oilPhaseIdx,
//                                                   /*gasPhaseIdx=*/FluidSystem::gasPhaseIdx>;
//           using EclMaterialLawManager = ::Opm::EclMaterialLawManagerSimple<Traits>;
//           using type = typename EclMaterialLawManager::MaterialLaw;
//       };

//       // Use the TPFA linearizer.
//       template<class TypeTag>
//       struct Linearizer<TypeTag, TTag::FlowSimpleProblem> { using type = TpfaLinearizer<TypeTag>; };

//       template<class TypeTag>
//       struct LocalResidual<TypeTag, TTag::FlowSimpleProblem> { using type = BlackOilLocalResidualTPFA<TypeTag>; };

//       // Diffusion.
//       template<class TypeTag>
//       struct EnableDiffusion<TypeTag, TTag::FlowSimpleProblem> { static constexpr bool value = true; };

//       template<class TypeTag>
//       struct EnableDisgasInWater<TypeTag, TTag::FlowSimpleProblem> { static constexpr bool value = true; };

//       template<class TypeTag>
//       struct EnableVapwat<TypeTag, TTag::FlowSimpleProblem> { static constexpr bool value = true; };
//       // template<class TypeTag>
//       // struct PrimaryVariables<TypeTag, TTag::FlowSimpleProblem> { using type = BlackOilPrimaryVariables<TypeTag, Opm::gpuistl::dense::FieldVector>; };
//   };

// }


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

template<class ProblemView>
__global__ void porosityFromFlowProblemBlackoilGpu(ProblemView prob, double* res)
{
  *res = prob.porosity(0, 0);
}

template<class ProblemView>
__global__ void rockReferencePressureFromFlowProblemBlackoilGpu(ProblemView prob, double* res)
{
  *res = prob.rockReferencePressure(0);
}

template<class ProblemView>
__global__ void materialLawParamsCallable(ProblemView prob)
{
  auto matLawParams = prob.materialLawParams(0);
}

template<class DirMobPtr, class ProblemView, class MobArr, class FluidState>
__global__ void updateRelPermsFromFlowProblemBlackoilGpu(ProblemView prob, MobArr mob, FluidState fs)
{
  auto dirPtr = DirMobPtr(); // produces nullptr, this value is not used in the function, but should match signature
  prob.updateRelperms(mob, dirPtr, fs, 0);
}



//BOOST_AUTO_TEST_CASE(TestInstantiateGpuFlowProblem)
//{
int main(int realArgc, char** realArgv) {
  using TypeTag = Opm::Properties::TTag::FlowSimpleProblem;
  // BOOST_CHECK(true);
  std::vector<std::string> args = {"test_gpuflowproblem", "very_simple_deck.DATA", ""};
  std::vector<char*> argv;
  for (auto& arg : args) {
    argv.push_back(static_cast<char*>(arg.data()));
  }

  // Check if the file specified in args[1] exists
  {
    std::ifstream file(args[1]);
    if (!file.good()) {
      throw std::runtime_error("File not found: " + args[1]);
    }
  }

  fmt::print("New version");
  fmt::print("realArgc: {}", realArgc);
  fmt::print("realArgv: {}", realArgv[0]);
  // BOOST_CHECK(true);
  //auto mainObject = Opm::Main(argv.size(), static_cast<char**>(argv.data()));
  auto mainObject = Opm::Main(realArgc, realArgv);
  // BOOST_CHECK(true);

  // char* argv[] = {const_cast<char*>("very_simple_deck.DATA")};
  // auto mainObject = Opm::Main(1, argv);
  mainObject.runStatic<TypeTag>();
  // mainObject.setupVanguard();
  // BOOST_CHECK(true);
  // auto flowMainObject = Opm::FlowMain<TypeTag>(argv.size(), argv.data(), false, false);
  // BOOST_CHECK(true);
  // // flowMainObject.runStatic<TypeTag>();
  // auto sim1 = flowMainObject.get_simulator();
  // BOOST_CHECK(true);
  // auto& sim = sim1->getSimulator();
  // BOOST_CHECK(true);
  // auto& problem = sim.problem();
  // auto& matLawParams = problem.materialLawParams(0);
  // auto& gasWaterParams = matLawParams.gasWaterParams();
  // gasWaterParams.printme();


//   using TypeTag = Opm::Properties::TTag::FlowSimpleProblem;
//   // FIXTURE FROM TEST EQUIL
//   int argc1 = boost::unit_test::framework::master_test_suite().argc;
//   char** argv1 = boost::unit_test::framework::master_test_suite().argv;

// #if HAVE_DUNE_FEM
//   Dune::Fem::MPIManager::initialize(argc1, argv1);
// #else
//   Dune::MPIHelper::instance(argc1, argv1);
// #endif

//   using namespace Opm;
//   FlowGenericVanguard::setCommunication(std::make_unique<Opm::Parallel::Communication>());
//   Opm::ThreadManager::registerParameters();
//   BlackoilModelParameters<double>::registerParameters();
//   AdaptiveTimeStepping<TypeTag>::registerParameters();
//   Parameters::Register<Parameters::EnableTerminalOutput>("Dummy added for the well model to compile.");
//   registerAllParameters_<TypeTag>();

//   // END OF FIXTURE FROM TEST EQUIL

//   using Simulator = Opm::GetPropType<TypeTag, Opm::Properties::Simulator>;

//   // TODO: will this actually refer to the very_simple_deck.DATA inside the gpuistl folder,
//   // TODO: do we need to keep track of the path since it can be hipified?
//   const char* filename = "very_simple_deck.DATA";
//   const auto filenameArg = std::string {"--ecl-deck-file-name="} + filename;

//   const char* argv2[] = {
//       "test_gpuflowproblem",
//       filenameArg.c_str(),
//       "--check-satfunc-consistency=false",
//   };

//   Opm::setupParameters_<TypeTag>(/*argc=*/sizeof(argv2)/sizeof(argv2[0]), argv2, /*registerParams=*/false);

//   Opm::FlowGenericVanguard::readDeck(filename);

//   auto sim = std::make_unique<Simulator>();

//   using ThreePhaseTraits = typename GetPropType<TypeTag, Properties::MaterialLaw>::Traits;

//   // using ThreePhaseParams = TypeTag::MaterialLaw::EclMaterialLawManager::MaterialLawParams;
//   // using ThreePhaseParams = typename GetPropType<TypeTag, Properties::MaterialLaw>::EclMaterialLawManager::MaterialLawParams;
//   using ThreePhaseParams = typename ::Opm::EclMaterialLawManagerSimple<ThreePhaseTraits>::MaterialLawParams;
//   using CpuGasWaterTwoPhaseLaw = ThreePhaseParams::GasWaterParams;

//   enum { waterPhaseIdx = ThreePhaseTraits::wettingPhaseIdx };
//   enum { oilPhaseIdx = ThreePhaseTraits::nonWettingPhaseIdx };
//   enum { gasPhaseIdx = ThreePhaseTraits::gasPhaseIdx };
//   enum { numPhases = ThreePhaseTraits::numPhases };
//   using GasWaterTraits = TwoPhaseMaterialTraits<double, waterPhaseIdx, gasPhaseIdx>;

//   using GPUBufferInterpolation = Opm::PiecewiseLinearTwoPhaseMaterialParams<GasWaterTraits, Opm::gpuistl::GpuBuffer<double>>;
//   using GPUViewInterpolation = Opm::PiecewiseLinearTwoPhaseMaterialParams<GasWaterTraits, Opm::gpuistl::GpuView<double>>;

//   auto problemGpuBuf = Opm::gpuistl::copy_to_gpu<double, Opm::gpuistl::GpuBuffer, Opm::gpuistl::DualBuffer, TypeTag, TypeTag>(sim->problem());
  // auto problemGpuView = Opm::gpuistl::make_view<Opm::gpuistl::GpuView, Opm::gpuistl::ValueAsPointer>(problemGpuBuf);

  // unsigned short satNumOnCpu;
  // unsigned short* satNumOnGpu;
  // std::ignore = cudaMalloc(&satNumOnGpu, sizeof(unsigned short));
  // satnumFromFlowProblemBlackoilGpu<<<1, 1>>>(problemGpuView, satNumOnGpu);
  // std::ignore = cudaMemcpy(&satNumOnCpu, satNumOnGpu, sizeof(unsigned short), cudaMemcpyDeviceToHost);
  // BOOST_CHECK_EQUAL(satNumOnCpu, sim->problem().satnumRegionIndex(0));
  // std::ignore = cudaFree(satNumOnGpu);

  // Opm::LinearizationType linTypeOnCpu;
  // Opm::LinearizationType* linTypeOnGpu;
  // std::ignore = cudaMalloc(&linTypeOnGpu, sizeof(Opm::LinearizationType));
  // linTypeFromFlowProblemBlackoilGpu<<<1, 1>>>(problemGpuView, linTypeOnGpu);
  // std::ignore = cudaMemcpy(&linTypeOnCpu, linTypeOnGpu, sizeof(Opm::LinearizationType), cudaMemcpyDeviceToHost);
  // auto linTypeFromCPUSimulator = sim->problem().model().linearizer().getLinearizationType();
  // BOOST_CHECK_EQUAL(linTypeOnCpu.type, linTypeFromCPUSimulator.type);
  // std::ignore = cudaFree(linTypeOnGpu);

  // double rocmCompressibilityOnCpu;
  // double* rockCompressibilityOnGpu;
  // std::ignore = cudaMalloc(&rockCompressibilityOnGpu, sizeof(double));
  // rockCompressibilityFromFlowProblemBlackoilGpu<<<1, 1>>>(problemGpuView, rockCompressibilityOnGpu);
  // std::ignore = cudaMemcpy(&rocmCompressibilityOnCpu, rockCompressibilityOnGpu, sizeof(double), cudaMemcpyDeviceToHost);
  // BOOST_CHECK_EQUAL(rocmCompressibilityOnCpu, sim->problem().rockCompressibility(0));
  // std::ignore = cudaFree(rockCompressibilityOnGpu);

  // double porosityOnCpu;
  // double* porosityOnGpu;
  // std::ignore = cudaMalloc(&porosityOnGpu, sizeof(double));
  // porosityFromFlowProblemBlackoilGpu<<<1, 1>>>(problemGpuView, porosityOnGpu);
  // std::ignore = cudaMemcpy(&porosityOnCpu, porosityOnGpu, sizeof(double), cudaMemcpyDeviceToHost);
  // BOOST_CHECK_EQUAL(porosityOnCpu, sim->problem().porosity(0, 0));
  // std::ignore = cudaFree(porosityOnGpu);

  // double referencePressureOnCpu;
  // double* referencePressureOnGpu;
  // std::ignore = cudaMalloc(&referencePressureOnGpu, sizeof(double));
  // rockReferencePressureFromFlowProblemBlackoilGpu<<<1, 1>>>(problemGpuView, referencePressureOnGpu);
  // std::ignore = cudaMemcpy(&referencePressureOnCpu, referencePressureOnGpu, sizeof(double), cudaMemcpyDeviceToHost);
  // BOOST_CHECK_EQUAL(referencePressureOnCpu, sim->problem().rockReferencePressure(0));
  // std::ignore = cudaFree(referencePressureOnGpu);

  // materialLawParamsCallable<<<1, 1>>>(problemGpuView);

  // using FluidSystem = Opm::BlackOilFluidSystem<double>;
  // using Evaluation = Opm::DenseAd::Evaluation<double,2>;
  // using Scalar = double;
  // // using DirectionalMobilityPtr = Utility::CopyablePtr<DirectionalMobility<TypeTag, Evaluation>>;
  // using DirectionalMobilityPtr = Utility::CopyablePtr<DirectionalMobility<TypeTag>>;
  
  // // Create the fluid system
  // Opm::Parser parser;
  // auto deck = parser.parseString(deckString1);
  // auto python = std::make_shared<Opm::Python>();
  // Opm::EclipseState eclState(deck);
  // Opm::Schedule schedule(deck, eclState, python);

  // FluidSystem::initFromState(eclState, schedule);
  // auto& dynamicFluidSystem = FluidSystem::getNonStaticInstance();
  // auto dynamicGpuFluidSystemBuffer = ::Opm::gpuistl::copy_to_gpu<::Opm::gpuistl::GpuBuffer, double>(dynamicFluidSystem);
  // auto dynamicGpuFluidSystemView = ::Opm::gpuistl::make_view<::Opm::gpuistl::GpuView, ::Opm::gpuistl::ValueAsPointer>(dynamicGpuFluidSystemBuffer);
  // auto gpufluidstate = BlackOilFluidState<double, decltype(dynamicGpuFluidSystemView)>(dynamicGpuFluidSystemView);
  // // Create MobArr
  // double testValue = 0.5;
  // // Create an array of Evaluations on CPU
  // using MobArr = std::array<Evaluation, 2>;
  // MobArr cpuMobArray;
  // cpuMobArray[0] = Evaluation(testValue, 0);
  // cpuMobArray[1] = Evaluation(testValue, 1);
  
  // // Copy to GPU
  // MobArr* d_mobArray;
  // OPM_GPU_SAFE_CALL(cudaMalloc(&d_mobArray, sizeof(MobArr)));
  // OPM_GPU_SAFE_CALL(cudaMemcpy(d_mobArray, &cpuMobArray, sizeof(MobArr), cudaMemcpyHostToDevice));
  
  // updateRelPermsFromFlowProblemBlackoilGpu<DirectionalMobilityPtr><<<1, 1>>>(problemGpuView, *d_mobArray, gpufluidstate);
}
#else 
/*
  Copyright 2024, SINTEF AS

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
// #include "config.h"
#define HAS_ATTRIBUTE_UNUSED 1
#define HAS_ATTRIBUTE_DEPRECATED 1
#define HAS_ATTRIBUTE_DEPRECATED_MSG 1
/* #undef HAVE_ARRAY */
/* #undef HAVE_BOOST_MAKE_SHARED_HPP */
/* #undef HAVE_BOOST_SHARED_PTR_HPP */
#define HAVE_DUNE_BOOST 1
/* #undef HAVE_GMP */
/* #undef HAVE_MAKE_SHARED */
#define HAVE_MPI 1
/* #undef HAVE_NULLPTR */
/* #undef HAVE_STATIC_ASSERT */
/* #undef HAVE_SHARED_PTR */
#define MPI_2 1
/* #undef SHARED_PTR_HEADER */
/* #undef SHARED_PTR_NAMESPACE */
/* #undef HAVE_TYPE_TRAITS */
/* #undef HAVE_TR1_TUPLE */
/* #undef HAVE_TUPLE */
#define HAVE_CXA_DEMANGLE 1
/* #undef HAVE_BOOST_FUSION */
/* #undef HAVE_MEM_USAGE_T_EXPANSIONS */
/* #undef HAVE_PARDISO */
#define HAVE_BOOST 1
/* #undef HAVE_PARMETIS */
#define HAVE_SUPERLU 1
#define HAVE_SUITESPARSE_UMFPACK 1
#define SUPERLU_INT_TYPE int
#define HAVE_OPENMP 1
#define HAVE_VALGRIND 1
/* #undef HAVE_FINAL */
#define HAVE_ECL_INPUT 1
#define HAVE_FNMATCH_H 1
/* #undef HAVE_ALGLIB */
/* #undef HAVE_UG */
/* #undef HAVE_DUNE_FEM */
/* #undef HAVE_GRIDTYPE */
/* #undef HAVE_GRAPE */
/* #undef HAVE_PSURFACE */
/* #undef HAVE_AMIRAMESH */
#define HAVE_ALBERTA 1
/* #undef HAVE_STDINT_H */
#define DUNE_GRID_EXPERIMENTAL_GRID_EXTENSIONS 1
#define DUNE_ALBERTA_VERSION 0x300
#define DUNE_GRID_VERSION_MAJOR 2
#define DUNE_GRID_VERSION_MINOR 9
#define DUNE_GRID_VERSION_REVISION 0
#define DUNE_GEOMETRY_VERSION_MAJOR 2
#define DUNE_GEOMETRY_VERSION_MINOR 9
#define DUNE_GEOMETRY_VERSION_REVISION 0
#define DUNE_COMMON_VERSION_MAJOR 2
#define DUNE_COMMON_VERSION_MINOR 9
#define DUNE_COMMON_VERSION_REVISION 1
#define HAVE_DUNE_ISTL 1
/* #undef HAVE_METIS */
/* #undef HAVE_PTSCOTCH */
#define IS_SCOTCH_METIS_HEADER 0
#define HAVE_ZOLTAN 1
#define HAVE_OPM_COMMON 1
#define HAVE_OPM_GRID 1
#define HAVE_PTHREAD 1
/* #undef HAVE_EWOMS */
/* #undef HAVE_PETSC */
/* #undef COMPILE_GPU_BRIDGE */
#define HAVE_CUDA 1
#define HAVE_OPENCL 1
#define HAVE_OPENCL_HPP 1
#define HAVE_AMGCL 1
/* #undef HAVE_AMGX */
/* #undef HAVE_VEXCL */
#define HAVE_ROCALUTION 1
#define HAVE_ROCSPARSE 1
#define HAVE_SUITESPARSE_UMFPACK_H 1
#define HAVE_DUNE_COMMON 1
/* #undef DUNE_ISTL_WITH_CHECKING */
#define DUNE_ISTL_VERSION_MAJOR 2
#define DUNE_ISTL_VERSION_MINOR 9
#define DUNE_ISTL_VERSION_REVISION 0
/* #undef HAVE_DAMARIS */
#define HAVE_HDF5 1
/* #undef HAVE_HYPRE */
#define USE_HIP 1
/* #undef USE_TRACY */
/* #undef FLOW_INSTANTIATE_FLOAT */
#define HAVE_FLOATING_POINT_FROM_CHARS 1
#define OPM_COMPILE_COMPONENTS_TEMPLATE_LIST 2,3,4,5,6,7
#define HAVE_DYNAMIC_BOOST_TEST 1


// #define BOOST_TEST_MODULE TestFlowSimple

// #include <boost/test/unit_test.hpp>

#include <opm/simulators/flow/Main.hpp>
#include <opm/material/fluidmatrixinteractions/EclMaterialLawManagerSimple.hpp>
#include <opm/models/discretization/common/tpfalinearizer.hh>
// do I need these?
#include <opm/simulators/flow/equil/EquilibrationHelpers.hpp>
#include <opm/simulators/flow/equil/InitStateEquil.hpp>

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
        // READD THIS ARGUMENT
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

    };

}
#ifndef BOOST_CHECK
#define BOOST_CHECK(x) std::cout << __LINE__ << std::endl;
#endif
// BOOST_AUTO_TEST_CASE(TestFlowSimple)
int main()
{

  using TypeTag = Opm::Properties::TTag::FlowSimpleProblem;
  BOOST_CHECK(true);
  std::vector<std::string> args = {"./../../../super_build_release/opm-simulators/bin/flow_simple", "very_simple_deck.DATA", ""};
  std::vector<char*> argv;
  for (auto& arg : args) {
    argv.push_back(static_cast<char*>(arg.data()));
  }

  // Check if the file specified in args[1] exists
  {
    std::ifstream file(args[1]);
    if (!file.good()) {
      throw std::runtime_error("File not found: " + args[1]);
    }
  }
  BOOST_CHECK(true);
  auto mainObject = Opm::Main(argv.size(), static_cast<char**>(argv.data()));
  BOOST_CHECK(true);
  
  // char* argv[] = {const_cast<char*>("very_simple_deck.DATA")};
  // auto mainObject = Opm::Main(1, argv);
  mainObject.runStatic<TypeTag>();
//    return Opm::start<TypeTag>(argc, argv);
}
#endif
