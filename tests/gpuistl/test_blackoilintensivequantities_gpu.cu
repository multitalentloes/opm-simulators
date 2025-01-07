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
#define HAVE_ECL_INPUT 1
#include <config.h>


#include <stdexcept>

#define BOOST_TEST_MODULE TestBlackOilIntensiveQuantitiesGPU

#include <cuda.h>
#include <cuda_runtime.h>
#include <boost/test/unit_test.hpp>
#include <opm/common/ErrorMacros.hpp>
#include <opm/models/io/dgfvanguard.hh>
#include <opm/models/utils/start.hh>
#include <opm/models/blackoil/blackoilmodel.hh>
#include <opm/models/discretization/common/fvbaseprimaryvariables.hh>
#include <opm/models/blackoil/blackoilprimaryvariables.hh>
#include <opm/simulators/linalg/gpuistl/detail/gpu_safe_call.hpp>
#include <opm/simulators/linalg/gpuistl/dense/DenseVector.hpp>
#include <opm/simulators/linalg/gpuistl/dense/FieldVector.hpp>
#include <opm/simulators/linalg/gpuistl/gpu_smart_pointer.hpp>
#include <opm/simulators/flow/Main.hpp>
#include <opm/models/discretization/common/tpfalinearizer.hh>
// do I need these?
#include <opm/simulators/flow/equil/EquilibrationHelpers.hpp>
#include <opm/simulators/flow/equil/InitStateEquil.hpp>
#include <opm/models/blackoil/blackoilintensivequantities.hh>
#include <opm/material/fluidmatrixinteractions/EclMaterialLawManagerSimple.hpp>
#include <opm/input/eclipse/Parser/Parser.hpp>
#include <opm/input/eclipse/Deck/Deck.hpp>
#include <opm/input/eclipse/EclipseState/EclipseState.hpp>
#include <opm/input/eclipse/Python/Python.hpp>
#include <opm/input/eclipse/Schedule/Schedule.hpp>
#include <opm/material/fluidsystems/BlackOilFluidSystem.hpp>
#include <opm/material/fluidsystems/BlackOilFluidSystemNonStatic.hpp>
#include <opm/simulators/flow/FlowProblemBlackoilGpu.hpp>

#include <opm/simulators/linalg/gpuistl/GpuBuffer.hpp>
#include <opm/simulators/linalg/gpuistl/GpuView.hpp>
#include <opm/simulators/linalg/gpuistl/gpu_smart_pointer.hpp>

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

#include <chrono>


static constexpr const char* deckString1 =
"-- =============== RUNSPEC\n"
"RUNSPEC\n"
"DIMENS\n"
"200 200 200 /\n"
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
"8000000*1 /\n"
"DY\n"
"8000000*1 /\n"
"DZ\n"
"8000000*1 /\n"
"TOPS\n"
"40000*0 /\n"
"PERMX\n"
"8000000*1013.25 /\n"
"PORO\n"
"8000000*0.25 /\n"
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
using BlackOilFluidSystemView = Opm::BlackOilFluidSystemNonStatic<double, Opm::BlackOilDefaultFluidSystemIndices, Opm::gpuistl::GpuView, Opm::gpuistl::ValueAsPointer>;
// template<class TypeTag>
// using FlowProblemView = Opm::FlowProblemBlackoilGpu<double, TypeTag, Opm::gpuistl::GpuView>;

template<class TypeTag>
struct DummyProblem {
  using EclMaterialLawManager = typename Opm::GetProp<TypeTag, Opm::Properties::MaterialLaw>::EclMaterialLawManager;
  using EclThermalLawManager = typename Opm::GetProp<TypeTag, Opm::Properties::SolidEnergyLaw>::EclThermalLawManager;
  using MaterialLawParams = typename EclMaterialLawManager::MaterialLawParams;
  using IntensiveQuantities = typename Opm::GetPropType<TypeTag, Opm::Properties::IntensiveQuantities>;
  struct {
    struct {
      OPM_HOST_DEVICE Opm::LinearizationType getLinearizationType() const { return Opm::LinearizationType(); }
    } lin_;

    OPM_HOST_DEVICE auto linearizer() const { return lin_; }
  } model_;

  OPM_HOST_DEVICE auto model() const { return model_; }

  OPM_HOST_DEVICE int satnumRegionIndex(std::size_t) const { return 0; }
  OPM_HOST_DEVICE MaterialLawParams materialLawParams(std::size_t) const { return MaterialLawParams(); }
  OPM_HOST_DEVICE double rockCompressibility(std::size_t) const { return 0.0; }
  OPM_HOST_DEVICE double rockReferencePressure(std::size_t) const { return 0.0; }
  OPM_HOST_DEVICE double porosity(std::size_t, unsigned int) const { return 0.0; }
  OPM_HOST_DEVICE double maxOilVaporizationFactor(unsigned int, std::size_t) const { return 0.0; }
  OPM_HOST_DEVICE double maxGasDissolutionFactor(unsigned int, std::size_t) const { return 0.0; }
  OPM_HOST_DEVICE double maxOilSaturation(std::size_t) const { return 0.0; }

  template<class Evaluation>
  OPM_HOST_DEVICE Evaluation rockCompPoroMultiplier(const IntensiveQuantities&, std::size_t) const {
    return Evaluation(0.0);
  }

  template<class A, class B, class C>
  OPM_HOST_DEVICE void updateRelperms(A&, B&, const C&, std::size_t) const {}

  template<class Evaluation>
  OPM_HOST_DEVICE Evaluation rockCompTransMultiplier(const IntensiveQuantities&, std::size_t) const {
    return Evaluation(0.0);
  }
};
namespace Opm {
  namespace Properties {
      namespace TTag {
          struct FlowSimpleProblem {
              using InheritsFrom = std::tuple<FlowProblem>;
          };

          struct FlowSimpleProblemGPU {
              using InheritsFrom = std::tuple<FlowSimpleProblem>;
          };

          struct FlowSimpleDummyProblemGPU {
            using InheritsFrom = std::tuple<FlowSimpleProblem>;
        };

          struct FlowSimpleDummyProblemCPU {
              using InheritsFrom = std::tuple<FlowSimpleProblem>;
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
      template<class TypeTag>
      struct EnableEnergy<TypeTag, TTag::FlowSimpleProblem> {
          static constexpr bool value = true;
      };

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

    //   template<class TypeTag>
    //   struct MaterialLaw<TypeTag, TTag::FlowSimpleProblemGPU>
    //   {
    //   private:
    //       using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    //       using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;

    //       using Traits = ThreePhaseMaterialTraits<Scalar,
    //                                               /*wettingPhaseIdx=*/FluidSystem::waterPhaseIdx,
    //                                               /*nonWettingPhaseIdx=*/FluidSystem::oilPhaseIdx,
    //                                               /*gasPhaseIdx=*/FluidSystem::gasPhaseIdx>;
    //   public:
    //       using EclMaterialLawManager = ::Opm::EclMaterialLawManagerSimple<Traits, Opm::gpuistl::GpuView>;
    //       using type = typename EclMaterialLawManager::MaterialLaw;
    //   };

      template<class TypeTag>
      struct MaterialLaw<TypeTag, TTag::FlowSimpleDummyProblemGPU>
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

      template<class TypeTag>
      struct MaterialLaw<TypeTag, TTag::FlowSimpleDummyProblemCPU>
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

      template<class TypeTag>
      struct PrimaryVariables<TypeTag, TTag::FlowSimpleProblem> { using type = BlackOilPrimaryVariables<TypeTag>; };

      template<class TypeTag>
      struct PrimaryVariables<TypeTag, TTag::FlowSimpleDummyProblemGPU> { using type = BlackOilPrimaryVariables<TypeTag, Opm::gpuistl::dense::FieldVector>; };

            template<class TypeTag>
      struct PrimaryVariables<TypeTag, TTag::FlowSimpleDummyProblemCPU> { using type = BlackOilPrimaryVariables<TypeTag>; };
      template<class TypeTag>
      struct IntensiveQuantities<TypeTag, TTag::FlowSimpleProblem> { using type = BlackOilIntensiveQuantities<TypeTag>; };

      template<class TypeTag>
      struct Problem<TypeTag, TTag::FlowSimpleDummyProblemGPU> { using type = DummyProblem<TypeTag>; };

      template<class TypeTag>
      struct FluidSystem<TypeTag, TTag::FlowSimpleDummyProblemGPU> { using type = BlackOilFluidSystemView; };

      template<class TypeTag>
      struct Problem<TypeTag, TTag::FlowSimpleDummyProblemCPU> { using type = DummyProblem<TypeTag>; };

      template<class TypeTag>
      struct FluidSystem<TypeTag, TTag::FlowSimpleDummyProblemCPU> { using type = BlackOilFluidSystem<double>; };

    //   template<class TypeTag>
    //   struct Problem<TypeTag, TTag::FlowSimpleProblemGPU> { using type = Opm::FlowProblemBlackoilGpu<double, TypeTag, Opm::gpuistl::GpuView>.template <TypeTag>; };

    //   template<class TypeTag>
    //   struct FluidSystem<TypeTag, TTag::FlowSimpleProblemGPU> { using type = BlackOilFluidSystemView; };

  };

}



using TypeTag = Opm::Properties::TTag::FlowSimpleProblem;
using TypeTagDummyGpu = Opm::Properties::TTag::FlowSimpleDummyProblemGPU;
using TypeTagDummyCpu = Opm::Properties::TTag::FlowSimpleDummyProblemCPU;

#if 1

#endif
namespace {

__host__ __device__ void wrapper(BlackOilFluidSystemView& fs, DummyProblem<TypeTagDummyGpu> p, Opm::BlackOilPrimaryVariables<TypeTagDummyGpu, Opm::gpuistl::dense::FieldVector> primvar, size_t idx) {
    Opm::BlackOilIntensiveQuantities<TypeTagDummyGpu> intensiveQuantities (&fs);

    intensiveQuantities.void_update(p, primvar, idx);
}

  __global__ void fake_update_gpu(BlackOilFluidSystemView fs, DummyProblem<TypeTagDummyGpu> p, Opm::BlackOilPrimaryVariables<TypeTagDummyGpu, Opm::gpuistl::dense::FieldVector>  primvar, size_t lim) {
      const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < lim) {
          wrapper(fs, p, primvar, idx);
      }
  }

//   template<class ProblemType>
//   __global__ void testCreationGPUWithProblem(BlackOilFluidSystemView fs, ProblemType problem) {


//     Opm::BlackOilPrimaryVariables<TypeTagDummyGpu, Opm::gpuistl::dense::FieldVector> primaryVariables;
//     Opm::BlackOilIntensiveQuantities<TypeTagDummyGpu> intensiveQuantities (&fs);
//     auto& state = intensiveQuantities.fluidState();
//     printf("BlackOilState density before update: %f\n", state.density(0).value());
//     intensiveQuantities.updatePhaseDensities();
//     printf("BlackOilState density after update: %f\n", state.density(0).value());

//     intensiveQuantities.update(problem, primaryVariables, 0, 0);
//     printf("Updating succeeded");
//   }
}

// int main(int, char**)
// {
BOOST_AUTO_TEST_CASE(TestPrimaryVariablesCreationGPU)
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

    // Opm::BlackOilIntensiveQuantities<TypeTag> intensiveQuantities;
    Opm::BlackOilIntensiveQuantities<TypeTagDummyCpu> intensiveQuantities;

    // intensiveQuantities.printme();
    // auto& state = intensiveQuantities.fluidState();
    // printf("(CPU) BlackOilState density before update: %f\n", state.density(0).value());
    // intensiveQuantities.updatePhaseDensities();
    // printf("(CPU) BlackOilState density after update: %f\n", state.density(0).value());

    DummyProblem<TypeTagDummyCpu> problem;
    Opm::BlackOilPrimaryVariables<TypeTagDummyCpu> primaryVariables;
    auto start_time = std::chrono::high_resolution_clock::now();
    intensiveQuantities.void_update(problem, primaryVariables, 10);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "CPU void_update execution time: " << duration << " microseconds" << std::endl;

    // using PrimaryVariables = Opm::GetPropType<TypeTag, Opm::Properties::PrimaryVariables>;
    // std::cout << typeid(PrimaryVariables).name() << std::endl;

    size_t cells = 10;
    size_t threads = 1024;
    size_t blocks = (cells + threads - 1) / threads;
    DummyProblem<TypeTagDummyGpu> gpuProblem;
    Opm::BlackOilPrimaryVariables<TypeTagDummyGpu, Opm::gpuistl::dense::FieldVector> primaryVariablesGpu;
    // Opm::BlackOilIntensiveQuantities<TypeTagDummyGpu>
    auto gpu_start_time = std::chrono::high_resolution_clock::now();
    fake_update_gpu<<<blocks, threads>>>(dynamicGpuFluidSystemView, gpuProblem, primaryVariablesGpu, cells);
    OPM_GPU_SAFE_CALL(cudaDeviceSynchronize());
    OPM_GPU_SAFE_CALL(cudaGetLastError());
    auto gpu_end_time = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end_time - gpu_start_time).count();
    std::cout << "GPU void_update execution time: " << gpu_duration << " microseconds" << std::endl;
    printf("GPU void_update finished\n");
}

#if 0
BOOST_AUTO_TEST_CASE(TestInstantiateGpuFlowProblem)
{
    BOOST_CHECK(true);
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
//   registerAllParameters_<TypeTag>(true);

//   // END OF FIXTURE FROM TEST EQUIL

//   using Simulator = Opm::GetPropType<TypeTag, Opm::Properties::Simulator>;

//   // TODO: will this actually refer to the very_simple_deck.DATA inside the gpuistl folder,
//   // TODO: do we need to keep track of the path since it can be hipified?
//   const std::string filename = "very_simple_deck.DATA";
//   const auto filenameArg = std::string {"--ecl-deck-file-name="} + filename;

//   const char* argv2[] = {
//       "test_gpuflowproblem",
//       filenameArg.c_str(),
//       "--check-satfunc-consistency=false",
//   };

//   Opm::setupParameters_<TypeTag>(/*argc=*/sizeof(argv2)/sizeof(argv2[0]), argv2, /*registerParams=*/false, false, true, 0);

//   Opm::FlowGenericVanguard::readDeck(filename);

//   auto sim = std::make_unique<Simulator>();

//   auto problemGpuBuf = Opm::gpuistl::copy_to_gpu<double, Opm::gpuistl::GpuBuffer, TypeTag, TypeTagDummyGpu>(sim->problem());
//   auto problemGpuView = Opm::gpuistl::make_view<Opm::gpuistl::GpuView>(problemGpuBuf);
//   auto& dynamicFluidSystem = FluidSystem::getNonStaticInstance();

//   auto dynamicGpuFluidSystemBuffer = ::Opm::gpuistl::copy_to_gpu<::Opm::gpuistl::GpuBuffer, double>(dynamicFluidSystem);
//   auto dynamicGpuFluidSystemView = ::Opm::gpuistl::make_view<::Opm::gpuistl::GpuView, ::Opm::gpuistl::ValueAsPointer>(dynamicGpuFluidSystemBuffer);

//   testCreationGPUWithProblem<<<1, 1>>>(dynamicGpuFluidSystemView, problemGpuView);
}



// ==========================================
// TMP JUST TO MAKE IT COMPILE:
// #include "opm/simulators/linalg/gpuistl/detail/cublas_safe_call.hpp"
// #include <config.h>

// #define BOOST_TEST_MODULE TestCublasHandle

// #include <cuda_runtime.h>
// #include <boost/test/unit_test.hpp>
// #include <opm/simulators/linalg/gpuistl/detail/CuBlasHandle.hpp>

// BOOST_AUTO_TEST_CASE(TestGetCublasVersion)
// {
// #if USE_HIP
//     // As of April 2024 it does not seem that hip has implemented the function
//     // that checks the version of blas programatically. Let the test pass for now.
//     BOOST_CHECK(true);
// #else
//     auto& cublasHandle = ::Opm::gpuistl::detail::CuBlasHandle::getInstance();
//     int cuBlasVersion = -1;
//     OPM_CUBLAS_SAFE_CALL(cublasGetVersion(cublasHandle.get(), &cuBlasVersion));

//     BOOST_CHECK_LT(0, cuBlasVersion);
// #endif
// }
#endif
