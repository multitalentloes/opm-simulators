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
#include "config.h"

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
