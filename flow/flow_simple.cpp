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

int main(int argc, char** argv)
{
    using TypeTag = Opm::Properties::TTag::FlowSimpleProblem;
    auto mainObject = Opm::Main(argc, argv);
    return mainObject.runStatic<TypeTag>();
//    return Opm::start<TypeTag>(argc, argv);
}
