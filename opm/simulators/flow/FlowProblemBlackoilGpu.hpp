// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  Copyright 2025 EQUINOR

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

#ifndef OPM_FLOW_PROBLEM_BLACKOILGPU_HPP
#define OPM_FLOW_PROBLEM_BLACKOILGPU_HPP

#include <opm/common/utility/VectorWithDefaultAllocator.hpp>
#include <opm/common/utility/gpuDecorators.hpp>

#include <opm/models/discretization/common/linearizationtype.hh>

#include <opm/simulators/flow/FlowProblemBlackoil.hpp>

namespace
{
struct Model {
    struct Linearizer {
        OPM_HOST_DEVICE Opm::LinearizationType getLinearizationType() const
        {
            return linearizationType_;
        }
        Opm::LinearizationType linearizationType_;
    };

    OPM_HOST_DEVICE Linearizer linearizer()
    {
        return linearizer_;
    }

    Linearizer linearizer_;
};

} // namespace

namespace Opm
{

// This class is a simplified version of FlowProblem that should be GPU-instantiable
template <class Scalar, class TypeTag, class MatLawParam, template <class> class Storage = VectorWithDefaultAllocator>
class FlowProblemBlackoilGpu
{
public:
    FlowProblemBlackoilGpu(Storage<unsigned short> satNum,
                           LinearizationType linearizationType,
                           Storage<unsigned short> rockTableIdx,
                           Storage<Scalar> rockCompressibility,
                           Storage<Scalar> rockReferencePressures,
                           std::array<Storage<Scalar>, 2> referencePorosity,
                           Storage<MatLawParam> materialLawParams)
        : satNum_(satNum)
        , linearizationType_(linearizationType)
        , rockTableIdx_(rockTableIdx)
        , rockCompressibility_(rockCompressibility)
        , rockReferencePressures_(rockReferencePressures)
        , referencePorosity_(referencePorosity)
        , materialLawParams_(materialLawParams)
    {
    }

    using EclMaterialLawManager = typename Opm::GetProp<TypeTag, Opm::Properties::MaterialLaw>::EclMaterialLawManager;
    using EclThermalLawManager = typename Opm::GetProp<TypeTag, Opm::Properties::SolidEnergyLaw>::EclThermalLawManager;
    using MaterialLawParams = typename EclMaterialLawManager::MaterialLawParams;
    using IntensiveQuantities = typename Opm::GetPropType<TypeTag, Opm::Properties::IntensiveQuantities>;

    OPM_HOST_DEVICE unsigned short satnumRegionIndex(size_t elemIdx) const
    {
        if (satNum_.size() == 0) {
            return 0;
        }

        return satNum_[elemIdx];
    }

    OPM_HOST_DEVICE Storage<unsigned short>& satnumRegionArray()
    {
        return satNum_;
    }

    // problem.model().linearizer().getLinearizationType()
    OPM_HOST_DEVICE Model model() const
    {
        Model m;
        m.linearizer_.linearizationType_ = linearizationType_;
        return m;
    }

    OPM_HOST_DEVICE Scalar rockCompressibility(unsigned globalSpaceIdx) const
    {
        if (rockCompressibility_.size() == 0)
            return 0.0;

        unsigned tableIdx = 0;
        if (rockTableIdx_.size() > 0) {
            tableIdx = rockTableIdx_[globalSpaceIdx];
        }
        return rockCompressibility_[tableIdx];
    }

    OPM_HOST_DEVICE Scalar rockReferencePressure(unsigned globalSpaceIdx) const
    {
        if (rockReferencePressures_.size() == 0)
            return 1e5;

        unsigned tableIdx = 0;
        if (rockTableIdx_.size() > 0) {
            tableIdx = rockTableIdx_[globalSpaceIdx];
        }
        return rockReferencePressures_[tableIdx];
    }

    OPM_HOST_DEVICE Storage<unsigned short>& rockTableIdx()
    {
        return rockTableIdx_;
    }

    OPM_HOST_DEVICE Storage<Scalar>& rockCompressibilitiesRaw()
    {
        return rockCompressibility_;
    }

    OPM_HOST_DEVICE Storage<Scalar>& rockReferencePressuresRaw()
    {
        return rockReferencePressures_;
    }

    OPM_HOST_DEVICE Scalar porosity(unsigned globalSpaceIdx, unsigned timeIdx) const
    {
        return referencePorosity_[timeIdx][globalSpaceIdx];
    }

    OPM_HOST_DEVICE auto referencePorosity() const
    {
        return referencePorosity_;
    }

    OPM_HOST_DEVICE MatLawParam materialLawParams(std::size_t idx) const
    {
        static_assert(!std::is_same_v<Storage<MatLawParam>, GpuBuffer<MatLawParam>>,
                  "GpuBuffer is not supported as a Storage type for materialLawParams.");
        return materialLawParams_[idx];
    }

    OPM_HOST_DEVICE std::size_t numMaterialLawParams() const
    {
        return materialLawParams_.size();
    }

    // =================================================================================
    // Below are the dummy functions, to be removed

    /*
        NOT USED IN SPE11
    */
    OPM_HOST_DEVICE double maxOilVaporizationFactor(unsigned int, std::size_t) const
    {
        return 0.0;
    }

    /*
        NOT USED IN SPE11
    */
    OPM_HOST_DEVICE double maxGasDissolutionFactor(unsigned int, std::size_t) const
    {
        return 0.0;
    }

    /*
        NOT USED IN SPE11
    */
    OPM_HOST_DEVICE double maxOilSaturation(std::size_t) const
    {
        return 0.0;
    }

    /*
        NOT USED IN SPE11
    */
    template <class Evaluation>
    OPM_HOST_DEVICE Evaluation rockCompPoroMultiplier(const IntensiveQuantities&, std::size_t) const
    {
        return Evaluation(0.0);
    }

    template <class MobArr, class DirMobPtr, class FluidState>
    OPM_HOST_DEVICE void updateRelperms(MobArr& mobility, DirMobPtr& dirMob, const FluidState& fluidstate, std::size_t globalSpaceIdx) const
    {
        // const auto& materialParams = materialLawManager_->materialLawParams(globalDofIdx);;
        // MaterialLaw::relativePermeabilities(mobility, materialParams, fluidState);
    }

    template <class Evaluation>
    OPM_HOST_DEVICE Evaluation rockCompTransMultiplier(const IntensiveQuantities&, std::size_t) const
    {
        return Evaluation(1.0);
    }
    // end dummy functions.

private:
    Storage<unsigned short> satNum_;
    Storage<unsigned short> rockTableIdx_;
    Storage<Scalar> rockCompressibility_;
    Storage<Scalar> rockReferencePressures_;
    std::array<Storage<Scalar>, 2> referencePorosity_;
    LinearizationType linearizationType_;
    Storage<MatLawParam> materialLawParams_;
};

namespace gpuistl
{
    // TODO: why are there two typetags?
    template <class Scalar, template <class> class ContainerT, class TypeTagFrom, class TypeTagTo>
    auto
    copy_to_gpu(FlowProblemBlackoil<TypeTagFrom>& problem)
    {
        using MatLaw = typename Opm::GetProp<TypeTagFrom, Opm::Properties::MaterialLaw>;
        using CpuMgr = typename MatLaw::EclMaterialLawManager;          // == EclMaterialLawManagerSimple<…>
        using CpuParams = typename CpuMgr::MaterialLawParams;              // == EclTwoPhaseMaterialParams<…CpuGasOil, CpuOilWater, CpuGasWater>
        using Traits = typename MatLaw::Traits;
        
        using GpuBuf = ContainerT<Scalar>;
        
        using GasOilTraits   = TwoPhaseMaterialTraits<Scalar,
                                                      Traits::nonWettingPhaseIdx,
                                                      Traits::gasPhaseIdx>;
        using OilWaterTraits = TwoPhaseMaterialTraits<Scalar,
                                                      Traits::wettingPhaseIdx,
                                                      Traits::nonWettingPhaseIdx>;
        using GasWaterTraits = TwoPhaseMaterialTraits<Scalar,
                                                      Traits::wettingPhaseIdx,
                                                      Traits::gasPhaseIdx>;
        
        // now build your new GPU param classes:
        using GpuGasOilParams   = PiecewiseLinearTwoPhaseMaterialParams<GasOilTraits,   GpuBuf>;
        using GpuOilWaterParams = PiecewiseLinearTwoPhaseMaterialParams<OilWaterTraits, GpuBuf>;
        using GpuGasWaterParams = PiecewiseLinearTwoPhaseMaterialParams<GasWaterTraits, GpuBuf>;

        static_assert(std::is_same_v<std::vector<Scalar>, decltype(problem.rockCompressibilitiesRaw())>);
        static_assert(std::is_same_v<std::vector<unsigned short>, decltype(problem.rockTableIdx())>);

        auto nParams = problem.materialLawManager()->numMaterialLawParams();

        using ThreePhaseMaterialParams = Opm::EclTwoPhaseMaterialParams<
            Traits,
            GpuGasOilParams,
            GpuOilWaterParams,
            GpuGasWaterParams
        >;

        auto materialLawParamsInVector = std::vector<ThreePhaseMaterialParams>(nParams);
        for (size_t i = 0; i < nParams; ++i) {
            materialLawParamsInVector[i] =
                ::Opm::gpuistl::copy_to_gpu<
                ContainerT<Scalar>,
                GpuGasOilParams,
                GpuOilWaterParams,
                GpuGasWaterParams,
                Traits
                >(problem.materialLawParams(i));
        }

        return FlowProblemBlackoilGpu<Scalar, TypeTagTo, ThreePhaseMaterialParams, ContainerT>(
            ContainerT(problem.satnumRegionArray()),
            problem.model().linearizer().getLinearizationType(),
            ContainerT(problem.rockTableIdx()),
            ContainerT(problem.rockCompressibilitiesRaw()),
            ContainerT(problem.rockReferencePressuresRaw()),
            std::array<ContainerT<Scalar>, 2>{ContainerT(problem.referencePorosity()[0]), ContainerT(problem.referencePorosity()[1])},
            ContainerT(materialLawParamsInVector)
        );
    }

    template <template <class> class ViewT, template <class> class PtrType, class TypeTag, template <class> class ContainerT, class Scalar, class OldThreePhaseMaterialParams>
    auto
    make_view(FlowProblemBlackoilGpu<Scalar, TypeTag, OldThreePhaseMaterialParams, ContainerT> problem)
    {

        using MatLaw = typename Opm::GetProp<TypeTag, Opm::Properties::MaterialLaw>;
        using CpuMgr = typename MatLaw::EclMaterialLawManager;          // == EclMaterialLawManagerSimple<…>
        using CpuParams = typename CpuMgr::MaterialLawParams;              // == EclTwoPhaseMaterialParams<…CpuGasOil, CpuOilWater, CpuGasWater>
        using Traits = typename MatLaw::Traits;
        
        // using GpuView = ContainerT<Scalar>;
        using GpuView = ViewT<Scalar>;
        
        using GasOilTraits   = TwoPhaseMaterialTraits<Scalar,
                                                      Traits::nonWettingPhaseIdx,
                                                      Traits::gasPhaseIdx>;
        using OilWaterTraits = TwoPhaseMaterialTraits<Scalar,
                                                      Traits::wettingPhaseIdx,
                                                      Traits::nonWettingPhaseIdx>;
        using GasWaterTraits = TwoPhaseMaterialTraits<Scalar,
                                                      Traits::wettingPhaseIdx,
                                                      Traits::gasPhaseIdx>;
        
        // now build your new GPU param classes:
        using GpuGasOilParams   = PiecewiseLinearTwoPhaseMaterialParams<GasOilTraits,   GpuView>;
        using GpuOilWaterParams = PiecewiseLinearTwoPhaseMaterialParams<OilWaterTraits, GpuView>;
        using GpuGasWaterParams = PiecewiseLinearTwoPhaseMaterialParams<GasWaterTraits, GpuView>;

        auto nParams = problem.numMaterialLawParams();

        using ThreePhaseMaterialParams = Opm::EclTwoPhaseMaterialParams<
            Traits,
            GpuGasOilParams,
            GpuOilWaterParams,
            GpuGasWaterParams,
            PtrType
        >;

        auto materialLawParamsInVector = std::vector<ThreePhaseMaterialParams>(nParams);
        for (size_t i = 0; i < nParams; ++i) {
            materialLawParamsInVector[i] =
                ::Opm::gpuistl::make_view<
                ViewT<Scalar>,
                GpuGasOilParams,
                GpuOilWaterParams,
                GpuGasWaterParams,
                PtrType,
                Traits
                >(problem.materialLawParams(i));
        }

        // I now have the values in a regular vector, I guess I can now first make it a buffer and then a view...
        auto parmsInBuffer = ContainerT(materialLawParamsInVector);
        auto parmsInView = make_view(parmsInBuffer);

        return FlowProblemBlackoilGpu<Scalar, TypeTag, ThreePhaseMaterialParams, ViewT>(make_view<unsigned short>(problem.satnumRegionArray()),
                                                              problem.model().linearizer().getLinearizationType(),
                                                              make_view<unsigned short>(problem.rockTableIdx()),
                                                              make_view<Scalar>(problem.rockCompressibilitiesRaw()),
                                                              make_view<Scalar>(problem.rockReferencePressuresRaw()),
                                                              std::array<ViewT<Scalar>, 2>{make_view<Scalar>(problem.referencePorosity()[0]),
                                                              make_view<Scalar>(problem.referencePorosity()[1])},
                                                              parmsInView
                                                            );
    }

} // namespace gpuistl

} // namespace Opm

#endif // OPM_FLOW_PROBLEM_BLACKOILGPU_HPP
