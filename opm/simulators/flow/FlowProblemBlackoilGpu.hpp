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
template <class Scalar, class TypeTag, template <class> class Storage = VectorWithDefaultAllocator>
class FlowProblemBlackoilGpu
{
public:
    FlowProblemBlackoilGpu(Storage<unsigned short> satNum,
                           LinearizationType linearizationType,
                           Storage<unsigned short> rockTableIdx,
                           Storage<Scalar> rockCompressibility,
                           Storage<Scalar> rockReferencePressures,
                           std::array<Storage<Scalar>, 2> referencePorosity)
        : satNum_(satNum)
        , linearizationType_(linearizationType)
        , rockTableIdx_(rockTableIdx)
        , rockCompressibility_(rockCompressibility)
        , rockReferencePressures_(rockReferencePressures)
        , referencePorosity_(referencePorosity)
    {
    }

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

    // Below are the dummy functions, to be removed
    using EclMaterialLawManager = typename Opm::GetProp<TypeTag, Opm::Properties::MaterialLaw>::EclMaterialLawManager;
    using EclThermalLawManager = typename Opm::GetProp<TypeTag, Opm::Properties::SolidEnergyLaw>::EclThermalLawManager;
    using MaterialLawParams = typename EclMaterialLawManager::MaterialLawParams;
    using IntensiveQuantities = typename Opm::GetPropType<TypeTag, Opm::Properties::IntensiveQuantities>;

    OPM_HOST_DEVICE MaterialLawParams materialLawParams(std::size_t) const
    {
        return MaterialLawParams();
    }

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

    template <class A, class B, class C>
    OPM_HOST_DEVICE void updateRelperms(A&, B&, const C&, std::size_t) const
    {
    }

    template <class Evaluation>
    OPM_HOST_DEVICE Evaluation rockCompTransMultiplier(const IntensiveQuantities&, std::size_t) const
    {
        return Evaluation(0.0);
    }
    // end dummy functions.

private:
    Storage<unsigned short> satNum_;
    Storage<unsigned short> rockTableIdx_;
    Storage<Scalar> rockCompressibility_;
    Storage<Scalar> rockReferencePressures_;
    std::array<Storage<Scalar>, 2> referencePorosity_;
    LinearizationType linearizationType_;
};

namespace gpuistl
{

    template <class Scalar, template <class> class ContainerT, class TypeTagFrom, class TypeTagTo>
    FlowProblemBlackoilGpu<Scalar, TypeTagTo, ContainerT> copy_to_gpu(FlowProblemBlackoil<TypeTagFrom>& problem)
    {

        static_assert(std::is_same_v<std::vector<Scalar>, decltype(problem.rockCompressibilitiesRaw())>);
        static_assert(std::is_same_v<std::vector<unsigned short>, decltype(problem.rockTableIdx())>);

        return FlowProblemBlackoilGpu<Scalar, TypeTagTo, ContainerT>(
            ContainerT(problem.satnumRegionArray()),
            problem.model().linearizer().getLinearizationType(),
            ContainerT(problem.rockTableIdx()),
            ContainerT(problem.rockCompressibilitiesRaw()),
            ContainerT(problem.rockReferencePressuresRaw()),
            std::array<ContainerT<Scalar>, 2>{ContainerT(problem.referencePorosity()[0]), ContainerT(problem.referencePorosity()[1])}
        );
    }

    template <template <class> class ViewT, class TypeTag, template <class> class ContainerT, class Scalar>
    FlowProblemBlackoilGpu<Scalar, TypeTag, ViewT>
    make_view(FlowProblemBlackoilGpu<Scalar, TypeTag, ContainerT> problem)
    {
        return FlowProblemBlackoilGpu<Scalar, TypeTag, ViewT>(make_view<unsigned short>(problem.satnumRegionArray()),
                                                              problem.model().linearizer().getLinearizationType(),
                                                              make_view<unsigned short>(problem.rockTableIdx()),
                                                              make_view<Scalar>(problem.rockCompressibilitiesRaw()),
                                                              make_view<Scalar>(problem.rockReferencePressuresRaw()),
                                                              std::array<ViewT<Scalar>, 2>{make_view<Scalar>(problem.referencePorosity()[0]),
                                                                                                make_view<Scalar>(problem.referencePorosity()[1])}
                                                            );
    }

} // namespace gpuistl

} // namespace Opm

#endif // OPM_FLOW_PROBLEM_BLACKOILGPU_HPP
