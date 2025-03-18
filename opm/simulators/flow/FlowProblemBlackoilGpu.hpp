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

#include <opm/simulators/flow/FlowProblemBlackoil.hpp>

namespace Opm {

// This class is a simplified version of FlowProblem that should be GPU-instantiable
template< template<class> class Storage = VectorWithDefaultAllocator >
class FlowProblemBlackoilGpu {

    FlowProblemBlackoilGpu(Storage<unsigned short> satNum)
        : satNum_(satNum)
    {}

    unsigned short satnumRegionIndex(size_t elemIdx) const
    {
        if (satNum_.size() == 0){
            return 0;
        }

        return satNum_[elemIdx];
    }

private:
    Storage<unsigned short> satNum_;
};

namespace gpuistl {

    template< template<class> class ContainerT, class TypeTag>
    FlowProblemBlackoilGpu<ContainerT> copy_to_gpu(FlowProblemBlackoil<TypeTag> problem) {
        return FlowProblemBlackoilGpu<ContainerT>(ContainerT(problem.satNum_));
    }

    template< template<class> class ViewT, template<class> class ContainerT>
    FlowProblemBlackoilGpu<ViewT> make_view(FlowProblemBlackoilGpu<ContainerT> problem) {
        return FlowProblemBlackoilGpu<ViewT>(ViewT(problem.satNum_));
    }

}

}

#endif // OPM_FLOW_PROBLEM_BLACKOILGPU_HPP
