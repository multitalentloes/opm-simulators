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
#ifndef OPM_DUALBUFFER_HEADER_HPP
#define OPM_DUALBUFFER_HEADER_HPP
#include <dune/common/fvector.hh>
#include <dune/istl/bvector.hh>
#include <exception>
#include <fmt/core.h>
#include <opm/common/ErrorMacros.hpp>
#include <opm/material/common/EnsureFinalized.hpp>
#include <opm/simulators/linalg/gpuistl/detail/safe_conversion.hpp>
#include <opm/simulators/linalg/gpuistl/GpuView.hpp>
#include <opm/simulators/linalg/gpuistl/detail/gpu_safe_call.hpp>
#include <opm/simulators/linalg/gpuistl/gpu_smart_pointer.hpp>
#include <opm/simulators/linalg/gpuistl/GpuBuffer.hpp>
#include <opm/simulators/linalg/gpuistl/GpuView.hpp>
#include <opm/common/utility/gpuDecorators.hpp>
#include <vector>
#include <string>
#include <cuda_runtime.h>


namespace Opm
{
namespace gpuistl
{
// This class is motivated by FlowProblemBlackoilGpu, where we need to
// have a buffer of material law parameters on the GPU, which themselves contain buffers.
// This cannot be done in the regular way because we need to iterate over each element and make_view
// on them, and the GPUBuffer does not support accessing elements from the CPU with an index.
// This class then contains both the CPU and GPU buffer on the CPU such that the values will
// be accessible so that we can call make_view on them. Additionally, this leaves us with a buffer
// of views that will not go out of scope so the outer view of the matlawparams will not go out of scope
// T may be EclTwoPhaseMaterialParams or similar
template<class T>
class DualBuffer : public EnsureFinalized {
public:
    using field_type = T;
    using ViewTypeT = typename ViewType<T>::type; // her må vi ha definert en ViewType<T> for de klassene vi trenger
    using GPUTypeT = typename GPUType<T>::type;

    // This copy constructor looks a bit suspicious
    // the GPUTypeT are not trivially copyable, so will the member depend on the outside vector still being alive?
    DualBuffer(std::vector<GPUTypeT>& cpuBuffer_)
        : m_cpuBuffer(cpuBuffer_)
    {
        // TODO: make the Buffer of Views on the GPU such that we can make a view of this type
    }

    DualBuffer(DualBuffer<T>& other)
        : m_cpuBuffer(other.m_cpuBuffer)
    {
        // m_gpuBuffer = other.m_gpuBuffer;
    }

    size_t size() const
    {
        return m_cpuBuffer.size();
    }

    // This function makes the views on the GPU that are stored in a buffer
    // After this function is called, the make_view function can be run in O(1) for this class
    //template<typename... Args>
    template<class MakeViewArgs>
    void prepareViews(MakeViewArgs makeView)
    {
        std::vector<ViewTypeT> views;
        for (auto& buf_item : m_cpuBuffer) {
            views.push_back(makeView(buf_item));
        }
        m_gpuBuffer = GpuBuffer<ViewTypeT>(views);

        EnsureFinalized::finalize();
    }

    GpuBuffer<ViewTypeT>& getGpuBuffer() {
        return m_gpuBuffer;
    }

    const std::vector<GPUTypeT>& getCPUBuffer() const {
        return m_cpuBuffer;
    }

    void ensureFinalized() const
    {
        EnsureFinalized::check();
    }

private:
    std::vector<GPUTypeT> m_cpuBuffer;
    GpuBuffer<ViewTypeT> m_gpuBuffer;
};

template<typename T>
GpuView<typename DualBuffer<T>::ViewTypeT> // return a view of the viewtype of the thing the dual buffer stores
make_view(DualBuffer<T>& dualBuffer)
{
    dualBuffer.ensureFinalized();
    return make_view(dualBuffer.getGpuBuffer());
}

} // namespace gpuistl
} // namespace Opm
#endif
