/*
  Copyright 2022-2023 SINTEF AS

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
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <fmt/core.h>
#include <opm/simulators/linalg/cuistl/CuView.hpp>
#include <opm/simulators/linalg/cuistl/detail/cuda_safe_call.hpp>
#include <opm/simulators/linalg/cuistl/detail/vector_operations.hpp>
#include <opm/common/utility/gpuDecorators.hpp>

namespace Opm::cuistl
{

template <class T>
OPM_HOST_DEVICE CuView<T>::CuView(std::vector<T>& data)
    : CuView(data.data(), detail::to_int(data.size()))
{
}

template <class T>
OPM_HOST_DEVICE CuView<T>::CuView(T* dataOnHost, size_t numberOfElements)
    : m_dataPtr(dataOnHost), m_numberOfElements(numberOfElements)
{
}

template <class T>
OPM_HOST_DEVICE T&
CuView<T>::operator[](size_t idx)
{
#ifndef NDEBUG
    if (idx >= m_numberOfElements) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("The index provided was not in the range [0, buffersize-1]"));
    }
#endif
    return m_dataPtr[idx];
}

template <class T>
OPM_HOST_DEVICE T
CuView<T>::operator[](size_t idx) const
{
#ifndef NDEBUG
    if (idx >= m_numberOfElements) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("The index provided was not in the range [0, buffersize-1]"));
    }
#endif
    return m_dataPtr[idx];
}

template <class T>
OPM_HOST_DEVICE CuView<T>::CuView(const CuView<T>& other)
    : CuView(other.m_dataPtr, other.m_numberOfElements)
{
}

template <class T>
OPM_HOST_DEVICE CuView<T>::~CuView()
{
    OPM_CUDA_WARN_IF_ERROR(cudaFree(m_dataPtr));
}

template <typename T>
OPM_HOST_DEVICE const T*
CuView<T>::data() const
{
    return m_dataPtr;
}

template <typename T>
OPM_HOST_DEVICE size_t
CuView<T>::size() const
{
    // Note that there is no way for m_numberOfElements to be non-positive,
    // but for sanity we still use the safe conversion function here.
    //
    // We also doubt that this will lead to any performance penalty, but should this prove
    // to be false, this can be replaced by a simple cast to size_t
    return detail::to_size_t(m_numberOfElements);
}

template <typename T>
OPM_HOST_DEVICE std::vector<T>
CuView<T>::asStdVector() const
{
    std::vector<T> temporary(detail::to_size_t(m_numberOfElements));
    copyToHost(temporary);
    return temporary;
}

template <typename T>
OPM_HOST_DEVICE void
CuView<T>::setZeroAtIndexSet(const CuView<int>& indexSet)
{
    detail::setZeroAtIndexSet(m_dataPtr, indexSet.size(), indexSet.data());
}

template <typename T>
OPM_HOST_DEVICE void
CuView<T>::assertSameSize(const CuView<T>& x) const
{
    assertSameSize(x.m_numberOfElements);
}

template <typename T>
OPM_HOST_DEVICE void
CuView<T>::assertSameSize(size_t size) const
{
    if (size != m_numberOfElements) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Given vector has {}, while we have {}.", size, m_numberOfElements));
    }
}

template <typename T>
OPM_HOST_DEVICE void
CuView<T>::assertHasElements() const
{
    if (m_numberOfElements <= 0) {
        OPM_THROW(std::invalid_argument, "We have 0 elements");
    }
}

template <typename T>
OPM_HOST_DEVICE T*
CuView<T>::data()
{
    return m_dataPtr;
}

template <typename T>
OPM_HOST_DEVICE T&
CuView<T>::front()
{
#ifndef NDEBUG
    if (m_numberOfElements < 1) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Can not fetch the front item of a CuView with no elements"));
    }
#endif
    return m_dataPtr[0];
}

template <typename T>
OPM_HOST_DEVICE T
CuView<T>::front() const
{
#ifndef NDEBUG
    if (m_numberOfElements < 1) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Can not fetch the front item of a CuView with no elements"));
    }
#endif
    return m_dataPtr[0];
}

template <typename T>
OPM_HOST_DEVICE T&
CuView<T>::back()
{
#ifndef NDEBUG
    if (m_numberOfElements < 1) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Can not fetch the back item of a CuView with no elements"));
    }
#endif
    return m_dataPtr[m_numberOfElements-1];
}

template <typename T>
OPM_HOST_DEVICE T
CuView<T>::back() const
{
#ifndef NDEBUG
    if (m_numberOfElements < 1) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Can not fetch the back item of a CuView with no elements"));
    }
#endif
    return m_dataPtr[m_numberOfElements-1];
}

template <class T>
OPM_HOST_DEVICE void
CuView<T>::copyFromHost(const T* dataPointer, size_t numberOfElements)
{
    if (numberOfElements > size()) {
        OPM_THROW(std::runtime_error,
                  fmt::format("Requesting to copy too many elements. Vector has {} elements, while {} was requested.",
                              size(),
                              numberOfElements));
    }
    OPM_CUDA_SAFE_CALL(cudaMemcpy(data(), dataPointer, numberOfElements * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T>
OPM_HOST_DEVICE void
CuView<T>::copyToHost(T* dataPointer, size_t numberOfElements) const
{
    assertSameSize(detail::to_int(numberOfElements));
    OPM_CUDA_SAFE_CALL(cudaMemcpy(dataPointer, data(), numberOfElements * sizeof(T), cudaMemcpyDeviceToHost));
}

template <class T>
OPM_HOST_DEVICE void
CuView<T>::copyFromHost(const std::vector<T>& data)
{
    copyFromHost(data.data(), data.size());
}
template <class T>
OPM_HOST_DEVICE void
CuView<T>::copyToHost(std::vector<T>& data) const
{
    copyToHost(data.data(), data.size());
}

template <typename T>
OPM_HOST_DEVICE void
CuView<T>::prepareSendBuf(CuView<T>& buffer, const CuView<int>& indexSet) const
{
    return detail::prepareSendBuf(m_dataPtr, buffer.data(), indexSet.size(), indexSet.data());
}
template <typename T>
OPM_HOST_DEVICE void
CuView<T>::syncFromRecvBuf(CuView<T>& buffer, const CuView<int>& indexSet) const
{
    return detail::syncFromRecvBuf(m_dataPtr, buffer.data(), indexSet.size(), indexSet.data());
}

template class CuView<double>;
template class CuView<float>;
template class CuView<int>;

} // namespace Opm::cuistl
