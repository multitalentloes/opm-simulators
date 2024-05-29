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
#include <opm/simulators/linalg/cuistl/CuBuffer.hpp>
#include <opm/simulators/linalg/cuistl/CuView.hpp>
#include <opm/simulators/linalg/cuistl/detail/cuda_safe_call.hpp>
#include <opm/simulators/linalg/cuistl/detail/vector_operations.hpp>

#include <config.h>
#include <opm/common/utility/gpuDecorators.hpp>
namespace Opm::cuistl
{

template <class T>
OPM_HOST_DEVICE CuBuffer<T>::CuBuffer(const std::vector<T>& data)
    : CuBuffer(data.data(), detail::to_int(data.size()))
{
}

template <class T>
OPM_HOST_DEVICE CuBuffer<T>::CuBuffer(const size_t numberOfElements)
    : m_numberOfElements(detail::to_int(numberOfElements))
{
    OPM_CUDA_SAFE_CALL(cudaMalloc(&m_dataOnDevice, sizeof(T) * detail::to_size_t(m_numberOfElements)));
}

template <class T>
OPM_HOST_DEVICE CuBuffer<T>::CuBuffer()
    : m_numberOfElements(0)
{
    OPM_CUDA_SAFE_CALL(cudaMalloc(&m_dataOnDevice, 0));
}

template <class T>
OPM_HOST_DEVICE CuBuffer<T>::CuBuffer(const T* dataOnHost, const size_t numberOfElements)
    : CuBuffer(numberOfElements)
{

    OPM_CUDA_SAFE_CALL(cudaMemcpy(
        m_dataOnDevice, dataOnHost, detail::to_size_t(m_numberOfElements) * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T>
OPM_HOST_DEVICE CuBuffer<T>::CuBuffer(const CuBuffer<T>& other)
    : CuBuffer(other.m_numberOfElements)
{
    assertHasElements();
    assertSameSize(other);
    OPM_CUDA_SAFE_CALL(cudaMemcpy(m_dataOnDevice,
                                  other.m_dataOnDevice,
                                  detail::to_size_t(m_numberOfElements) * sizeof(T),
                                  cudaMemcpyDeviceToDevice));
}

// template <class T>
// OPM_HOST_DEVICE CuBuffer<T>::CuBuffer(const CuBuffer<T>& other)
//     : m_dataOnDevice(other.data()), m_numberOfElements(other.m_numberOfElements)
// {
//     assertHasElements();
//     assertSameSize(other);
// }


template <class T>
CuBuffer<T>::CuBuffer(const CuBuffer<T>& other)
    : CuBuffer(other.m_numberOfElements)
{
    assertHasElements();
    assertSameSize(other);
    OPM_CUDA_SAFE_CALL(cudaMemcpy(m_dataOnDevice,
                                  other.m_dataOnDevice,
                                  detail::to_size_t(m_numberOfElements) * sizeof(T),
                                  cudaMemcpyDeviceToDevice));
}

template <class T>
OPM_HOST_DEVICE CuBuffer<T>::~CuBuffer()
{
    OPM_CUDA_WARN_IF_ERROR(cudaFree(m_dataOnDevice));
}

template <typename T>
OPM_HOST_DEVICE const T*
CuBuffer<T>::data() const
{
    return m_dataOnDevice;
}

template <typename T>
OPM_HOST_DEVICE typename CuBuffer<T>::size_type
CuBuffer<T>::size() const
{
    return detail::to_size_t(m_numberOfElements);
}

template <typename T>
OPM_HOST_DEVICE void
CuBuffer<T>::resize(int newSize)
{
    if (newSize < 0) {
        OPM_THROW(std::invalid_argument, "Setting a CuBuffer size to a negative number is not allowed");
    }
    // Allocate memory for the new buffer
    T* tmpBuffer = nullptr;
    OPM_CUDA_SAFE_CALL(cudaMalloc(&tmpBuffer, sizeof(T) * detail::to_size_t(newSize)));

    // Move the data from the old to the new buffer with truncation
    int sizeOfMove = std::min({m_numberOfElements, newSize});
    OPM_CUDA_SAFE_CALL(cudaMemcpy(tmpBuffer,
                                  m_dataOnDevice,
                                  detail::to_size_t(sizeOfMove) * sizeof(T),
                                  cudaMemcpyDeviceToDevice));

    // free the old buffer
    OPM_CUDA_SAFE_CALL(cudaFree(m_dataOnDevice));

    // swap the buffers
    m_dataOnDevice = tmpBuffer;

    // update size
    m_numberOfElements = newSize;
}

template <typename T>
OPM_HOST_DEVICE std::vector<T>
CuBuffer<T>::asStdVector() const
{
    std::vector<T> temporary(detail::to_size_t(m_numberOfElements));
    copyToHost(temporary);
    return temporary;
}

template <typename T>
void
CuBuffer<T>::assertSameSize(const CuBuffer<T>& x) const
{
    assertSameSize(x.m_numberOfElements);
}

template <typename T>
OPM_HOST_DEVICE void
CuBuffer<T>::assertSameSize(int size) const
{
    if (size != m_numberOfElements) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Given vector has {}, while we have {}.", size, m_numberOfElements));
    }
}

template <typename T>
OPM_HOST_DEVICE void
CuBuffer<T>::assertHasElements() const
{
    if (m_numberOfElements <= 0) {
        OPM_THROW(std::invalid_argument, "We have 0 elements");
    }
}

template <typename T>
OPM_HOST_DEVICE T*
CuBuffer<T>::data()
{
    return m_dataOnDevice;
}

template <typename T>
const T*
CuBuffer<T>::data() const
{
    return m_dataOnDevice;
}

template <class T>
OPM_HOST_DEVICE void
CuBuffer<T>::copyFromHost(const T* dataPointer, size_t numberOfElements)
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
CuBuffer<T>::copyToHost(T* dataPointer, size_t numberOfElements) const
{
    assertSameSize(detail::to_int(numberOfElements));
    OPM_CUDA_SAFE_CALL(cudaMemcpy(dataPointer, data(), numberOfElements * sizeof(T), cudaMemcpyDeviceToHost));
}

template <class T>
OPM_HOST_DEVICE void
CuBuffer<T>::copyFromHost(const std::vector<T>& data)
{
    copyFromHost(data.data(), data.size());
}
template <class T>
OPM_HOST_DEVICE void
CuBuffer<T>::copyToHost(std::vector<T>& data) const
{
    copyToHost(data.data(), data.size());
}

template class CuBuffer<double>;
template class CuBuffer<float>;
template class CuBuffer<int>;

template <class T>
CuView<T> make_view(CuBuffer<T>& buf) {
    return CuView<T>(buf.data(), buf.size());
}

template CuView<const double> make_view<const double, double>(const CuBuffer<double>&);
template CuView<const float> make_view<const float, float>(const CuBuffer<float>&);
template CuView<const int> make_view<const int, int>(const CuBuffer<int>&);

template <class T>
CuView<const T> make_view(const CuBuffer<T>& buf) {
    return CuView<const T>(buf.data(), buf.size());
}

template CuView<const double> make_view<double>(const CuBuffer<double>&);
template CuView<const float> make_view<float>(const CuBuffer<float>&);
template CuView<const int> make_view<int>(const CuBuffer<int>&);

template <class T>
CuView<T> make_view(CuBuffer<T>& buf) {
    return CuView<T>(buf.data(), buf.size());
}

template CuView<double> make_view<double>(CuBuffer<double>&);
template CuView<float> make_view<float>(CuBuffer<float>&);
template CuView<int> make_view<int>(CuBuffer<int>&);

} // namespace Opm::cuistl
