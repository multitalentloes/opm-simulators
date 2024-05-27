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
#ifndef OPM_CUVIEW_HEADER_HPP
#define OPM_CUVIEW_HEADER_HPP
#include <dune/common/fvector.hh>
#include <dune/istl/bvector.hh>
#include <exception>
#include <fmt/core.h>
#include <opm/common/ErrorMacros.hpp>
#include <opm/simulators/linalg/cuistl/detail/safe_conversion.hpp>
#include <vector>
#include <string>
#include <opm/common/utility/gpuDecorators.hpp>


namespace Opm::cuistl
{

/**
 * @brief The CuView class is provides a view of some data allocated on the GPU
 *
 *  This class supports being used from inside a CUDA/HIP Kernel.
 *  Implementations being placed in separate files will cause large overhead on GPUS
 *  For this reason we have the implementation in the header and avoid using
 *  Relocated Device Code (compiling with -rdc)
 *
 **/
template <typename T>
struct CuView
{
    T* m_dataPtr;
    size_t m_numberOfElements;

    /**
     * @brief Default constructor that will initialize cublas and allocate 0 bytes of memory
     */
    OPM_HOST_DEVICE explicit CuView() = default;

    CuView(std::vector<T>& data);

    /**
     * @brief operator[] to retrieve a reference to an item in the buffer
     *
     * @note This does asynchronous operations
     *
     * @param idx The index of the element
     */
    OPM_HOST_DEVICE T& operator[](size_t idx){
#ifndef NDEBUG
        if (idx >= m_numberOfElements) {
            OPM_THROW(std::invalid_argument,
                    fmt::format("The index provided was not in the range [0, buffersize-1]"));
        }
#endif
    return m_dataPtr[idx];
    }

    /**
     * @brief operator[] to retrieve a copy of an item in the buffer
     *
     * @note This does asynchronous operations
     *
     * @param idx The index of the element
     */
    OPM_HOST_DEVICE T operator[](size_t idx) const {
#ifndef NDEBUG
        if (idx >= m_numberOfElements) {
            OPM_THROW(std::invalid_argument,
                    fmt::format("The index provided was not in the range [0, buffersize-1]"));
        }
#endif
    return m_dataPtr[idx];
    }


    /**
     * @brief CuView allocates new GPU memory of size numberOfElements * sizeof(T) and copies numberOfElements from
     * data
     *
     * @note This assumes the data is on the CPU.
     *
     * @param numberOfElements number of T elements to allocate
     * @param dataOnHost data on host/CPU
     *
     * @note For now numberOfElements needs to be within the limits of int due to restrictions in cublas
     */
    OPM_HOST_DEVICE CuView(T* dataOnHost, size_t numberOfElements)
        : m_dataPtr(dataOnHost), m_numberOfElements(numberOfElements)
    {
    }

    /**
     * @brief ~CuView calls cudaFree
     */
    OPM_HOST_DEVICE ~CuView() = default;

    /**
     * @return the raw pointer to the GPU data
     */
    OPM_HOST_DEVICE T* data(){
        return m_dataPtr;
    }

    /**
     * @return the raw pointer to the GPU data
     */
    OPM_HOST_DEVICE const T* data() const{
        return m_dataPtr;
    }

    /**
     * @return fetch the first element in a CuView
     */
    OPM_HOST_DEVICE T& front()
    {
#ifndef NDEBUG
    if (m_numberOfElements < 1) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Can not fetch the front item of a CuView with no elements"));
    }
#endif
    return m_dataPtr[0];
    }

    /**
     * @return fetch the last element in a CuView
     */
    OPM_HOST_DEVICE T& back()
    {
#ifndef NDEBUG
    if (m_numberOfElements < 1) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Can not fetch the back item of a CuView with no elements"));
    }
#endif
    return m_dataPtr[m_numberOfElements-1];
    }

    //TODO: do we need another version of front and back if the elements are to be const as well?
    /**
     * @return fetch the first element in a CuView
     */
    OPM_HOST_DEVICE T front() const
    {
#ifndef NDEBUG
        if (m_numberOfElements < 1) {
            OPM_THROW(std::invalid_argument,
                    fmt::format("Can not fetch the front item of a CuView with no elements"));
        }
#endif
    return m_dataPtr[0];
    }

    /**
     * @return fetch the last element in a CuView
     */
    OPM_HOST_DEVICE T back() const
    {
#ifndef NDEBUG
    if (m_numberOfElements < 1) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Can not fetch the back item of a CuView with no elements"));
    }
#endif
    return m_dataPtr[m_numberOfElements-1];
    }

    /**
     * @brief copyFromHost copies numberOfElements from the CPU memory dataPointer
     * @param dataPointer raw pointer to CPU memory
     * @param numberOfElements number of elements to copy
     * @note This does synchronous transfer.
     * @note assumes that this vector has numberOfElements elements
     */
    void copyFromHost(const T* dataPointer, size_t numberOfElements);

    /**
     * @brief copyFromHost copies numberOfElements to the CPU memory dataPointer
     * @param dataPointer raw pointer to CPU memory
     * @param numberOfElements number of elements to copy
     * @note This does synchronous transfer.
     * @note assumes that this vector has numberOfElements elements
     */
    void copyToHost(T* dataPointer, size_t numberOfElements) const;

    /**
     * @brief copyToHost copies data from an std::vector
     * @param data the vector to copy from
     *
     * @note This does synchronous transfer.
     * @note This assumes that the size of this vector is equal to the size of the input vector.
     */
    void copyFromHost(const std::vector<T>& data);

    /**
     * @brief copyToHost copies data to an std::vector
     * @param data the vector to copy to
     *
     * @note This does synchronous transfer.
     * @note This assumes that the size of this vector is equal to the size of the input vector.
     */
    void copyToHost(std::vector<T>& data) const;

    void prepareSendBuf(CuView<T>& buffer, const CuView<int>& indexSet) const;
    void syncFromRecvBuf(CuView<T>& buffer, const CuView<int>& indexSet) const;

    /**
     * @brief size returns the size (number of T elements) in the vector
     * @return number of elements
     */
    OPM_HOST_DEVICE size_t size() const{
        return detail::to_size_t(m_numberOfElements);
    }

    /**
     * @brief creates an std::vector of the same size and copies the GPU data to this std::vector
     * @return an std::vector containing the elements copied from the GPU.
     */
    std::vector<T> asStdVector() const;

    class iterator {
    public:
        // Iterator typedefs
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = T*;
        using reference = T&;

        OPM_HOST_DEVICE iterator(T* ptr) : m_ptr(ptr) {}

        // Dereference operator
        OPM_HOST_DEVICE reference operator*() const {
            return *m_ptr;
        }

        // Pre-increment operator
        OPM_HOST_DEVICE iterator& operator++() {
            ++m_ptr;
            return *this;
        }

        // Post-increment operator
        OPM_HOST_DEVICE iterator operator++(int) {
            iterator tmp = *this;
            ++m_ptr;
            return tmp;
        }

        // Pre-decrement operator
        OPM_HOST_DEVICE iterator& operator--() {
            --m_ptr;
            return *this;
        }

        // Post-decrement operator
        OPM_HOST_DEVICE iterator operator--(int) {
            iterator tmp = *this;
            --m_ptr;
            return tmp;
        }

        // Inequality comparison operator
        OPM_HOST_DEVICE bool operator!=(const iterator& other) const {
            return !(*m_ptr == *other.m_ptr);
        }
        OPM_HOST_DEVICE bool operator==(const iterator& other) const {
            return *m_ptr == *other.m_ptr;
        }

        // Subtraction operator
        OPM_HOST_DEVICE difference_type operator-(const iterator& other) const {
            return std::distance(other.m_ptr, m_ptr);
        }
        OPM_HOST_DEVICE iterator operator-(int n) const {
            return iterator(m_ptr-n);
        }

        // Addition operator
        OPM_HOST_DEVICE iterator operator+(difference_type n) const {
            return iterator(m_ptr + n);
        }

        // Less than operator
        OPM_HOST_DEVICE bool operator<(const iterator& other) const {
            return m_ptr < other.m_ptr;
        }

        // Greater than operator
        OPM_HOST_DEVICE bool operator>(const iterator& other) const {
            return m_ptr > other.m_ptr;
        }

    private:
        // Pointer to the current element
        T* m_ptr;
    };

    /**
     * @brief Get an iterator pointing to the first element of the buffer
     * @param iterator to traverse the buffer
     */
    OPM_HOST_DEVICE iterator begin(){
        return iterator(m_dataPtr);
    }

    OPM_HOST_DEVICE iterator begin() const {
        return iterator(m_dataPtr);
    }

    /**
     * @brief Get an iterator pointing to the address after the last element of the buffer
     * @param iterator pointing to the first value after the end of the buffer
     */
    OPM_HOST_DEVICE iterator end(){
        return iterator(m_dataPtr + m_numberOfElements);
    }

    OPM_HOST_DEVICE iterator end() const {
        return iterator(m_dataPtr + m_numberOfElements);
    }

    OPM_HOST_DEVICE void assertSameSize(const CuView<T>& other) const
    {
        assertSameSize(other.m_numberOfElements);
    }
    OPM_HOST_DEVICE void assertSameSize(size_t size) const
    {
        if (size != m_numberOfElements) {
            OPM_THROW(std::invalid_argument,
                    fmt::format("Given vector has {}, while we have {}.", size, m_numberOfElements));
        }
    }

    OPM_HOST_DEVICE void assertHasElements() const
    {
        if (m_numberOfElements <= 0) {
            OPM_THROW(std::invalid_argument, "We have 0 elements");
        }
    }
};

} // namespace Opm::cuistl
#endif
