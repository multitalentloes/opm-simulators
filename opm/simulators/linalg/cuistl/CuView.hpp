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


namespace Opm::cuistl
{

/**
 * @brief The CuView class is a simple container class for the GPU.
 *
 *
 * Example usage:
 *
 * @code{.cpp}
 * #include <opm/simulators/linalg/cuistl/CuView.hpp>
 *
 * void someFunction() {
 *     auto someDataOnCPU = std::vector<double>({1.0, 2.0, 42.0, 59.9451743, 10.7132692});
 *
 *     auto dataOnGPU = CuView<double>(someDataOnCPU);
 *
 *     // Get data back on CPU in another vector:
 *     auto stdVectorOnCPU = dataOnGPU.asStdVector();
 * }
 *
 * @tparam T the type to store. Can be either float, double or int.
 */
template <typename T>
struct CuView
{
    T* m_dataPtr;
    size_t m_numberOfElements;
    /**
     * @brief CuView allocates new GPU memory of the same size as other and copies the content of the other vector to
     * this newly allocated memory.
     *
     * @note This does synchronous transfer.
     *
     * @param other the vector to copy from
     */
    CuView(const CuView<T>& other);

    /**
     * @brief CuView allocates new GPU memory of the same size as data and copies the content of the data vector to
     * this newly allocated memory.
     *
     * @note This does CPU to GPU transfer.
     * @note This does synchronous transfer.
     *
     * @note For now data.size() needs to be within the limits of int due to restrctions of CuBlas.
     *
     * @param data the vector to copy from
     */
    explicit CuView(std::vector<T>& data);
    /**
     * @brief Default constructor that will initialize cublas and allocate 0 bytes of memory
     */
    explicit CuView();

    /**
     * @brief operator[] to retrieve a reference to an item in the buffer
     *
     * @note This does asynchronous operations
     *
     * @param idx The index of the element
     */
    T& operator[](size_t idx);

    /**
     * @brief operator[] to retrieve a copy of an item in the buffer
     *
     * @note This does asynchronous operations
     *
     * @param idx The index of the element
     */
    T operator[](size_t idx) const;


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
    CuView(T* dataOnHost, size_t numberOfElements);

    /**
     * @brief ~CuView calls cudaFree
     */
    virtual ~CuView();

    /**
     * @return the raw pointer to the GPU data
     */
    T* data();

    /**
     * @return the raw pointer to the GPU data
     */
    const T* data() const;

    /**
     * @return fetch the first element in a CuView
     */
    T& front();

    /**
     * @return fetch the last element in a CuView
     */
    T& back();

    //TODO: do we need another version of front and back if the elements are to be const as well?
    /**
     * @return fetch the first element in a CuView
     */
    T front() const;

    /**
     * @return fetch the last element in a CuView
     */
    T back() const;

    /**
     * @brief copyFromHost copies data from a Dune::BlockVector
     * @param bvector the vector to copy from
     *
     * @note This does synchronous transfer.
     * @note This assumes that the size of this vector is equal to the size of the input vector.
     */
    template <int BlockDimension>
    void copyFromHost(const Dune::BlockVector<Dune::FieldVector<T, BlockDimension>>& bvector)
    {
        // TODO: [perf] vector.size() can be replaced by bvector.N() * BlockDimension
        if (detail::to_size_t(m_numberOfElements) != bvector.size()) {
            OPM_THROW(std::runtime_error,
                      fmt::format("Given incompatible vector size. CuView has size {}, \n"
                                  "however, BlockVector has N() = {}, and size = {}.",
                                  m_numberOfElements,
                                  bvector.N(),
                                  bvector.size()));
        }
        const auto dataPointer = static_cast<const T*>(&(bvector[0][0]));
        copyFromHost(dataPointer, m_numberOfElements);
    }

    /**
     * @brief copyToHost copies data to a Dune::BlockVector
     * @param bvector the vector to copy to
     *
     * @note This does synchronous transfer.
     * @note This assumes that the size of this vector is equal to the size of the input vector.
     */
    template <int BlockDimension>
    void copyToHost(Dune::BlockVector<Dune::FieldVector<T, BlockDimension>>& bvector) const
    {
        // TODO: [perf] vector.size() can be replaced by bvector.N() * BlockDimension
        if (detail::to_size_t(m_numberOfElements) != bvector.size()) {
            OPM_THROW(std::runtime_error,
                      fmt::format("Given incompatible vector size. CuView has size {},\n however, the BlockVector "
                                  "has has N() = {}, and size() = {}.",
                                  m_numberOfElements,
                                  bvector.N(),
                                  bvector.size()));
        }
        const auto dataPointer = static_cast<T*>(&(bvector[0][0]));
        copyToHost(dataPointer, m_numberOfElements);
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
    size_t size() const;

    /**
     * @brief creates an std::vector of the same size and copies the GPU data to this std::vector
     * @return an std::vector containing the elements copied from the GPU.
     */
    std::vector<T> asStdVector() const;

    /**
     * @brief creates an std::vector of the same size and copies the GPU data to this std::vector
     * @return an std::vector containing the elements copied from the GPU.
     */
    template <int blockSize>
    Dune::BlockVector<Dune::FieldVector<T, blockSize>> asDuneBlockVector() const
    {
        OPM_ERROR_IF(size() % blockSize != 0,
                     fmt::format("blockSize is not a multiple of size(). Given blockSize = {}, and size() = {}",
                                 blockSize,
                                 size()));

        Dune::BlockVector<Dune::FieldVector<T, blockSize>> returnValue(size() / blockSize);
        copyToHost(returnValue);
        return returnValue;
    }


    /**
     * @brief setZeroAtIndexSet for each element in indexSet, sets the index of this vector to be zero
     * @param indexSet the set of indices to set to zero
     *
     * @note Assumes all indices are within range
     *
     * This is supposed to do the same as the following code on the CPU:
     * @code{.cpp}
     * for (int index : indexSet) {
     *     this->data[index] = T(0.0);
     * }
     * @endcode
     */
    void setZeroAtIndexSet(const CuView<int>& indexSet);

    // Slow method that creates a string representation of a CuView for debug purposes
    std::string toDebugString()
    {
        std::vector<T> v = asStdVector();
        std::string res = "";
        for (T element : v){
            res += std::to_string(element) + " ";
        }
        res += std::to_string(v[v.size()-1]);
        return res;
    }

    class iterator {
    public:
        // Iterator typedefs
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = T*;
        using reference = T&;

        iterator(T* ptr) : m_ptr(ptr) {}

        // Dereference operator
        reference operator*() const {
            return *m_ptr;
        }

        // Pre-increment operator
        iterator& operator++() {
            ++m_ptr;
            return *this;
        }

        // Post-increment operator
        iterator operator++(int) {
            iterator tmp = *this;
            ++m_ptr;
            return tmp;
        }

        // Pre-decrement operator
        iterator& operator--() {
            --m_ptr;
            return *this;
        }

        // Post-decrement operator
        iterator operator--(int) {
            iterator tmp = *this;
            --m_ptr;
            return tmp;
        }

        // Inequality comparison operator
        bool operator!=(const iterator& other) const {
            return !(m_ptr == other.m_ptr);
        }

        // Subtraction operator
        difference_type operator-(const iterator& other) const {
            return std::distance(other.m_ptr, m_ptr);
        }
        iterator operator-(int n) const {
            return iterator(m_ptr-n);
        }

        // Addition operator
        iterator operator+(difference_type n) const {
            return iterator(m_ptr + n);
        }

        // Less than operator
        bool operator<(const iterator& other) const {
            return m_ptr < other.m_ptr;
        }

        // Greater than operator
        bool operator>(const iterator& other) const {
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
    iterator begin(){
        return iterator(m_dataPtr);
    }

    iterator begin() const {
        return iterator(m_dataPtr);
    }

    /**
     * @brief Get an iterator pointing to the address after the last element of the buffer
     * @param iterator pointing to the first value after the end of the buffer
     */
    iterator end(){
        return iterator(m_dataPtr + m_numberOfElements);
    }

    iterator end() const {
        return iterator(m_dataPtr + m_numberOfElements);
    }

    void assertSameSize(const CuView<T>& other) const;
    void assertSameSize(size_t size) const;

    void assertHasElements() const;
};

} // namespace Opm::cuistl
#endif
