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
#ifndef OPM_CUBLASHANDLE_HPP_HIP
#define OPM_CUBLASHANDLE_HPP_HIP
#include <hipblas/hipblas.h>
#include <memory>

namespace Opm::hipistl::detail
{

/**
 * @brief The CuBlasHandle class provides a singleton for the simulator universal cuBlasHandle.
 *
 * Example use:
 * @code{.cpp}
 * #include <opm/simulators/linalg/hipistl/detail/CuBlasHandle.hpp>
 * void someFunction() {
 *     auto& cublasHandle = ::Opm::hipistl::detail::CuBlasHandle::getInstance();
 *     int cuBlasVersion = -1;
 *     OPM_CUBLAS_SAFE_CALL(cublasGetVersion(cublasHandle.get(), &cuBlasVersion));
 * }
 * @endcode
 *
 */
class CuBlasHandle
{
public:
    // This should not be copyable.
    CuBlasHandle(const CuBlasHandle&) = delete;
    CuBlasHandle& operator=(const CuBlasHandle&) = delete;

    /**
     * Calls hipblasDestroy() on the handle
     */
    ~CuBlasHandle();

    /**
     * @brief get returns the underlying cuBlas handle (to be used in calls to cublas)
     */
    hipblasHandle_t get();

    /**
     * @brief getInstance creates (if necessary) and returns the single unique instance of CuBlasHandle (singleton)
     */
    static CuBlasHandle& getInstance();

private:
    CuBlasHandle();
    hipblasHandle_t m_handle;
};
} // namespace Opm::hipistl::detail
#endif // OPM_CUBLASHANDLE_HPP
