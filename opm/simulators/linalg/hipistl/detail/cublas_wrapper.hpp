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

/**
 * Contains wrappers to make the CuBLAS library behave as a modern C++ library with function overlading.
 *
 * In simple terms, this allows one to call say cublasScal on both double and single precisision,
 * instead of calling hipblasDscal and hipblasSscal respectively.
 */

#ifndef OPM_CUBLASWRAPPER_HEADER_INCLUDED
#define OPM_CUBLASWRAPPER_HEADER_INCLUDED
#include <hipblas.h>
#include <opm/common/ErrorMacros.hpp>

namespace Opm::hipistl::detail
{

inline hipblasStatus_t
cublasScal(hipblasHandle_t handle,
           int n,
           const double* alpha, /* host or device pointer */
           double* x,
           int incx)
{
    return hipblasDscal(handle,
                       n,
                       alpha, /* host or device pointer */
                       x,
                       incx);
}

inline hipblasStatus_t
cublasScal(hipblasHandle_t handle,
           int n,
           const float* alpha, /* host or device pointer */
           float* x,
           int incx)
{
    return hipblasSscal(handle,
                       n,
                       alpha, /* host or device pointer */
                       x,
                       incx);
}

inline hipblasStatus_t
cublasScal([[maybe_unused]] hipblasHandle_t handle,
           [[maybe_unused]] int n,
           [[maybe_unused]] const int* alpha, /* host or device pointer */
           [[maybe_unused]] int* x,
           [[maybe_unused]] int incx)
{
    OPM_THROW(std::runtime_error, "cublasScal multiplication for integer vectors is not implemented yet.");
}
inline hipblasStatus_t
cublasAxpy(hipblasHandle_t handle,
           int n,
           const double* alpha, /* host or device pointer */
           const double* x,
           int incx,
           double* y,
           int incy)
{
    return hipblasDaxpy(handle,
                       n,
                       alpha, /* host or device pointer */
                       x,
                       incx,
                       y,
                       incy);
}

inline hipblasStatus_t
cublasAxpy(hipblasHandle_t handle,
           int n,
           const float* alpha, /* host or device pointer */
           const float* x,
           int incx,
           float* y,
           int incy)
{
    return hipblasSaxpy(handle,
                       n,
                       alpha, /* host or device pointer */
                       x,
                       incx,
                       y,
                       incy);
}

inline hipblasStatus_t
cublasAxpy([[maybe_unused]] hipblasHandle_t handle,
           [[maybe_unused]] int n,
           [[maybe_unused]] const int* alpha, /* host or device pointer */
           [[maybe_unused]] const int* x,
           [[maybe_unused]] int incx,
           [[maybe_unused]] int* y,
           [[maybe_unused]] int incy)
{
    OPM_THROW(std::runtime_error, "axpy multiplication for integer vectors is not implemented yet.");
}

inline hipblasStatus_t
cublasDot(hipblasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result)
{
    return hipblasDdot(handle, n, x, incx, y, incy, result);
}

inline hipblasStatus_t
cublasDot(hipblasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result)
{
    return hipblasSdot(handle, n, x, incx, y, incy, result);
}

inline hipblasStatus_t
cublasDot([[maybe_unused]] hipblasHandle_t handle,
          [[maybe_unused]] int n,
          [[maybe_unused]] const int* x,
          [[maybe_unused]] int incx,
          [[maybe_unused]] const int* y,
          [[maybe_unused]] int incy,
          [[maybe_unused]] int* result)
{
    OPM_THROW(std::runtime_error, "inner product for integer vectors is not implemented yet.");
}

inline hipblasStatus_t
cublasNrm2(hipblasHandle_t handle, int n, const double* x, int incx, double* result)
{
    return hipblasDnrm2(handle, n, x, incx, result);
}


inline hipblasStatus_t
cublasNrm2(hipblasHandle_t handle, int n, const float* x, int incx, float* result)
{
    return hipblasSnrm2(handle, n, x, incx, result);
}

inline hipblasStatus_t
cublasNrm2([[maybe_unused]] hipblasHandle_t handle,
           [[maybe_unused]] int n,
           [[maybe_unused]] const int* x,
           [[maybe_unused]] int incx,
           [[maybe_unused]] int* result)
{
    OPM_THROW(std::runtime_error, "norm2 for integer vectors is not implemented yet.");
}

} // namespace Opm::hipistl::detail
#endif
