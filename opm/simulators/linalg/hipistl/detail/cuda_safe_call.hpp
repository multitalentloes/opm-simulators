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
#ifndef OPM_CUDA_SAFE_CALL_HPP_HIP
#define OPM_CUDA_SAFE_CALL_HPP_HIP
#include <hip/hip_runtime.h>
#include <fmt/core.h>
#include <opm/common/ErrorMacros.hpp>
#include <opm/common/OpmLog/OpmLog.hpp>
#include <string_view>

namespace Opm::hipistl::detail
{
/**
 * @brief getCudaErrorMessage generates the error message to display for a given error.
 *
 * @param error the error code from cublas
 * @param expression the expresison (say "hipMalloc(&pointer, 1)")
 * @param filename the code file the error occured in (typically __FILE__)
 * @param functionName name of the function the error occured in (typically __func__)
 * @param lineNumber the line number the error occured in (typically __LINE__)
 *
 * @todo Refactor to use std::source_location once we shift to C++20
 *
 * @return An error message to be displayed.
 *
 * @note This function is mostly for internal use.
 */
inline std::string
getCudaErrorMessage(hipError_t error,
                    const std::string_view& expression,
                    const std::string_view& filename,
                    const std::string_view& functionName,
                    size_t lineNumber)
{
    return fmt::format("CUDA expression did not execute correctly. Expression was: \n"
                       "    {}\n"
                       "CUDA error was {}\n"
                       "in function {}, in {}, at line {}\n",
                       expression,
                       hipGetErrorString(error),
                       functionName,
                       filename,
                       lineNumber);
}

/**
 * @brief cudaSafeCall checks the return type of the CUDA expression (function call) and throws an exception if it
 * does not equal hipSuccess.
 *
 * Example usage:
 * @code{.cpp}
 * #include <opm/simulators/linalg/hipistl/detail/cuda_safe_call.hpp>
 * #include <hip/hip_runtime.h>
 *
 * void some_function() {
 *     void* somePointer;
 *     cudaSafeCall(hipMalloc(&somePointer, 1), "hipMalloc(&somePointer, 1)", __FILE__, __func__, __LINE__);
 * }
 * @endcode
 *
 * @note It is probably easier to use the macro OPM_CUDA_SAFE_CALL
 *
 * @todo Refactor to use std::source_location once we shift to C++20
 */
inline void
cudaSafeCall(hipError_t error,
             const std::string_view& expression,
             const std::string_view& filename,
             const std::string_view& functionName,
             size_t lineNumber)
{
    if (error != hipSuccess) {
        OPM_THROW(std::runtime_error, getCudaErrorMessage(error, expression, filename, functionName, lineNumber));
    }
}

/**
 * @brief cudaWarnIfError checks the return type of the CUDA expression (function call) and issues a warning if it
 * does not equal hipSuccess.
 *
 * @param error the error code from cublas
 * @param expression the expresison (say "hipMalloc(&pointer, 1)")
 * @param filename the code file the error occured in (typically __FILE__)
 * @param functionName name of the function the error occured in (typically __func__)
 * @param lineNumber the line number the error occured in (typically __LINE__)
 *
 * @return the error sent in (for convenience).
 *
 * Example usage:
 * @code{.cpp}
 * #include <opm/simulators/linalg/hipistl/detail/cuda_safe_call.hpp>
 * #include <hip/hip_runtime.h>
 *
 * void some_function() {
 *     void* somePointer;
 *     cudaWarnIfError(hipMalloc(&somePointer, 1), "hipMalloc(&somePointer, 1)", __FILE__, __func__, __LINE__);
 * }
 * @endcode
 *
 * @note It is probably easier to use the macro OPM_CUDA_WARN_IF_ERROR
 *
 * @note Prefer the cudaSafeCall/OPM_CUDA_SAFE_CALL counterpart unless you really don't want to throw an exception.
 *
 * @todo Refactor to use std::source_location once we shift to C++20
 */
inline hipError_t
cudaWarnIfError(hipError_t error,
                const std::string_view& expression,
                const std::string_view& filename,
                const std::string_view& functionName,
                size_t lineNumber)
{
    if (error != hipSuccess) {
        OpmLog::warning(getCudaErrorMessage(error, expression, filename, functionName, lineNumber));
    }

    return error;
}
} // namespace Opm::hipistl::detail

/**
 * @brief OPM_CUDA_SAFE_CALL checks the return type of the CUDA expression (function call) and throws an exception if it
 * does not equal hipSuccess.
 *
 * Example usage:
 * @code{.cpp}
 * #include <opm/simulators/linalg/hipistl/detail/cuda_safe_call.hpp>
 * #include <hip/hip_runtime.h>
 *
 * void some_function() {
 *     void* somePointer;
 *     OPM_CUDA_SAFE_CALL(hipMalloc(&somePointer, 1));
 * }
 * @endcode
 *
 * @note This should be used for any call to the CUDA runtime API unless you have a good reason not to.
 */
#define OPM_CUDA_SAFE_CALL(expression)                                                                                 \
    ::Opm::hipistl::detail::cudaSafeCall(expression, #expression, __FILE__, __func__, __LINE__)


/**
 * @brief OPM_CUDA_WARN_IF_ERROR checks the return type of the CUDA expression (function call) and issues a warning if
 * it does not equal hipSuccess.
 *
 * Example usage:
 * @code{.cpp}
 * #include <opm/simulators/linalg/hipistl/detail/cuda_safe_call.hpp>
 * #include <hip/hip_runtime.h>
 *
 * void some_function() {
 *     void* somePointer;
 *     OPM_CUDA_WARN_IF_ERROR(hipMalloc(&somePointer, 1));
 * }
 * @endcode
 *
 * @note Prefer the cudaSafeCall/OPM_CUDA_SAFE_CALL counterpart unless you really don't want to throw an exception.
 */
#define OPM_CUDA_WARN_IF_ERROR(expression)                                                                             \
    ::Opm::hipistl::detail::cudaWarnIfError(expression, #expression, __FILE__, __func__, __LINE__)

#endif
