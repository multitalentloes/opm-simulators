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
#ifndef OPM_CUSPARSE_SAFE_CALL_HPP_HIP
#define OPM_CUSPARSE_SAFE_CALL_HPP_HIP
#include <hipsparse/hipsparse.h>
#include <exception>
#include <fmt/core.h>
#include <opm/common/ErrorMacros.hpp>
#include <opm/common/OpmLog/OpmLog.hpp>

namespace Opm::hipistl::detail
{

#define CHECK_CUSPARSE_ERROR_TYPE(code, x)                                                                             \
    if (code == x) {                                                                                                   \
        return #x;                                                                                                     \
    }
/**
 * @brief getCusparseErrorCodeToString Converts an error code returned from a cusparse function a human readable string.
 * @param code an error code from a cusparse routine
 * @return a human readable string.
 */
inline std::string
getCusparseErrorCodeToString(int code)
{
    CHECK_CUSPARSE_ERROR_TYPE(code, HIPSPARSE_STATUS_SUCCESS);
    CHECK_CUSPARSE_ERROR_TYPE(code, HIPSPARSE_STATUS_NOT_INITIALIZED);
    CHECK_CUSPARSE_ERROR_TYPE(code, HIPSPARSE_STATUS_ALLOC_FAILED);
    CHECK_CUSPARSE_ERROR_TYPE(code, HIPSPARSE_STATUS_INVALID_VALUE);
    CHECK_CUSPARSE_ERROR_TYPE(code, HIPSPARSE_STATUS_ARCH_MISMATCH);
    CHECK_CUSPARSE_ERROR_TYPE(code, HIPSPARSE_STATUS_MAPPING_ERROR);
    CHECK_CUSPARSE_ERROR_TYPE(code, HIPSPARSE_STATUS_EXECUTION_FAILED);
    CHECK_CUSPARSE_ERROR_TYPE(code, HIPSPARSE_STATUS_INTERNAL_ERROR);
    CHECK_CUSPARSE_ERROR_TYPE(code, HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    CHECK_CUSPARSE_ERROR_TYPE(code, HIPSPARSE_STATUS_ZERO_PIVOT);
    CHECK_CUSPARSE_ERROR_TYPE(code, HIPSPARSE_STATUS_NOT_SUPPORTED);
    CHECK_CUSPARSE_ERROR_TYPE(code, HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES);
    return fmt::format("UNKNOWN CUSPARSE ERROR {}.", code);
}

#undef CHECK_CUSPARSE_ERROR_TYPE
/**
 * @brief getCusparseErrorMessage generates the error message to display for a given error.
 *
 * @param error the error code from cublas
 * @param expression the expresison (say "hipsparseCreate(&handle)")
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
getCusparseErrorMessage(hipsparseStatus_t error,
                        const std::string_view& expression,
                        const std::string_view& filename,
                        const std::string_view& functionName,
                        size_t lineNumber)
{
    return fmt::format("cuSparse expression did not execute correctly. Expression was: \n\n"
                       "    {}\n\nin function {}, in {}, at line {}\n"
                       "CuSparse error code was: {}\n",
                       expression,
                       functionName,
                       filename,
                       lineNumber,
                       getCusparseErrorCodeToString(error));
}

/**
 * @brief cusparseSafeCall checks the return type of the CUSPARSE expression (function call) and throws an exception if
 * it does not equal HIPSPARSE_STATUS_SUCCESS.
 *
 * Example usage:
 * @code{.cpp}
 * #include <opm/simulators/linalg/hipistl/detail/cusparse_safe_call.hpp>
 * #include <hipblas/hipblas.h>
 *
 * void some_function() {
 *     hipsparseHandle_t cusparseHandle;
 *     cusparseSafeCall(hipsparseCreate(&cusparseHandle), "hipsparseCreate(&cusparseHandle)", __FILE__, __func__,
 * __LINE__);
 * }
 * @endcode
 *
 * @note It is probably easier to use the macro OPM_CUBLAS_SAFE_CALL
 *
 * @todo Refactor to use std::source_location once we shift to C++20
 */
inline void
cusparseSafeCall(hipsparseStatus_t error,
                 const std::string_view& expression,
                 const std::string_view& filename,
                 const std::string_view& functionName,
                 size_t lineNumber)
{
    if (error != HIPSPARSE_STATUS_SUCCESS) {
        OPM_THROW(std::runtime_error, getCusparseErrorMessage(error, expression, filename, functionName, lineNumber));
    }
}

/**
 * @brief cusparseWarnIfError checks the return type of the CUSPARSE expression (function call) and issues a warning if
 * it does not equal HIPSPARSE_STATUS_SUCCESS.
 *
 * @param error the error code from cublas
 * @param expression the expresison (say "hipblasCreate(&handle)")
 * @param filename the code file the error occured in (typically __FILE__)
 * @param functionName name of the function the error occured in (typically __func__)
 * @param lineNumber the line number the error occured in (typically __LINE__)
 *
 * @return the error sent in (for convenience).
 *
 * Example usage:
 * @code{.cpp}
 * #include <opm/simulators/linalg/hipistl/detail/cusparse_safe_call.hpp>
 * #include <hipblas/hipblas.h>
 *
 * void some_function() {
 *     hipsparseHandle_t cusparseHandle;
 *     cusparseWarnIfError(hipsparseCreate(&cusparseHandle), "hipsparseCreate(&cusparseHandle)", __FILE__, __func__,
 * __LINE__);
 * }
 * @endcode
 *
 * @note It is probably easier to use the macro OPM_CUSPARSE_WARN_IF_ERROR
 * @note Prefer the cusparseSafeCall/OPM_CUSPARSE_SAFE_CALL counterpart unless you really don't want to throw an
 * exception.
 * @todo Refactor to use std::source_location once we shift to C++20
 */
inline hipsparseStatus_t
cusparseWarnIfError(hipsparseStatus_t error,
                    const std::string_view& expression,
                    const std::string_view& filename,
                    const std::string_view& functionName,
                    size_t lineNumber)
{
    if (error != HIPSPARSE_STATUS_SUCCESS) {
        OpmLog::warning(getCusparseErrorMessage(error, expression, filename, functionName, lineNumber));
    }

    return error;
}
} // namespace Opm::hipistl::detail



/**
 * @brief OPM_HIPSPARSE_SAFE_CALL checks the return type of the cusparse expression (function call) and throws an
 * exception if it does not equal HIPSPARSE_STATUS_SUCCESS.
 *
 * Example usage:
 * @code{.cpp}
 * #include <opm/simulators/linalg/hipistl/detail/cusparse_safe_call.hpp>
 * #include <hipsparse/hipsparse.h>
 *
 * void some_function() {
 *     hipsparseHandle_t cusparseHandle;
 *     OPM_HIPSPARSE_SAFE_CALL(hipsparseCreate(&cusparseHandle));
 * }
 * @endcode
 *
 * @note This should be used for any call to cuSparse unless you have a good reason not to.
 */
#define OPM_HIPSPARSE_SAFE_CALL(expression)                                                                             \
    ::Opm::hipistl::detail::cusparseSafeCall(expression, #expression, __FILE__, __func__, __LINE__)

/**
 * @brief OPM_HIPSPARSE_WARN_IF_ERROR checks the return type of the cusparse expression (function call) and issues a
 * warning if it does not equal HIPSPARSE_STATUS_SUCCESS.
 *
 * Example usage:
 * @code{.cpp}
 * #include <opm/simulators/linalg/hipistl/detail/cusparse_safe_call.hpp>
 * #include <hipsparse/hipsparse.h>
 *
 * void some_function() {
 *     hipsparseHandle_t cusparseHandle;
 *     OPM_HIPSPARSE_WARN_IF_ERROR(hipsparseCreate(&cusparseHandle));
 * }
 * @endcode
 *
 * @note Prefer the cusparseSafeCall/OPM_CUBLAS_SAFE_CALL counterpart unless you really don't want to throw an
 * exception.
 */
#define OPM_HIPSPARSE_WARN_IF_ERROR(expression)                                                                         \
    ::Opm::hipistl::detail::cusparseWarnIfError(expression, #expression, __FILE__, __func__, __LINE__)
#endif // OPM_HIPSPARSE_SAFE_CALL_HPP
