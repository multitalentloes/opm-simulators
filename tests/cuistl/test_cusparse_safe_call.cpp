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
#include <config.h>

#define BOOST_TEST_MODULE TestCusparseSafeCall

#include <boost/test/unit_test.hpp>
#include <hipsparse.h>
#include <opm/simulators/linalg/cuistl/detail/cusparse_safe_call.hpp>

BOOST_AUTO_TEST_CASE(TestCreateHandle)
{
    hipsparseHandle_t cusparseHandle;
    BOOST_CHECK_NO_THROW(OPM_CUSPARSE_SAFE_CALL(hipsparseCreate(&cusparseHandle)););
}

BOOST_AUTO_TEST_CASE(TestThrows)
{
    std::vector<hipsparseStatus_t> errorCodes {{HIPSPARSE_STATUS_NOT_INITIALIZED,
                                               HIPSPARSE_STATUS_ALLOC_FAILED,
                                               HIPSPARSE_STATUS_INVALID_VALUE,
                                               HIPSPARSE_STATUS_ARCH_MISMATCH,
                                               HIPSPARSE_STATUS_MAPPING_ERROR,
                                               HIPSPARSE_STATUS_EXECUTION_FAILED,
                                               HIPSPARSE_STATUS_INTERNAL_ERROR,
                                               HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED,
                                               HIPSPARSE_STATUS_ZERO_PIVOT,
                                               HIPSPARSE_STATUS_NOT_SUPPORTED,
                                               HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES}};
    for (auto code : errorCodes) {
        BOOST_CHECK_THROW(OPM_CUSPARSE_SAFE_CALL(code), std::exception);
    }
}
