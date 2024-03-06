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
#include <opm/simulators/linalg/hipistl/detail/CuSparseHandle.hpp>
#include <opm/simulators/linalg/hipistl/detail/cusparse_safe_call.hpp>
namespace Opm::hipistl::detail
{


CuSparseHandle::CuSparseHandle()
{
    OPM_CUSPARSE_SAFE_CALL(hipsparseCreate(&m_handle));
    OPM_CUSPARSE_SAFE_CALL(hipsparseSetStream(m_handle, 0));
}

CuSparseHandle::~CuSparseHandle()
{
    OPM_CUSPARSE_WARN_IF_ERROR(hipsparseDestroy(m_handle));
}

hipsparseHandle_t
CuSparseHandle::get()
{
    return m_handle;
}

CuSparseHandle&
CuSparseHandle::getInstance()
{
    static CuSparseHandle instance;
    return instance;
}

} // namespace Opm::hipistl::detail
