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
#include <exception>
#include <opm/common/ErrorMacros.hpp>
#include <opm/simulators/linalg/cuistl/detail/cusparse_safe_call.hpp>

namespace Opm::gpuistl::detail
{

namespace
{
    template <class T>
    struct GpuSparseDeleteAndCreate {
    };

    template <>
    struct GpuSparseDeleteAndCreate<bsrilu02Info_t> {
        using DeleterType = typename GpuSparseResource<bsrilu02Info_t>::DeleterType;
        using CreatorType = typename GpuSparseResource<bsrilu02Info_t>::CreatorType;

        static DeleterType getDeleter()
        {
            return cusparseDestroyBsrilu02Info;
        }

        static CreatorType getCreator()
        {
            return cusparseCreateBsrilu02Info;
        }
    };

    template <>
    struct GpuSparseDeleteAndCreate<bsrsv2Info_t> {
        using DeleterType = typename GpuSparseResource<bsrsv2Info_t>::DeleterType;
        using CreatorType = typename GpuSparseResource<bsrsv2Info_t>::CreatorType;

        static DeleterType getDeleter()
        {
            return cusparseDestroyBsrsv2Info;
        }

        static CreatorType getCreator()
        {
            return cusparseCreateBsrsv2Info;
        }
    };

    template <>
    struct GpuSparseDeleteAndCreate<cusparseMatDescr_t> {
        using DeleterType = typename GpuSparseResource<cusparseMatDescr_t>::DeleterType;
        using CreatorType = typename GpuSparseResource<cusparseMatDescr_t>::CreatorType;

        static DeleterType getDeleter()
        {
            return cusparseDestroyMatDescr;
        }

        static CreatorType getCreator()
        {
            return cusparseCreateMatDescr;
        }
    };

} // namespace
template <class T>
GpuSparseResource<T>::GpuSparseResource(CreatorType creator, DeleterType deleter)
    : m_deleter(deleter)
{
    // TODO: This should probably not use this macro since it will disguise the
    // proper name of the function being called.
    OPM_GPUSPARSE_SAFE_CALL(creator(&m_resource));
}

template <class T>
GpuSparseResource<T>::GpuSparseResource()
    : GpuSparseResource<T>(GpuSparseDeleteAndCreate<T>::getCreator(), GpuSparseDeleteAndCreate<T>::getDeleter())
{
}

template <class T>
GpuSparseResource<T>::~GpuSparseResource()
{
    // TODO: This should probably not use this macro since it will disguise the
    // proper name of the function being called.
    OPM_GPUSPARSE_WARN_IF_ERROR(m_deleter(m_resource));
}
} // namespace Opm::gpuistl::detail
