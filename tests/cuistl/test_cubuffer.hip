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

#define BOOST_TEST_MODULE TestCuBuffer

#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <dune/common/fvector.hh>
#include <dune/istl/bvector.hh>
#include <opm/simulators/linalg/cuistl/CuBuffer.hpp>
#include <opm/simulators/linalg/cuistl/CuView.hpp>
#include <opm/simulators/linalg/cuistl/detail/cuda_safe_call.hpp>
#include <random>
#include <array>
#include <algorithm>
#include <type_traits>

BOOST_AUTO_TEST_CASE(TestMakeView)
{
    auto buf = std::vector<int>({1, 2, 3, 4, 5, 6});
    auto gpubuf = ::Opm::cuistl::CuBuffer<int>(buf);
    auto gpuview = ::Opm::cuistl::CuView<int>(buf.data(), buf.size());
    bool gpuBufCreatedView = std::is_same<::Opm::cuistl::CuView<int>, decltype(gpuview)>::value;

    BOOST_CHECK(gpuBufCreatedView);

    auto gpuview2 = ::Opm::cuistl::make_view(gpubuf);
    bool gpuBufCreatedView2 = std::is_same<::Opm::cuistl::CuView<int>, decltype(gpuview2)>::value;
    BOOST_CHECK(gpuBufCreatedView2);
}
