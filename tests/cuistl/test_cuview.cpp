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

#define BOOST_TEST_MODULE TestCuView

#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <dune/common/fvector.hh>
#include <dune/istl/bvector.hh>
#include <opm/simulators/linalg/cuistl/CuView.hpp>
#include <opm/simulators/linalg/cuistl/CuBuffer.hpp>
#include <random>
#include <array>
#include <algorithm>

//TODO: clang format this file when sketch is done

BOOST_AUTO_TEST_CASE(TestCreationAndIndexing)
{
    // A simple test to check that we can move data to and from the GPU
    auto cpubuffer = std::vector<double>({1.0, 2.0, 42.0, 59.9451743, 10.7132692});
    auto cubuffer = ::Opm::cuistl::CuBuffer<double>(cpubuffer);
    auto cuview = ::Opm::cuistl::CuView<double>(cubuffer.data(), cubuffer.size());

    auto stdVecOfCuView = cuview.asStdVector();

    BOOST_CHECK_EQUAL_COLLECTIONS(
        stdVecOfCuView.begin(), stdVecOfCuView.end(), cpubuffer.begin(), cpubuffer.end());
}
