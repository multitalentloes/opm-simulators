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
#include <opm/simulators/linalg/cuistl/detail/cusparse_matrix_operations.hpp>
#include <vector>
namespace Opm::cuistl::detail
{

namespace
{
    template <class T> __global__ void cuflatten(T* d_mat, std::vector<int> rowIndices, std::vector<int> colIndices, size_t numberOfElements, size_t blocksize, T* d_vec)
    {
        const auto globalIndex = blockDim.x * blockIdx.x + threadIdx.x;
        if (globalIndex < numberOfElements){
            if (blocksize == 2){
                return;
            }
            else if(blocksize == 3){
                return;
            }
        }
    }

    constexpr inline size_t getThreads([[maybe_unused]] size_t numberOfElements)
    {
        return 1024;
    }

    inline size_t getBlocks(size_t numberOfElements)
    {
        const auto threads = getThreads(numberOfElements);
        return (numberOfElements + threads - 1) / threads;
    }
} // namespace

template <class T>
void
flatten(T* d_mat, std::vector<int> rowIndices, std::vector<int> colIndices, size_t numberOfElements, size_t blocksize, T* d_vec)
{
    cuflatten<<<getBlocks(numberOfElements), getThreads(numberOfElements)>>>(
        T* d_mat, vector<int> rowIndices, vector<int> colIndices, size_t numberOfElements, size_t blocksize, T* d_vec);
}

template void flatten(double*, int*, int*, size_t, size_t, double*);
template void flatten(float*, int*, int*, size_t, size_t, float*);

} // namespace Opm::cuistl::impl
