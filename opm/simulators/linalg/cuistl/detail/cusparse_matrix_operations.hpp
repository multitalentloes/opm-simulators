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
#ifndef OPM_CUISTL_CUSPARSE_MATRIX_OPERATIONS_HPP
#define OPM_CUISTL_CUSPARSE_MATRIX_OPERATIONS_HPP
#include <cstddef>
#include <vector>
namespace Opm::cuistl::detail
{

/**
 * @brief setVectorValue sets every element of deviceData to value
 * @param deviceData pointer to GPU memory
 * @param numberOfElements number of elements to set to value
 * @param value the value to use
 */
template <class T>
void flatten(T* d_mat, int rowIndices[], int colIndices[], size_t numberOfElements, size_t blocksize, T* d_vec);

} // namespace Opm::cuistl::detail
#endif
