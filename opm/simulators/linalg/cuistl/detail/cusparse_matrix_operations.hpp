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
 * @brief This function receives a matrix, and the inverse of the matrix containing only its diagonal is stored in d_vec
 * @param mat pointer to GPU memory containing nonzerovalues of the sparse matrix
 * @param rowIndices Pointer to vector on GPU containing row indices compliant wiht bsr format
 * @param colIndices Pointer to vector on GPU containing col indices compliant wiht bsr format
 * @param numberOfRows Integer describing the number of rows in the matrix
 * @param blocksize Integer describing the sidelength of a block in the sparse matrix
 * @param[out] vec Pointer to the vector where the inverse of the diagonal matrix should be stored
 */
template <class T>
void invertDiagonalAndFlatten(T* mat, int* rowIndices, int* colIndices, size_t numberOfRows, size_t blocksize, T* vec);

/**
 * @brief Perform a lower solve on certain rows in a matrix that can safely be computed in parallel
 * @param reorderedMat pointer to GPU memory containing nonzerovalues of the sparse matrix. The matrix reordered such that rows in the same level sets are contiguous
 * @param rowIndices Pointer to vector on GPU containing row indices compliant wiht bsr format
 * @param colIndices Pointer to vector on GPU containing col indices compliant wiht bsr format
 * @param numberOfRows Integer describing the number of rows in the matrix
 * @param indexConversion Integer array containing mapping an index in the reordered matrix to its corresponding index in the natural ordered matrix
 * @param startIdx Index of the first row of the matrix to be solve
 * @param rowsInLevelSet Number of rows in this level set, which number the amount of rows solved in parallel by this function
 * @param dInv The diagonal matrix used by the Diagonal ILU preconditioner
 * @param d Stores the defect
 * @param [out] v Will store the results of the lower solve
 */
template <class T, int blocksize>
void computeLowerSolveLevelSet(T* reorderedMat, int* rowIndices, int* colIndices, size_t numberOfRows, int* indexConversion, const int startIdx, int rowsInLevelSet, T* dInv, const T* d, T* v);

// TODO: document this version when it is stable
template <class T, int blocksize>
void computeUpperSolveLevelSet(T* mat, int* rowIndices, int* colIndices, size_t numberOfRows, int* indexConversion, const int startIdx, int rowsInLevelSet, T* dInv, const T* d, T* v);

// TODO: document this version when it is stable
template <class T, int blocksize>
void computeDiluDiagonal(T* mat, int* rowIndices, int* colIndices, size_t numberOfRows, int* reorderedToNatural, int* naturalToReordered, const int startIdx, int rowsInLevelSet, T* dInv);

template <class T, int blocksize>
void moveMatDataToReordered(T* srcMatrix, int* srcRowIndices, int* srcColIndices, T* dstMatrix, int* dstRowIndices, int* dstColIndices, int* naturalToReordered, size_t numberOfRows);
} // namespace Opm::cuistl::detail
#endif
