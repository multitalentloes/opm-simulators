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
#include <opm/common/ErrorMacros.hpp>
#include <opm/simulators/linalg/cuistl/detail/cusparse_matrix_operations.hpp>
#include <stdexcept>
namespace Opm::cuistl::detail
{
namespace
{

    // TODO: figure out if this can be generalized effectively, this seems excessively verbose
    // explicit formulas based on Dune cpu code
    template <class T, int blocksize>
    __device__ __forceinline__ void invBlockOutOfPlace(const T* __restrict__ srcBlock, T* __restrict__ dstBlock)
    {
        if (blocksize == 1) {
            dstBlock[0] = 1.0 / (srcBlock[0]);
        } else if (blocksize == 2) {
            T detInv = 1.0 / (srcBlock[0] * srcBlock[3] - srcBlock[1] * srcBlock[2]);
            dstBlock[0] = srcBlock[3] * detInv;
            dstBlock[1] = -srcBlock[1] * detInv;
            dstBlock[2] = -srcBlock[2] * detInv;
            dstBlock[3] = srcBlock[0] * detInv;
        } else if (blocksize == 3) {
            // based on Dune implementation
            T t4 = srcBlock[0] * srcBlock[4];
            T t6 = srcBlock[0] * srcBlock[5];
            T t8 = srcBlock[1] * srcBlock[3];
            T t10 = srcBlock[2] * srcBlock[3];
            T t12 = srcBlock[1] * srcBlock[6];
            T t14 = srcBlock[2] * srcBlock[6];

            T t17 = 1.0
                / (t4 * srcBlock[8] - t6 * srcBlock[7] - t8 * srcBlock[8] + t10 * srcBlock[7] + t12 * srcBlock[5]
                   - t14 * srcBlock[4]); // t17 is 1/determinant

            dstBlock[0] = (srcBlock[4] * srcBlock[8] - srcBlock[5] * srcBlock[7]) * t17;
            dstBlock[1] = -(srcBlock[1] * srcBlock[8] - srcBlock[2] * srcBlock[7]) * t17;
            dstBlock[2] = (srcBlock[1] * srcBlock[5] - srcBlock[2] * srcBlock[4]) * t17;
            dstBlock[3] = -(srcBlock[3] * srcBlock[8] - srcBlock[5] * srcBlock[6]) * t17;
            dstBlock[4] = (srcBlock[0] * srcBlock[8] - t14) * t17;
            dstBlock[5] = -(t6 - t10) * t17;
            dstBlock[6] = (srcBlock[3] * srcBlock[7] - srcBlock[4] * srcBlock[6]) * t17;
            dstBlock[7] = -(srcBlock[0] * srcBlock[7] - t12) * t17;
            dstBlock[8] = (t4 - t8) * t17;
        }
    }

    // explicit formulas based on Dune cpu code
    template <class T, int blocksize>
    __device__ __forceinline__ void invBlockInPlace(T* __restrict__ block)
    {
        if (blocksize == 1) {
            block[0] = 1.0 / (block[0]);
        } else if (blocksize == 2) {
            T detInv = 1.0 / (block[0] * block[3] - block[1] * block[2]);

            T temp = block[0];
            block[0] = block[3] * detInv;
            block[1] = -block[1] * detInv;
            block[2] = -block[2] * detInv;
            block[3] = temp * detInv;
        } else if (blocksize == 3) {
            T t4 = block[0] * block[4];
            T t6 = block[0] * block[5];
            T t8 = block[1] * block[3];
            T t10 = block[2] * block[3];
            T t12 = block[1] * block[6];
            T t14 = block[2] * block[6];

            T det = (t4 * block[8] - t6 * block[7] - t8 * block[8] + t10 * block[7] + t12 * block[5] - t14 * block[4]);
            T t17 = T(1.0) / det;

            T matrix01 = block[1];
            T matrix00 = block[0];
            T matrix10 = block[3];
            T matrix11 = block[4];

            block[0] = (block[4] * block[8] - block[5] * block[7]) * t17;
            block[1] = -(block[1] * block[8] - block[2] * block[7]) * t17;
            block[2] = (matrix01 * block[5] - block[2] * block[4]) * t17;
            block[3] = -(block[3] * block[8] - block[5] * block[6]) * t17;
            block[4] = (matrix00 * block[8] - t14) * t17;
            block[5] = -(t6 - t10) * t17;
            block[6] = (matrix10 * block[7] - matrix11 * block[6]) * t17;
            block[7] = -(matrix00 * block[7] - t12) * t17;
            block[8] = (t4 - t8) * t17;
        }
    }

    enum class MVType { SET, PLUS, MINUS };
    // SET:   c  = A*b
    // PLS:   c += A*b
    // MINUS: c -= A*b
    template <class T, int blocksize, MVType OP>
    __device__ __forceinline__ void mv(T* A, T* b, T* c)
    {
        for (int i = 0; i < blocksize; ++i) {
            if (OP == MVType::SET) {
                c[i] = 0;
            }

            for (int j = 0; j < blocksize; ++j) {
                if (OP == MVType::SET || OP == MVType::PLUS) {
                    c[i] += A[i * blocksize + j] * b[j];
                } else if (OP == MVType::MINUS) {
                    c[i] -= A[i * blocksize + j] * b[j];
                }
            }
        }
    }

    // TODO: verify if this is dumb
    // The intention is to allow for more optimization by having a function that does
    // two consecutive small matrix products in a single functions call
    // dst = A*B*C
    template <class T, int blocksize>
    __device__ __forceinline__ void mmx2Subtraction(T* A, T* B, T* C, T* dst)
    {

        T tmp[blocksize * blocksize] = {0};
        // tmp = A*B
        for (int i = 0; i < blocksize; ++i) {
            for (int k = 0; k < blocksize; ++k) {
                for (int j = 0; j < blocksize; ++j) {
                    tmp[i * blocksize + j] += A[i * blocksize + k] * B[k * blocksize + j];
                }
            }
        }

        // dst = tmp*C
        for (int i = 0; i < blocksize; ++i) {
            for (int k = 0; k < blocksize; ++k) {
                for (int j = 0; j < blocksize; ++j) {
                    dst[i * blocksize + j] -= tmp[i * blocksize + k] * C[k * blocksize + j];
                }
            }
        }
    }

    template <class T, int blocksize>
    __global__ void
    cuInvertDiagonalAndFlatten(T* matNonZeroValues, int* rowIndices, int* colIndices, size_t numberOfRows, T* vec)
    {
        const auto row = blockDim.x * blockIdx.x + threadIdx.x;

        if (row < numberOfRows) {
            size_t nnzIdx = rowIndices[row];
            size_t nnzIdxLim = rowIndices[row + 1];

            // this loop will cause some extra checks that we are within the limit in the case of the diagonal having a
            // zero element
            while (colIndices[nnzIdx] != row && nnzIdx <= nnzIdxLim) {
                ++nnzIdx;
            }

            // diagBlock points to the start of where the diagonal block is stored
            T* diagBlock = &matNonZeroValues[blocksize * blocksize * nnzIdx];
            // vecBlock points to the start of the block element in the vector where the inverse of the diagonal block
            // element should be stored
            T* vecBlock = &vec[blocksize * blocksize * row];

            invBlockOutOfPlace<T, blocksize>(diagBlock, vecBlock);
        }
    }

    template <class T, int blocksize>
    __global__ void cuComputeLowerSolveLevelSet(T* mat,
                                                int* rowIndices,
                                                int* colIndices,
                                                size_t numberOfRows,
                                                int* indexConversion,
                                                const int startIdx,
                                                int rowsInLevelSet,
                                                T* dInv,
                                                const T* d,
                                                T* v)
    {
        const auto reorderedRowIdx = startIdx + (blockDim.x * blockIdx.x + threadIdx.x);
        if (reorderedRowIdx < rowsInLevelSet + startIdx) {

            size_t nnzIdx = rowIndices[reorderedRowIdx];
            int naturalRowIdx = indexConversion[reorderedRowIdx];

            T rhs[blocksize];
            for (int i = 0; i < blocksize; i++)
                rhs[i] = d[naturalRowIdx * blocksize + i];

            for (int block = nnzIdx; colIndices[block] < naturalRowIdx; ++block) {
                const int col = colIndices[block];
                mv<T, blocksize, MVType::MINUS>(&mat[block * blocksize * blocksize], &v[col * blocksize], rhs);
            }

            mv<T, blocksize, MVType::SET>(
                &dInv[reorderedRowIdx * blocksize * blocksize], rhs, &v[naturalRowIdx * blocksize]);
        }
    }

    template <class T, int blocksize>
    __global__ void cuComputeUpperSolveLevelSet(T* mat,
                                                int* rowIndices,
                                                int* colIndices,
                                                size_t numberOfRows,
                                                int* indexConversion,
                                                const int startIdx,
                                                int rowsInLevelSet,
                                                T* dInv,
                                                const T* d,
                                                T* v)
    {
        const auto reorderedRowIdx = startIdx + (blockDim.x * blockIdx.x + threadIdx.x);
        if (reorderedRowIdx < rowsInLevelSet + startIdx) {
            size_t nnzIdxLim = rowIndices[reorderedRowIdx + 1];
            int naturalRowIdx = indexConversion[reorderedRowIdx];

            T rhs[blocksize] = {0};

            for (int block = nnzIdxLim - 1; colIndices[block] > naturalRowIdx; --block) {
                const int col = colIndices[block];
                mv<T, blocksize, MVType::PLUS>(&mat[block * blocksize * blocksize], &v[col * blocksize], rhs);
            }

            mv<T, blocksize, MVType::MINUS>(
                &dInv[reorderedRowIdx * blocksize * blocksize], rhs, &v[naturalRowIdx * blocksize]);
        }
    }

    template <class T, int blocksize>
    __global__ void cuComputeDiluDiagonal(T* mat,
                                          int* rowIndices,
                                          int* colIndices,
                                          size_t numberOfRows,
                                          int* reorderedToNatural,
                                          int* naturalToReordered,
                                          const int startIdx,
                                          int rowsInLevelSet,
                                          T* dInv)
    {
        const auto reorderedRowIdx = startIdx + blockDim.x * blockIdx.x + threadIdx.x;
        if (reorderedRowIdx < rowsInLevelSet + startIdx) {
            int naturalRowIdx = reorderedToNatural[reorderedRowIdx];
            size_t nnzIdx = rowIndices[reorderedRowIdx];

            int diagIdx = nnzIdx;
            while (colIndices[diagIdx] != naturalRowIdx)
                ++diagIdx;

            T dInvTmp[blocksize * blocksize];
            for (int i = 0; i < blocksize; ++i) {
                for (int j = 0; j < blocksize; ++j) {
                    dInvTmp[i * blocksize + j] = mat[diagIdx * blocksize * blocksize + i * blocksize + j];
                    // dInvTmp[i*blocksize + j]  = dInv[reorderedRowIdx*blocksize*blocksize + i*blocksize + j];
                }
            }


            for (int block = nnzIdx; colIndices[block] < naturalRowIdx; ++block) {
                const int col = naturalToReordered[colIndices[block]];
                // find element with indices swapped
                // Binary search over block in the right row, [rowIndices[col], rowindices[col+1]-1] defines the range
                // we binary search over
                int left = rowIndices[col];
                int right = rowIndices[col + 1] - 1;
                int mid;

                while (left <= right) {
                    mid = left + (right - left) / 2; // overflow-safe average
                    const int col = colIndices[mid];

                    if (col == naturalRowIdx){
                        break;
                    }
                    else if (col < naturalRowIdx) {
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
                }

                int mirror = mid;

                mmx2Subtraction<T, blocksize>(&mat[block * blocksize * blocksize],
                                              &dInv[col * blocksize * blocksize],
                                              &mat[mirror * blocksize * blocksize],
                                              dInvTmp);
            }

            invBlockInPlace<T, blocksize>(dInvTmp);

            for (int i = 0; i < blocksize; ++i) {
                for (int j = 0; j < blocksize; ++j) {
                    dInv[reorderedRowIdx * blocksize * blocksize + i * blocksize + j] = dInvTmp[i * blocksize + j];
                }
            }
        }
    }

    // consider rewriting to use a warp per row instead of a thread
    // this might give much better memory throughput because of coalesced memory accesses
    template <class T, int blocksize>
    __global__ void cuMoveDataToReordered(T* srcMatrix,
                                          int* srcRowIndices,
                                          int* srcColIndices,
                                          T* dstMatrix,
                                          int* dstRowIndices,
                                          int* dstColIndices,
                                          int* indexConversion,
                                          size_t numberOfRows)
    {
        const auto srcRow = blockDim.x * blockIdx.x + threadIdx.x;
        if (srcRow < numberOfRows) {

            const auto dstRow = indexConversion[srcRow];

            for (int srcBlock = srcRowIndices[srcRow], dstBlock = dstRowIndices[dstRow];
                 srcBlock < srcRowIndices[srcRow + 1];
                 ++srcBlock, ++dstBlock) {
                for (int i = 0; i < blocksize; ++i) {
                    for (int j = 0; j < blocksize; ++j) {
                        dstMatrix[dstBlock * blocksize * blocksize + i * blocksize + j]
                            = srcMatrix[srcBlock * blocksize * blocksize + i * blocksize + j];
                    }
                }
            }
        }
    }

    constexpr inline size_t getThreads([[maybe_unused]] size_t numberOfRows)
    {
        return 1024;
    }

    inline size_t getBlocks(size_t numberOfRows)
    {
        const auto threads = getThreads(numberOfRows);
        return (numberOfRows + threads - 1) / threads;
    }
} // namespace

template <class T>
void
invertDiagonalAndFlatten(T* mat, int* rowIndices, int* colIndices, size_t numberOfRows, size_t blocksize, T* vec)
{
    switch (blocksize) {
    case 1:
        cuInvertDiagonalAndFlatten<T, 1>
            <<<getBlocks(numberOfRows), getThreads(numberOfRows)>>>(mat, rowIndices, colIndices, numberOfRows, vec);
        break;
    case 2:
        cuInvertDiagonalAndFlatten<T, 2>
            <<<getBlocks(numberOfRows), getThreads(numberOfRows)>>>(mat, rowIndices, colIndices, numberOfRows, vec);
        break;
    case 3:
        cuInvertDiagonalAndFlatten<T, 3>
            <<<getBlocks(numberOfRows), getThreads(numberOfRows)>>>(mat, rowIndices, colIndices, numberOfRows, vec);
        break;
    default:
        // TODO: Figure out what is why it did not produce an error or any output in the output stream or the DBG file
        // when I forced this case to execute
        OPM_THROW(std::invalid_argument, "Inverting diagonal is not implemented for blocksizes > 3");
        break;
    }
}

template void invertDiagonalAndFlatten(double*, int*, int*, size_t, size_t, double*);
template void invertDiagonalAndFlatten(float*, int*, int*, size_t, size_t, float*);

// perform the lower solve for all rows in the same level set
template <class T, int blocksize>
void
computeLowerSolveLevelSet(T* mat,
                          int* rowIndices,
                          int* colIndices,
                          size_t numberOfRows,
                          int* indexConversion,
                          const int startIdx,
                          int rowsInLevelSet,
                          T* dInv,
                          const T* d,
                          T* v)
{
    cuComputeLowerSolveLevelSet<T, blocksize><<<getBlocks(rowsInLevelSet), getThreads(rowsInLevelSet)>>>(
        mat, rowIndices, colIndices, numberOfRows, indexConversion, startIdx, rowsInLevelSet, dInv, d, v);
}

// perform the upper solve for all rows in the same level set
template <class T, int blocksize>
void
computeUpperSolveLevelSet(T* mat,
                          int* rowIndices,
                          int* colIndices,
                          size_t numberOfRows,
                          int* indexConversion,
                          const int startIdx,
                          int rowsInLevelSet,
                          T* dInv,
                          const T* d,
                          T* v)
{
    cuComputeUpperSolveLevelSet<T, blocksize><<<getBlocks(rowsInLevelSet), getThreads(rowsInLevelSet)>>>(
        mat, rowIndices, colIndices, numberOfRows, indexConversion, startIdx, rowsInLevelSet, dInv, d, v);
}

template <class T, int blocksize>
void
computeDiluDiagonal(T* mat,
                    int* rowIndices,
                    int* colIndices,
                    size_t numberOfRows,
                    int* reorderedToNatural,
                    int* naturalToReordered,
                    const int startIdx,
                    int rowsInLevelSet,
                    T* dInv)
{
    if (blocksize <= 3) {
        cuComputeDiluDiagonal<T, blocksize>
            <<<getBlocks(rowsInLevelSet), getThreads(rowsInLevelSet)>>>(mat,
                                                                        rowIndices,
                                                                        colIndices,
                                                                        numberOfRows,
                                                                        reorderedToNatural,
                                                                        naturalToReordered,
                                                                        startIdx,
                                                                        rowsInLevelSet,
                                                                        dInv);
    } else {
        OPM_THROW(std::invalid_argument, "Inverting diagonal is not implemented for blocksizes > 3");
    }
}

template <class T, int blocksize>
void
moveMatDataToReordered(T* srcMatrix,
                       int* srcRowIndices,
                       int* srcColIndices,
                       T* dstMatrix,
                       int* dstRowIndices,
                       int* dstColIndices,
                       int* naturalToReordered,
                       size_t numberOfRows)
{
    cuMoveDataToReordered<T, blocksize><<<getBlocks(numberOfRows), getThreads(numberOfRows)>>>(srcMatrix,
                                                                                               srcRowIndices,
                                                                                               srcColIndices,
                                                                                               dstMatrix,
                                                                                               dstRowIndices,
                                                                                               dstColIndices,
                                                                                               naturalToReordered,
                                                                                               numberOfRows);
}

#define INSTANTIATE_KERNEL_WRAPPERS(blocksize)                                                                         \
    template void moveMatDataToReordered<float, blocksize>(float*, int*, int*, float*, int*, int*, int*, size_t);      \
    template void moveMatDataToReordered<double, blocksize>(double*, int*, int*, double*, int*, int*, int*, size_t);   \
    template void computeDiluDiagonal<float, blocksize>(                                                               \
        float*, int*, int*, size_t, int*, int*, const int, int, float*);                                               \
    template void computeDiluDiagonal<double, blocksize>(                                                              \
        double*, int*, int*, size_t, int*, int*, const int, int, double*);                                             \
    template void computeUpperSolveLevelSet<float, blocksize>(                                                         \
        float*, int*, int*, size_t, int*, const int, int, float*, const float*, float*);                               \
    template void computeUpperSolveLevelSet<double, blocksize>(                                                        \
        double*, int*, int*, size_t, int*, const int, int, double*, const double*, double*);                           \
    template void computeLowerSolveLevelSet<float, blocksize>(                                                         \
        float*, int*, int*, size_t, int*, const int, int, float*, const float*, float*);                               \
    template void computeLowerSolveLevelSet<double, blocksize>(                                                        \
        double*, int*, int*, size_t, int*, const int, int, double*, const double*, double*);

INSTANTIATE_KERNEL_WRAPPERS(1);
INSTANTIATE_KERNEL_WRAPPERS(2);
INSTANTIATE_KERNEL_WRAPPERS(3);
INSTANTIATE_KERNEL_WRAPPERS(4);
INSTANTIATE_KERNEL_WRAPPERS(5);
INSTANTIATE_KERNEL_WRAPPERS(6);

} // namespace Opm::cuistl::detail
