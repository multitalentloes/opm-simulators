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

/**
 * Contains wrappers to make the CuSPARSE library behave as a modern C++ library with function overlading.
 *
 * In simple terms, this allows one to call say cusparseBsrilu02_analysis on both double and single precisision,
 * instead of calling hipsparseDbsrilu02_analysis and hipsparseDbsrilu02_analysis respectively.
 */
#include <hipsparse/hipsparse.h>
#include <type_traits>
#ifndef OPM_CUSPARSE_WRAPPER_HPP_HIP
#define OPM_CUSPARSE_WRAPPER_HPP_HIP
namespace Opm::hipistl::detail
{

inline hipsparseStatus_t
cusparseBsrilu02_analysis(hipsparseHandle_t handle,
                          hipsparseDirection_t dirA,
                          int mb,
                          int nnzb,
                          const hipsparseMatDescr_t descrA,
                          double* bsrSortedVal,
                          const int* bsrSortedRowPtr,
                          const int* bsrSortedColInd,
                          int blockDim,
                          bsrilu02Info_t info,
                          hipsparseSolvePolicy_t policy,
                          void* pBuffer)
{
    return hipsparseDbsrilu02_analysis(handle,
                                      dirA,
                                      mb,
                                      nnzb,
                                      descrA,
                                      bsrSortedVal,
                                      bsrSortedRowPtr,
                                      bsrSortedColInd,
                                      blockDim,
                                      info,
                                      policy,
                                      pBuffer);
}

inline hipsparseStatus_t
cusparseBsrsv2_analysis(hipsparseHandle_t handle,
                        hipsparseDirection_t dirA,
                        hipsparseOperation_t transA,
                        int mb,
                        int nnzb,
                        const hipsparseMatDescr_t descrA,
                        const double* bsrSortedValA,
                        const int* bsrSortedRowPtrA,
                        const int* bsrSortedColIndA,
                        int blockDim,
                        bsrsv2Info_t info,
                        hipsparseSolvePolicy_t policy,
                        void* pBuffer)
{
    return hipsparseDbsrsv2_analysis(handle,
                                    dirA,
                                    transA,
                                    mb,
                                    nnzb,
                                    descrA,
                                    bsrSortedValA,
                                    bsrSortedRowPtrA,
                                    bsrSortedColIndA,
                                    blockDim,
                                    info,
                                    policy,
                                    pBuffer);
}

inline hipsparseStatus_t
cusparseBsrsv2_analysis(hipsparseHandle_t handle,
                        hipsparseDirection_t dirA,
                        hipsparseOperation_t transA,
                        int mb,
                        int nnzb,
                        const hipsparseMatDescr_t descrA,
                        const float* bsrSortedValA,
                        const int* bsrSortedRowPtrA,
                        const int* bsrSortedColIndA,
                        int blockDim,
                        bsrsv2Info_t info,
                        hipsparseSolvePolicy_t policy,
                        void* pBuffer)
{
    return hipsparseSbsrsv2_analysis(handle,
                                    dirA,
                                    transA,
                                    mb,
                                    nnzb,
                                    descrA,
                                    bsrSortedValA,
                                    bsrSortedRowPtrA,
                                    bsrSortedColIndA,
                                    blockDim,
                                    info,
                                    policy,
                                    pBuffer);
}

inline hipsparseStatus_t
cusparseBsrilu02_analysis(hipsparseHandle_t handle,
                          hipsparseDirection_t dirA,
                          int mb,
                          int nnzb,
                          const hipsparseMatDescr_t descrA,
                          float* bsrSortedVal,
                          const int* bsrSortedRowPtr,
                          const int* bsrSortedColInd,
                          int blockDim,
                          bsrilu02Info_t info,
                          hipsparseSolvePolicy_t policy,
                          void* pBuffer)
{
    return hipsparseSbsrilu02_analysis(handle,
                                      dirA,
                                      mb,
                                      nnzb,
                                      descrA,
                                      bsrSortedVal,
                                      bsrSortedRowPtr,
                                      bsrSortedColInd,
                                      blockDim,
                                      info,
                                      policy,
                                      pBuffer);
}

inline hipsparseStatus_t
cusparseBsrsv2_solve(hipsparseHandle_t handle,
                     hipsparseDirection_t dirA,
                     hipsparseOperation_t transA,
                     int mb,
                     int nnzb,
                     const double* alpha,
                     const hipsparseMatDescr_t descrA,
                     const double* bsrSortedValA,
                     const int* bsrSortedRowPtrA,
                     const int* bsrSortedColIndA,
                     int blockDim,
                     bsrsv2Info_t info,
                     const double* f,
                     double* x,
                     hipsparseSolvePolicy_t policy,
                     void* pBuffer)
{
    return hipsparseDbsrsv2_solve(handle,
                                 dirA,
                                 transA,
                                 mb,
                                 nnzb,
                                 alpha,
                                 descrA,
                                 bsrSortedValA,
                                 bsrSortedRowPtrA,
                                 bsrSortedColIndA,
                                 blockDim,
                                 info,
                                 f,
                                 x,
                                 policy,
                                 pBuffer);
}


inline hipsparseStatus_t
cusparseBsrsv2_solve(hipsparseHandle_t handle,
                     hipsparseDirection_t dirA,
                     hipsparseOperation_t transA,
                     int mb,
                     int nnzb,
                     const float* alpha,
                     const hipsparseMatDescr_t descrA,
                     const float* bsrSortedValA,
                     const int* bsrSortedRowPtrA,
                     const int* bsrSortedColIndA,
                     int blockDim,
                     bsrsv2Info_t info,
                     const float* f,
                     float* x,
                     hipsparseSolvePolicy_t policy,
                     void* pBuffer)
{
    return hipsparseSbsrsv2_solve(handle,
                                 dirA,
                                 transA,
                                 mb,
                                 nnzb,
                                 alpha,
                                 descrA,
                                 bsrSortedValA,
                                 bsrSortedRowPtrA,
                                 bsrSortedColIndA,
                                 blockDim,
                                 info,
                                 f,
                                 x,
                                 policy,
                                 pBuffer);
}


inline hipsparseStatus_t
cusparseBsrilu02_bufferSize(hipsparseHandle_t handle,
                            hipsparseDirection_t dirA,
                            int mb,
                            int nnzb,
                            const hipsparseMatDescr_t descrA,
                            double* bsrSortedVal,
                            const int* bsrSortedRowPtr,
                            const int* bsrSortedColInd,
                            int blockDim,
                            bsrilu02Info_t info,
                            int* pBufferSizeInBytes)
{
    return hipsparseDbsrilu02_bufferSize(handle,
                                        dirA,
                                        mb,
                                        nnzb,
                                        descrA,
                                        bsrSortedVal,
                                        bsrSortedRowPtr,
                                        bsrSortedColInd,
                                        blockDim,
                                        info,
                                        pBufferSizeInBytes);
}


inline hipsparseStatus_t
cusparseBsrilu02_bufferSize(hipsparseHandle_t handle,
                            hipsparseDirection_t dirA,
                            int mb,
                            int nnzb,
                            const hipsparseMatDescr_t descrA,
                            float* bsrSortedVal,
                            const int* bsrSortedRowPtr,
                            const int* bsrSortedColInd,
                            int blockDim,
                            bsrilu02Info_t info,
                            int* pBufferSizeInBytes)
{
    return hipsparseSbsrilu02_bufferSize(handle,
                                        dirA,
                                        mb,
                                        nnzb,
                                        descrA,
                                        bsrSortedVal,
                                        bsrSortedRowPtr,
                                        bsrSortedColInd,
                                        blockDim,
                                        info,
                                        pBufferSizeInBytes);
}

inline hipsparseStatus_t
cusparseBsrsv2_bufferSize(hipsparseHandle_t handle,
                          hipsparseDirection_t dirA,
                          hipsparseOperation_t transA,
                          int mb,
                          int nnzb,
                          const hipsparseMatDescr_t descrA,
                          double* bsrSortedValA,
                          const int* bsrSortedRowPtrA,
                          const int* bsrSortedColIndA,
                          int blockDim,
                          bsrsv2Info_t info,
                          int* pBufferSizeInBytes)
{
    return hipsparseDbsrsv2_bufferSize(handle,
                                      dirA,
                                      transA,
                                      mb,
                                      nnzb,
                                      descrA,
                                      bsrSortedValA,
                                      bsrSortedRowPtrA,
                                      bsrSortedColIndA,
                                      blockDim,
                                      info,
                                      pBufferSizeInBytes);
}
inline hipsparseStatus_t
cusparseBsrsv2_bufferSize(hipsparseHandle_t handle,
                          hipsparseDirection_t dirA,
                          hipsparseOperation_t transA,
                          int mb,
                          int nnzb,
                          const hipsparseMatDescr_t descrA,
                          float* bsrSortedValA,
                          const int* bsrSortedRowPtrA,
                          const int* bsrSortedColIndA,
                          int blockDim,
                          bsrsv2Info_t info,
                          int* pBufferSizeInBytes)
{
    return hipsparseSbsrsv2_bufferSize(handle,
                                      dirA,
                                      transA,
                                      mb,
                                      nnzb,
                                      descrA,
                                      bsrSortedValA,
                                      bsrSortedRowPtrA,
                                      bsrSortedColIndA,
                                      blockDim,
                                      info,
                                      pBufferSizeInBytes);
}

inline hipsparseStatus_t
cusparseBsrilu02(hipsparseHandle_t handle,
                 hipsparseDirection_t dirA,
                 int mb,
                 int nnzb,
                 const hipsparseMatDescr_t descrA,
                 double* bsrSortedVal,
                 const int* bsrSortedRowPtr,
                 const int* bsrSortedColInd,
                 int blockDim,
                 bsrilu02Info_t info,
                 hipsparseSolvePolicy_t policy,
                 void* pBuffer)
{
    return hipsparseDbsrilu02(handle,
                             dirA,
                             mb,
                             nnzb,
                             descrA,
                             bsrSortedVal,
                             bsrSortedRowPtr,
                             bsrSortedColInd,
                             blockDim,
                             info,
                             policy,
                             pBuffer);
}
inline hipsparseStatus_t
cusparseBsrilu02(hipsparseHandle_t handle,
                 hipsparseDirection_t dirA,
                 int mb,
                 int nnzb,
                 const hipsparseMatDescr_t descrA,
                 float* bsrSortedVal,
                 const int* bsrSortedRowPtr,
                 const int* bsrSortedColInd,
                 int blockDim,
                 bsrilu02Info_t info,
                 hipsparseSolvePolicy_t policy,
                 void* pBuffer)
{
    return hipsparseSbsrilu02(handle,
                             dirA,
                             mb,
                             nnzb,
                             descrA,
                             bsrSortedVal,
                             bsrSortedRowPtr,
                             bsrSortedColInd,
                             blockDim,
                             info,
                             policy,
                             pBuffer);
}

inline hipsparseStatus_t
cusparseBsrmv(hipsparseHandle_t handle,
              hipsparseDirection_t dirA,
              hipsparseOperation_t transA,
              int mb,
              int nb,
              int nnzb,
              const double* alpha,
              const hipsparseMatDescr_t descrA,
              const double* bsrSortedValA,
              const int* bsrSortedRowPtrA,
              const int* bsrSortedColIndA,
              int blockDim,
              const double* x,
              const double* beta,
              double* y)
{
    return hipsparseDbsrmv(handle,
                          dirA,
                          transA,
                          mb,
                          nb,
                          nnzb,
                          alpha,
                          descrA,
                          bsrSortedValA,
                          bsrSortedRowPtrA,
                          bsrSortedColIndA,
                          blockDim,
                          x,
                          beta,
                          y);
}

inline hipsparseStatus_t
cusparseBsrmv(hipsparseHandle_t handle,
              hipsparseDirection_t dirA,
              hipsparseOperation_t transA,
              int mb,
              int nb,
              int nnzb,
              const float* alpha,
              const hipsparseMatDescr_t descrA,
              const float* bsrSortedValA,
              const int* bsrSortedRowPtrA,
              const int* bsrSortedColIndA,
              int blockDim,
              const float* x,
              const float* beta,
              float* y)
{
    return hipsparseSbsrmv(handle,
                          dirA,
                          transA,
                          mb,
                          nb,
                          nnzb,
                          alpha,
                          descrA,
                          bsrSortedValA,
                          bsrSortedRowPtrA,
                          bsrSortedColIndA,
                          blockDim,
                          x,
                          beta,
                          y);
}
} // namespace Opm::hipistl::detail
#endif
