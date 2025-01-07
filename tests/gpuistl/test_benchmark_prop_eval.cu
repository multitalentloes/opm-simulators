#define HAVE_ECL_INPUT 1
#include <config.h>


#include <stdexcept>

#define BOOST_TEST_MODULE TestBenchmarkPropEval

#include <cuda.h>
#include <cuda_runtime.h>
#include <boost/test/unit_test.hpp>
#include <opm/common/ErrorMacros.hpp>
#include <opm/models/blackoil/blackoilintensivequantities.hh>

// Example of a simple CUDA kernel that could be used for property evaluation
__global__ void dummyKernel() {
    Opm::void_update(); // blackoilintensivequantities.hh
}

BOOST_AUTO_TEST_CASE(benchmark_prop_eval)
{
    // This test is a placeholder for benchmarking property evaluation on GPU.
    // It is expected to be replaced with actual implementation details.
    


    // Launch the dummy kernel
    dummyKernel<<<1, 1>>>();
    
    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }
}
