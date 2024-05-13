#include <cuda_runtime.h>
#include <opm/material/densead/Evaluation.hpp>
#include <opm/simulators/linalg/cuistl/detail/AD_TEST.hpp>
#include <opm/simulators/linalg/cuistl/detail/cuda_safe_call.hpp>

namespace{

// do some basic test with AD objects
__global__ void instansiate_ad_object(Opm::DenseAd::Evaluation<float, 3>* adObj, double value){
    *adObj = Opm::DenseAd::Evaluation<float, 3>(value, 0);
}

} // END EMPTY NAMESPACE


namespace Opm::cuistl::detail{

Opm::DenseAd::Evaluation<float, 3> test_ad_instansiate_kernel(double value){
    auto h_ad = Opm::DenseAd::Evaluation<float, 3>(value, 0);
    Opm::DenseAd::Evaluation<float, 3> *d_ad;
    OPM_CUDA_SAFE_CALL(cudaMallocManaged(&d_ad, sizeof(Opm::DenseAd::Evaluation<float, 3>)));
    instansiate_ad_object<<<1,1>>>(d_ad, value);
    OPM_CUDA_SAFE_CALL(cudaDeviceSynchronize());

    OPM_CUDA_SAFE_CALL(cudaMemcpy(&h_ad, d_ad, sizeof(Opm::DenseAd::Evaluation<float, 3>), cudaMemcpyDeviceToHost));

    OPM_CUDA_SAFE_CALL(cudaFree(d_ad));

    return h_ad;
}

} // END NAMESPACE
