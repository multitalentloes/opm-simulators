#include <config.h>

#define BOOST_TEST_MODULE TestGpuCo2GasPvt

#include <boost/test/unit_test.hpp>
#include <opm/material/common/MathToolbox.hpp>
#include <opm/material/densead/Evaluation.hpp>
#include <opm/material/densead/Math.hpp>
#include <opm/material/common/UniformTabulated2DFunction.hpp>
#include <opm/material/components/CO2Tables.hpp>
#include <opm/material/components/CO2.hpp>
#include <opm/material/components/SimpleHuDuanH2O.hpp>
#include <opm/material/components/BrineDynamic.hpp>
#include <opm/material/binarycoefficients/Brine_CO2.hpp>
#include <opm/material/fluidsystems/blackoilpvt/Co2GasPvt.hpp>
#include <opm/material/fluidsystems/blackoilpvt/BrineCo2Pvt.hpp>
#include <opm/input/eclipse/EclipseState/Co2StoreConfig.hpp>

#include <opm/simulators/linalg/gpuistl/detail/gpu_safe_call.hpp>
#include <opm/simulators/linalg/gpuistl/GpuBuffer.hpp>
#include <opm/simulators/linalg/gpuistl/GpuView.hpp>

#include <cuda_runtime.h>
#include <vector>
#include <utility>
#include <cmath> // For tolerance-based floating-point comparison

using Evaluation = Opm::DenseAd::Evaluation<float, 3>;
using GpuB = const Opm::gpuistl::GpuBuffer<double>;
using GpuV = Opm::gpuistl::GpuView<const double>;

using GpuTab = Opm::UniformTabulated2DFunction<double, GpuV>;

using GpuBufCo2Tables = Opm::CO2Tables<double, GpuB>;
using GpuViewCO2Tables = Opm::CO2Tables<double, GpuV>;
template Opm::CO2Tables<double, GpuB>::CO2Tables(const Opm::UniformTabulated2DFunction<double, GpuB>&, const Opm::UniformTabulated2DFunction<double, GpuB>&);
template Opm::CO2Tables<double, GpuV>::CO2Tables(const Opm::UniformTabulated2DFunction<double, GpuV>&, const Opm::UniformTabulated2DFunction<double, GpuV>&);
using GpuCO2 = Opm::CO2<double, GpuViewCO2Tables>;

using HuDuan = Opm::SimpleHuDuanH2O<double>;
using BrineDyn = Opm::BrineDynamic<double, HuDuan>;

using CpuBrine_CO2 = Opm::BinaryCoeff::Brine_CO2<double, HuDuan, Opm::CO2<double>>;
using GpuBrine_CO2 = Opm::BinaryCoeff::Brine_CO2<double, HuDuan, GpuCO2>;

using CpuCo2Pvt = Opm::Co2GasPvt<double>;
using GpuBufCo2Pvt = Opm::Co2GasPvt<double, GpuBufCo2Tables, GpuB>;
using GpuViewCo2Pvt = Opm::Co2GasPvt<double, GpuViewCO2Tables, GpuV>;

template GpuBufCo2Pvt::Co2GasPvt(GpuBufCo2Tables, GpuB, GpuB, GpuB, bool, bool, int, Co2StoreConfig::GasMixingType);
template GpuViewCo2Pvt::Co2GasPvt(GpuViewCO2Tables, GpuV, GpuV, GpuV, bool, bool, int, Co2StoreConfig::GasMixingType);

using CpuBrineCo2Pvt = Opm::BrineCo2Pvt<double>;
using GpuBufBrineCo2Pvt = Opm::BrineCo2Pvt<double, GpuBufCo2Tables, GpuB>;
using GpuViewBrineCo2Pvt = Opm::BrineCo2Pvt<double, GpuViewCO2Tables, GpuV>;

template GpuBufBrineCo2Pvt::BrineCo2Pvt(GpuB, GpuB, GpuB, int, int, int, GpuBufCo2Tables);
template GpuViewBrineCo2Pvt::BrineCo2Pvt(GpuV, GpuV, GpuV, int, int, int, GpuViewCO2Tables);
namespace {

/*
    This file containts tests using Co2GasPvt on the GPU in addition to helper objects
*/
// TODO: Rewrite these tests using a fixture pattern to greatly reduce the code duplication

// struct Fixture {
//     Fixture(){
//         double viscosity = Opm::CO2<double>::gasViscosity<Evaluation>(temp, pressure, true).value();

//     // make a nonstatic version of the CPU CO2
//     Opm::CO2NonStatic<double> CO2(Opm::CO2<double>::getEnthalpy(), Opm::CO2<double>::getDensity());

//     const auto gpuEnthalpyBuffer = Opm::gpuistl::move_to_gpu<double, GpuB>(CO2.getEnthalpy());
//     const auto gpuDensityBuffer = Opm::gpuistl::move_to_gpu<double, GpuB>(CO2.getDensity());

//     const auto gpuEnthalpyView = Opm::gpuistl::make_view<double, GpuB, GpuV>(gpuEnthalpyBuffer);
//     const auto gpuDensityView = Opm::gpuistl::make_view<double, GpuB, GpuV>(gpuDensityBuffer);
//     }
//     ~Fixture(){

//     }
// };

// Kernel to evaluate a 2D function on the GPU
__global__ void gpuEvaluateUniformTabulated2DFunction(GpuTab gpuTab, Evaluation* inputX, Evaluation* inputY, double* res) {
    *res = gpuTab.eval(*inputX, *inputY, true).value();
}

} // END EMPTY NAMESPACE

// Test case for evaluating a tabulated 2D function on both CPU and GPU
BOOST_AUTO_TEST_CASE(TestEvaluateUniformTabulated2DFunctionOnGpu) {
    // Example tabulated data (2D)
    std::vector<std::vector<double>> tabData = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};

    // CPU-side function definition
    Opm::UniformTabulated2DFunction<double> cpuTab(1.0, 6.0, 3, 1.0, 6.0, 2, tabData);

    // Move data to GPU buffer and create a view for GPU operations
    Opm::UniformTabulated2DFunction<double, GpuB> gpuBufTab = Opm::gpuistl::move_to_gpu<double, GpuB>(cpuTab);
    GpuTab gpuViewTab = Opm::gpuistl::make_view<double, GpuB, GpuV>(gpuBufTab);

    // Evaluation points on the CPU
    Evaluation a(2.3);
    Evaluation b(4.5);

    // Allocate memory for the result on the GPU
    double* resultOnGpu = nullptr;
    OPM_GPU_SAFE_CALL(cudaMalloc(&resultOnGpu, sizeof(double)));

    // Allocate GPU memory for the Evaluation inputs
    Evaluation* gpuA = nullptr;
    Evaluation* gpuB = nullptr;
    OPM_GPU_SAFE_CALL(cudaMalloc(&gpuA, sizeof(Evaluation)));
    OPM_GPU_SAFE_CALL(cudaMemcpy(gpuA, &a, sizeof(Evaluation), cudaMemcpyHostToDevice));
    OPM_GPU_SAFE_CALL(cudaMalloc(&gpuB, sizeof(Evaluation)));
    OPM_GPU_SAFE_CALL(cudaMemcpy(gpuB, &b, sizeof(Evaluation), cudaMemcpyHostToDevice));

    // Launch kernel to evaluate the function on the GPU
    gpuEvaluateUniformTabulated2DFunction<<<1, 1>>>(gpuViewTab, gpuA, gpuB, resultOnGpu);

    // Check for any errors in kernel launch
    OPM_GPU_SAFE_CALL(cudaPeekAtLastError());
    OPM_GPU_SAFE_CALL(cudaDeviceSynchronize());

    // Retrieve the result from the GPU to the CPU
    double resultOnCpu = 0.0;
    OPM_GPU_SAFE_CALL(cudaMemcpy(&resultOnCpu, resultOnGpu, sizeof(double), cudaMemcpyDeviceToHost));

    // Free allocated GPU memory
    OPM_GPU_SAFE_CALL(cudaFree(resultOnGpu));
    OPM_GPU_SAFE_CALL(cudaFree(gpuA));
    OPM_GPU_SAFE_CALL(cudaFree(gpuB));

    // Verify that the CPU and GPU results match within a reasonable tolerance
    const double cpuResult = cpuTab.eval(a, b, true).value();
    const double tolerance = 1e-6; // Tolerance for floating-point comparison
    BOOST_CHECK(std::fabs(resultOnCpu - cpuResult) < tolerance);
}


namespace {

// Kernel to use a CO2 object on the GPU
__global__ void gpuCO2(GpuViewCO2Tables gpuViewCo2Tables, Evaluation* temp, Evaluation* pressure, double* resultViscosity) {
    *resultViscosity = GpuCO2::gasViscosity<Evaluation>(gpuViewCo2Tables, *temp, *pressure, true).value();
}

} // END EMPTY NAMESPACE

// Test case evaluating CO2 pvt properties on CPU and GPU
BOOST_AUTO_TEST_CASE(TestUseCO2OnGpu) {
    Evaluation temp(290.5); // [K]
    Evaluation pressure(200000.0); // [Pa]

    // make a nonstatic version of the CPU CO2tables
    Opm::CO2Tables<double, std::vector<double>> co2Tables;

    // use the CO2 tables to aquire the viscosity at 290[K] and 2e5[Pa]
    double viscosity = Opm::CO2<double, Opm::CO2Tables<double, std::vector<double>>>::gasViscosity<Evaluation>(co2Tables, temp, pressure, true).value();

    GpuBufCo2Tables gpuBufCo2Table = Opm::gpuistl::move_to_gpu<double, std::vector<double>, GpuB>(co2Tables);
    GpuViewCO2Tables gpuViewCo2Table = Opm::gpuistl::make_view<double, GpuB, GpuV>(gpuBufCo2Table);
    // GpuCO2 gpuCo2(gpuEnthalpyView, gpuDensityView);

    // Allocate memory for the result on the GPU
    double* resultOnGpu = nullptr;
    OPM_GPU_SAFE_CALL(cudaMalloc(&resultOnGpu, sizeof(double)));

    // Allocate GPU memory for the Evaluation inputs
    Evaluation* gpuTemp = nullptr;
    Evaluation* gpuPressure = nullptr;
    OPM_GPU_SAFE_CALL(cudaMalloc(&gpuTemp, sizeof(Evaluation)));
    OPM_GPU_SAFE_CALL(cudaMemcpy(gpuTemp, &temp, sizeof(Evaluation), cudaMemcpyHostToDevice));
    OPM_GPU_SAFE_CALL(cudaMalloc(&gpuPressure, sizeof(Evaluation)));
    OPM_GPU_SAFE_CALL(cudaMemcpy(gpuPressure, &pressure, sizeof(Evaluation), cudaMemcpyHostToDevice));

    gpuCO2<<<1,1>>>(gpuViewCo2Table, gpuTemp, gpuPressure, resultOnGpu);

    // Check for any errors in kernel launch
    OPM_GPU_SAFE_CALL(cudaPeekAtLastError());
    OPM_GPU_SAFE_CALL(cudaDeviceSynchronize());

    // Retrieve the result from the GPU to the CPU
    double resultOnCpu = 0.0;
    OPM_GPU_SAFE_CALL(cudaMemcpy(&resultOnCpu, resultOnGpu, sizeof(double), cudaMemcpyDeviceToHost));

    // Free allocated GPU memory
    OPM_GPU_SAFE_CALL(cudaFree(resultOnGpu));
    OPM_GPU_SAFE_CALL(cudaFree(gpuTemp));
    OPM_GPU_SAFE_CALL(cudaFree(gpuPressure));

    // Verify that the CPU and GPU results match within a reasonable tolerance
    const double tolerance = 1e-6; // Tolerance for floating-point comparison
    BOOST_CHECK(std::fabs(resultOnCpu - viscosity) < tolerance);
}


namespace {

// Kernel to use a SimpleHuDuanH20 object on a GPU
__global__ void liquidDensity(Evaluation* temp, Evaluation* pressure, double* resultDensity) {
    *resultDensity = HuDuan::liquidDensity<Evaluation>(*temp, *pressure, true).value();
}

} // END EMPTY NAMESPACE

// Test case evaluating pvt values for SimpleHuDuanH20 on a GPU and CPU
BOOST_AUTO_TEST_CASE(TestUseH2OOnGpu) {
    Evaluation temp(290.5); // [K]
    Evaluation pressure(200000.0); // [Pa]

    // use the CO2 tables to aquire the densityReference at 290[K] and 2e5[Pa]
    double densityReference = HuDuan::liquidDensity<Evaluation>(temp, pressure, true).value();

    // Allocate memory for the result on the GPU
    double* resultOnGpu = nullptr;
    OPM_GPU_SAFE_CALL(cudaMalloc(&resultOnGpu, sizeof(double)));

    // Allocate GPU memory for the Evaluation inputs
    Evaluation* gpuTemp = nullptr;
    Evaluation* gpuPressure = nullptr;
    OPM_GPU_SAFE_CALL(cudaMalloc(&gpuTemp, sizeof(Evaluation)));
    OPM_GPU_SAFE_CALL(cudaMemcpy(gpuTemp, &temp, sizeof(Evaluation), cudaMemcpyHostToDevice));
    OPM_GPU_SAFE_CALL(cudaMalloc(&gpuPressure, sizeof(Evaluation)));
    OPM_GPU_SAFE_CALL(cudaMemcpy(gpuPressure, &pressure, sizeof(Evaluation), cudaMemcpyHostToDevice));

    liquidDensity<<<1,1>>>(gpuTemp, gpuPressure, resultOnGpu);

    // Check for any errors in kernel launch
    OPM_GPU_SAFE_CALL(cudaPeekAtLastError());
    OPM_GPU_SAFE_CALL(cudaDeviceSynchronize());

    // Retrieve the result from the GPU to the CPU
    double resultOnCpu = 0.0;
    OPM_GPU_SAFE_CALL(cudaMemcpy(&resultOnCpu, resultOnGpu, sizeof(double), cudaMemcpyDeviceToHost));

    // Free allocated GPU memory
    OPM_GPU_SAFE_CALL(cudaFree(resultOnGpu));
    OPM_GPU_SAFE_CALL(cudaFree(gpuTemp));
    OPM_GPU_SAFE_CALL(cudaFree(gpuPressure));

    // Verify that the CPU and GPU results match within a reasonable tolerance
    const double tolerance = 1e-6; // Tolerance for floating-point comparison
    BOOST_CHECK(std::fabs(resultOnCpu - densityReference) < tolerance);
}


namespace {

// Kernel to use a BrineDynamic object on a GPU
__global__ void liquidEnthalpy(Evaluation* temp, Evaluation* pressure, Evaluation* salinity, double* resultEnthalpy) {
    *resultEnthalpy = BrineDyn::liquidEnthalpy<Evaluation>(*temp, *pressure, *salinity).value();
}

} // END EMPTY NAMESPACE

// Test case evaluating pvt values for BrineDynamic on a GPU and CPU
BOOST_AUTO_TEST_CASE(TestUseBrineDynamicOnGpu) {
    Evaluation temp(290.5); // [K]
    Evaluation pressure(200000.0); // [Pa]
    Evaluation salinity(0.1); // [g/Kg]

    // use the CO2 tables to aquire the enthalpyReference at 290[K] and 2e5[Pa]
    double enthalpyReference = BrineDyn::liquidEnthalpy<Evaluation>(temp, pressure, salinity).value();

    // Allocate memory for the result on the GPU
    double* resultOnGpu = nullptr;
    OPM_GPU_SAFE_CALL(cudaMalloc(&resultOnGpu, sizeof(double)));

    // Allocate GPU memory for the Evaluation inputs
    Evaluation* gpuTemp = nullptr;
    Evaluation* gpuPressure = nullptr;
    Evaluation* gpuSalinity = nullptr;
    OPM_GPU_SAFE_CALL(cudaMalloc(&gpuTemp, sizeof(Evaluation)));
    OPM_GPU_SAFE_CALL(cudaMemcpy(gpuTemp, &temp, sizeof(Evaluation), cudaMemcpyHostToDevice));
    OPM_GPU_SAFE_CALL(cudaMalloc(&gpuPressure, sizeof(Evaluation)));
    OPM_GPU_SAFE_CALL(cudaMemcpy(gpuPressure, &pressure, sizeof(Evaluation), cudaMemcpyHostToDevice));
    OPM_GPU_SAFE_CALL(cudaMalloc(&gpuSalinity, sizeof(Evaluation)));
    OPM_GPU_SAFE_CALL(cudaMemcpy(gpuSalinity, &salinity, sizeof(Evaluation), cudaMemcpyHostToDevice));

    liquidEnthalpy<<<1,1>>>(gpuTemp, gpuPressure, gpuSalinity, resultOnGpu);

    // Check for any errors in kernel launch
    OPM_GPU_SAFE_CALL(cudaPeekAtLastError());
    OPM_GPU_SAFE_CALL(cudaDeviceSynchronize());

    // Retrieve the result from the GPU to the CPU
    double resultOnCpu = 0.0;
    OPM_GPU_SAFE_CALL(cudaMemcpy(&resultOnCpu, resultOnGpu, sizeof(double), cudaMemcpyDeviceToHost));

    // Free allocated GPU memory
    OPM_GPU_SAFE_CALL(cudaFree(resultOnGpu));
    OPM_GPU_SAFE_CALL(cudaFree(gpuTemp));
    OPM_GPU_SAFE_CALL(cudaFree(gpuPressure));

    // Verify that the CPU and GPU results match within a reasonable tolerance
    const double tolerance = 1e-6; // Tolerance for floating-point comparison
    BOOST_CHECK(std::fabs(resultOnCpu - enthalpyReference) < tolerance);
}


namespace {

// Kernel to use a BrineDynamic object on a GPU
__global__ void gasDiffCoeff(GpuViewCO2Tables co2tables, Evaluation* temp, Evaluation* pressure, double* result) {
    *result = GpuBrine_CO2::gasDiffCoeff<Evaluation, GpuViewCO2Tables>(co2tables, *temp, *pressure, true).value();
}

} // END EMPTY NAMESPACE

// Test case evaluating pvt values for BrineDynamic on a GPU and CPU
BOOST_AUTO_TEST_CASE(TestBrine_CO2OnGPU) {
    Evaluation temp(290.5); // [K]
    Evaluation pressure(200000.0); // [Pa]


    // make a nonstatic version of the CPU CO2tables
    Opm::CO2Tables<double, std::vector<double>> co2Tables;

    // use the CO2 tables to aquire the enthalpyReference at 290[K] and 2e5[Pa]
    double enthalpyReference = CpuBrine_CO2::gasDiffCoeff<Evaluation>(co2Tables, temp, pressure, true).value();

    // use the CO2 tables to aquire the viscosity at 290[K] and 2e5[Pa]
    double viscosity = Opm::CO2<double, Opm::CO2Tables<double, std::vector<double>>>::gasViscosity<Evaluation>(co2Tables, temp, pressure, true).value();

    GpuBufCo2Tables gpuBufCo2Table = Opm::gpuistl::move_to_gpu<double, std::vector<double>, GpuB>(co2Tables);
    GpuViewCO2Tables gpuViewCo2Table = Opm::gpuistl::make_view<double, GpuB, GpuV>(gpuBufCo2Table);

    // Allocate memory for the result on the GPU
    double* resultOnGpu = nullptr;
    OPM_GPU_SAFE_CALL(cudaMalloc(&resultOnGpu, sizeof(double)));

    // Allocate GPU memory for the Evaluation inputs
    Evaluation* gpuTemp = nullptr;
    Evaluation* gpuPressure = nullptr;
    OPM_GPU_SAFE_CALL(cudaMalloc(&gpuTemp, sizeof(Evaluation)));
    OPM_GPU_SAFE_CALL(cudaMemcpy(gpuTemp, &temp, sizeof(Evaluation), cudaMemcpyHostToDevice));
    OPM_GPU_SAFE_CALL(cudaMalloc(&gpuPressure, sizeof(Evaluation)));
    OPM_GPU_SAFE_CALL(cudaMemcpy(gpuPressure, &pressure, sizeof(Evaluation), cudaMemcpyHostToDevice));

    gasDiffCoeff<<<1,1>>>(gpuViewCo2Table, gpuTemp, gpuPressure, resultOnGpu);

    // Check for any errors in kernel launch
    OPM_GPU_SAFE_CALL(cudaPeekAtLastError());
    OPM_GPU_SAFE_CALL(cudaDeviceSynchronize());

    // Retrieve the result from the GPU to the CPU
    double resultOnCpu = 0.0;
    OPM_GPU_SAFE_CALL(cudaMemcpy(&resultOnCpu, resultOnGpu, sizeof(double), cudaMemcpyDeviceToHost));

    // Free allocated GPU memory
    OPM_GPU_SAFE_CALL(cudaFree(resultOnGpu));
    OPM_GPU_SAFE_CALL(cudaFree(gpuTemp));
    OPM_GPU_SAFE_CALL(cudaFree(gpuPressure));

    // Verify that the CPU and GPU results match within a reasonable tolerance
    const double tolerance = 1e-6; // Tolerance for floating-point comparison
    BOOST_CHECK(std::fabs(resultOnCpu - enthalpyReference) < tolerance);
}

namespace {

// Kernel to use a BrineDynamic object on a GPU
__global__ void pvtInternalEnergy(GpuViewCo2Pvt gpuViewCo2Pvt, Evaluation* temp, Evaluation* pressure, double* result) {
    *result = gpuViewCo2Pvt.internalEnergy<Evaluation>(1, *temp, *pressure, Evaluation(0.4), Evaluation(0.0)).value();
}

bool compareSignificantDigits(double a, double b, int significantDigits) {
    // Handle the case where both values are exactly zero
    if (a == b) {
        return true;
    }

    // Calculate the relative error
    double relativeError = std::abs(a - b) / std::max(std::abs(a), std::abs(b));

    // Compute the number of matching digits
    double digitsMatched = -std::log10(relativeError);

    // Return true if they match the required number of significant digits
    return digitsMatched >= significantDigits;
}

} // END EMPTY NAMESPACE

// Test case evaluating pvt values for BrineDynamic on a GPU and CPU
BOOST_AUTO_TEST_CASE(TestCo2GasPvt) {
    Evaluation temp(290.5); // [K]
    Evaluation pressure(200000.0); // [Pa]
    std::vector<double> salinities = {0.2, 0.3, 0.4};
    // make a nonstatic version of the CPU CO2tables
    Opm::CO2Tables<double, std::vector<double>> co2Tables;

    CpuCo2Pvt cpuCo2Pvt(salinities);
    double internalEnergyReference = cpuCo2Pvt.internalEnergy<Evaluation>(1, temp, pressure, Evaluation(0.4), Evaluation(0.0)).value();

    const GpuBufCo2Pvt gpuBufCo2Pvt = Opm::gpuistl::move_to_gpu<double, GpuBufCo2Tables, GpuB>(cpuCo2Pvt);
    const auto brineReferenceDensityCPUCopy = gpuBufCo2Pvt.getBrineReferenceDensity().asStdVector();
    const GpuViewCo2Pvt gpuViewCo2Pvt = Opm::gpuistl::make_view<double, GpuBufCo2Tables, GpuViewCO2Tables, GpuB, GpuV>(gpuBufCo2Pvt);

    // Allocate memory for the result on the GPU
    double* resultOnGpu = nullptr;
    OPM_GPU_SAFE_CALL(cudaMalloc(&resultOnGpu, sizeof(double)));

    // Allocate GPU memory for the Evaluation inputs
    Evaluation* gpuTemp = nullptr;
    Evaluation* gpuPressure = nullptr;
    OPM_GPU_SAFE_CALL(cudaMalloc(&gpuTemp, sizeof(Evaluation)));
    OPM_GPU_SAFE_CALL(cudaMemcpy(gpuTemp, &temp, sizeof(Evaluation), cudaMemcpyHostToDevice));
    OPM_GPU_SAFE_CALL(cudaMalloc(&gpuPressure, sizeof(Evaluation)));
    OPM_GPU_SAFE_CALL(cudaMemcpy(gpuPressure, &pressure, sizeof(Evaluation), cudaMemcpyHostToDevice));

    pvtInternalEnergy<<<1,1>>>(gpuViewCo2Pvt, gpuTemp, gpuPressure, resultOnGpu);

    // Check for any errors in kernel launch
    OPM_GPU_SAFE_CALL(cudaPeekAtLastError());
    OPM_GPU_SAFE_CALL(cudaDeviceSynchronize());

    // Retrieve the result from the GPU to the CPU
    double resultOnCpu = 0.0;
    OPM_GPU_SAFE_CALL(cudaMemcpy(&resultOnCpu, resultOnGpu, sizeof(double), cudaMemcpyDeviceToHost));

    // Free allocated GPU memory
    OPM_GPU_SAFE_CALL(cudaFree(resultOnGpu));
    OPM_GPU_SAFE_CALL(cudaFree(gpuTemp));
    OPM_GPU_SAFE_CALL(cudaFree(gpuPressure));

    // Verify that the CPU and GPU results match within a reasonable tolerance
    const double tolerance = 1e-6; // Tolerance for floating-point comparison
    printf("%lf %lf\n", resultOnCpu, internalEnergyReference);
    // BOOST_CHECK(std::fabs(resultOnCpu - internalEnergyReference) < tolerance);
    BOOST_CHECK(compareSignificantDigits(resultOnCpu, internalEnergyReference, 6));
}


namespace {

// Kernel to use a BrineDynamic object on a GPU
__global__ void brineCo2PvtInternalEnergy(GpuViewBrineCo2Pvt gpuViewBrineCo2Pvt, Evaluation* temp, Evaluation* pressure, Evaluation* rs, Evaluation saltConcentration, double* result) {
    *result = gpuViewBrineCo2Pvt.internalEnergy<Evaluation>(1, *temp, *pressure, rs, saltConcentration).value();
}

} // END EMPTY NAMESPACE

// Test case evaluating pvt values for BrineDynamic on a GPU and CPU
BOOST_AUTO_TEST_CASE(TestBrineCo2Pvt) {
    Evaluation temp(290.5); // [K]
    Evaluation pressure(200000.0); // [Pa]
    Evaluation rs(0.3);
    Evaluation saltConcentration(0.1);

    std::vector<double> salinities = {0.2, 0.3, 0.4};
    // make a nonstatic version of the CPU CO2tables
    Opm::CO2Tables<double, std::vector<double>> co2Tables;

    CpuBrineCo2Pvt cpuBrineCo2Pvt(salinities);
    double internalEnergyReference = cpuBrineCo2Pvt.internalEnergy<Evaluation>(1, temp, pressure, rs, saltConcentration).value();

    const GpuViewBrineCo2Pvt gpuBufBrineCo2Pvt = Opm::gpuistl::move_to_gpu<double, GpuBufCo2Tables, GpuB>(cpuBrineCo2Pvt);
    const GpuViewBrineCo2Pvt gpuViewBrineCo2Pvt = Opm::gpuistl::make_view<double, GpuBufCo2Tables, GpuViewCO2Tables, GpuB, GpuV>(gpuBufCo2Pvt);

    // Allocate memory for the result on the GPU
    double* resultOnGpu = nullptr;
    OPM_GPU_SAFE_CALL(cudaMalloc(&resultOnGpu, sizeof(double)));

    // Allocate GPU memory for the Evaluation inputs
    Evaluation* gpuRs = nullptr;
    Evaluation* gpuSaltConcentration = nullptr;
    OPM_GPU_SAFE_CALL(cudaMalloc(&gpuRs, sizeof(Evaluation)));
    OPM_GPU_SAFE_CALL(cudaMemcpy(gpuRs, &rs, sizeof(Evaluation), cudaMemcpyHostToDevice));
    OPM_GPU_SAFE_CALL(cudaMalloc(&gpuSaltConcentration, sizeof(Evaluation)));
    OPM_GPU_SAFE_CALL(cudaMemcpy(gpuSaltConcentration, &saltConcentration, sizeof(Evaluation), cudaMemcpyHostToDevice));
    Evaluation* gpuTemp = nullptr;
    Evaluation* gpuPressure = nullptr;
    OPM_GPU_SAFE_CALL(cudaMalloc(&gpuTemp, sizeof(Evaluation)));
    OPM_GPU_SAFE_CALL(cudaMemcpy(gpuTemp, &temp, sizeof(Evaluation), cudaMemcpyHostToDevice));
    OPM_GPU_SAFE_CALL(cudaMalloc(&gpuPressure, sizeof(Evaluation)));
    OPM_GPU_SAFE_CALL(cudaMemcpy(gpuPressure, &pressure, sizeof(Evaluation), cudaMemcpyHostToDevice));

    brineCo2PvtInternalEnergy<<<1,1>>>(gpuViewBrineCo2Pvt, gpuTemp, gpuPressure, gpuRs, gpuSaltConcentration, resultOnGpu);

    // Check for any errors in kernel launch
    OPM_GPU_SAFE_CALL(cudaPeekAtLastError());
    OPM_GPU_SAFE_CALL(cudaDeviceSynchronize());

    // Retrieve the result from the GPU to the CPU
    double resultOnCpu = 0.0;
    OPM_GPU_SAFE_CALL(cudaMemcpy(&resultOnCpu, resultOnGpu, sizeof(double), cudaMemcpyDeviceToHost));

    // Free allocated GPU memory
    OPM_GPU_SAFE_CALL(cudaFree(resultOnGpu));
    OPM_GPU_SAFE_CALL(cudaFree(gpuTemp));
    OPM_GPU_SAFE_CALL(cudaFree(gpuPressure));

    // Verify that the CPU and GPU results match within a reasonable tolerance
    const double tolerance = 1e-6; // Tolerance for floating-point comparison
    printf("%lf %lf\n", resultOnCpu, internalEnergyReference);
    // BOOST_CHECK(std::fabs(resultOnCpu - internalEnergyReference) < tolerance);
    BOOST_CHECK(compareSignificantDigits(resultOnCpu, internalEnergyReference, 6));
}
