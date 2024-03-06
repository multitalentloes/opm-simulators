#include <hip/hip_runtime.h>
#include <stdio.h>

// CUDA kernel to print "Hello, CUDA!" from each thread
__global__ void helloCUDA() {
    printf("Hello from GPU!\n");
}

int main() {
    // Launch the kernel with 1 block and 1 thread
    helloCUDA<<<1, 1>>>();

    // Synchronize to wait for the kernel to finish
    hipDeviceSynchronize();

    // Check for and print any errors that occurred during the kernel launch
    hipError_t hipError_t = hipGetLastError();
    if (hipError_t != hipSuccess) {
        fprintf(stderr, "error: %s\n", hipGetErrorString(hipError_t));
        return 1;
    }

#ifdef __HIP_PLATFORM_AMD__
    printf("hip platform amd defined\n");
#endif


#ifdef __HIP_PLATFORM_NVIDIA__
    printf("hip platform nvidia defined\n");
#endif

    return 0;
}