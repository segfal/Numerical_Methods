#include "../includes/RootFinding.hpp"
#include <cuda_runtime.h>

// CUDA device functions
__device__ double f(double x) {
    return x * x - 2;
}

__device__ double f_prime(double x) {
    return 2 * x;
}

// CUDA kernel for Newton-Raphson method
__global__ void newton_raphson_kernel(double* result, double x0, double tolerance, int max_iterations) {
    double x = x0;
    double h;
    
    for (int i = 0; i < max_iterations; i++) {
        h = f(x) / f_prime(x);
        x = x - h;
        
        if (fabs(h) < tolerance) {
            break;
        }
    }
    
    *result = x;
}

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

double RootFinding::find_root(std::function<double(double)> f, double a, double b, 
                            std::function<double(double)> f_prime, double x0, double x1) {
    if (method != RootFindingMethod::NewtonRaphson) {
        throw std::invalid_argument("Only Newton-Raphson method is supported in CUDA version");
    }

    // Initialize CUDA device
    CUDA_CHECK(cudaSetDevice(0));
    
    // Allocate device memory
    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
    
    // Launch kernel
    newton_raphson_kernel<<<1, 1>>>(d_result, x0, tolerance, max_iterations);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    double h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_result));
    
    return h_result;
} 