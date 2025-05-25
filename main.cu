#include <iostream>
#include <cmath>
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

int main() {
    // Initialize CUDA device
    CUDA_CHECK(cudaSetDevice(0));
    
    // Allocate device memory
    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
    
    // Set initial values
    double x0 = 1.0;  // Initial guess
    double tolerance = 0.0001;
    int max_iterations = 100;
    
    // Launch kernel
    newton_raphson_kernel<<<1, 1>>>(d_result, x0, tolerance, max_iterations);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    double h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    
    // Print result
    std::cout << "Root: " << h_result << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_result));
    
    return 0;
} 