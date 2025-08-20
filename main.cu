// Simple CUDA matrix multiplication (C = A * B) with CPU verification
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CUDA_CHECK(expr)                                                                 \
    do {                                                                                 \
        cudaError_t _err = (expr);                                                       \
        if (_err != cudaSuccess) {                                                       \
            std::cerr << "CUDA error: " << cudaGetErrorString(_err)                     \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;          \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)

__global__ void matMulKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

static void cpuMatMul(const std::vector<float>& A,
                      const std::vector<float>& B,
                      std::vector<float>& C,
                      int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    const int N = 256; // Matrix dimension (N x N)
    const size_t bytes = static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(float);

    std::cout << "Running CUDA matrix multiplication for " << N << "x" << N << " matrices..." << std::endl;

    // Host allocations
    std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N, 0.0f), h_C_ref(N * N, 0.0f);

    // Initialize inputs with deterministic values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>((i % 100) - 50) / 25.0f;   // values in roughly [-2, 2]
        h_B[i] = static_cast<float>(((i * 7) % 100) - 50) / 50.0f; // values in roughly [-1, 1]
    }

    // Device allocations
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matMulKernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // CPU reference for verification
    cpuMatMul(h_A, h_B, h_C_ref, N);

    // Compare results
    double maxAbsDiff = 0.0;
    double sumAbsDiff = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double diff = std::fabs(static_cast<double>(h_C[i]) - static_cast<double>(h_C_ref[i]));
        maxAbsDiff = std::max(maxAbsDiff, diff);
        sumAbsDiff += diff;
    }

    std::cout << "Max abs diff: " << maxAbsDiff << ", Avg abs diff: "
              << (sumAbsDiff / (static_cast<double>(N) * N)) << std::endl;
    std::cout << "Sample C[0]: GPU=" << h_C[0] << ", CPU=" << h_C_ref[0] << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    std::cout << "Done." << std::endl;
    return 0;
}