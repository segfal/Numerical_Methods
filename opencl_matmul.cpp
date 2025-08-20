// OpenCL matrix multiplication (C = A * B) with CPU verification
// Build on Linux: g++ -std=c++17 opencl_matmul.cpp -lOpenCL -o opencl_matmul
// Build on macOS: g++ -std=c++17 opencl_matmul.cpp -framework OpenCL -o opencl_matmul

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>

static void checkOrExit(cl_int status, const char* msg) {
    if (status != CL_SUCCESS) {
        std::cerr << "OpenCL error (" << status << "): " << msg << std::endl;
        std::exit(EXIT_FAILURE);
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

    std::cout << "Running OpenCL matrix multiplication for " << N << "x" << N << " matrices..." << std::endl;

    // Host buffers
    std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N, 0.0f), h_C_ref(N * N, 0.0f);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>((i % 100) - 50) / 25.0f;
        h_B[i] = static_cast<float>(((i * 7) % 100) - 50) / 50.0f;
    }

    // Kernel source
    const char* kernelSrc = R"CLC(
    __kernel void matmul(__global const float* A,
                         __global const float* B,
                         __global float* C,
                         const int N) {
        const int col = get_global_id(0);
        const int row = get_global_id(1);
        if (row < N && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    )CLC";

    cl_int status = CL_SUCCESS;

    // Discover platforms
    cl_uint numPlatforms = 0;
    status = clGetPlatformIDs(0, nullptr, &numPlatforms);
    checkOrExit(status, "clGetPlatformIDs count");
    if (numPlatforms == 0) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        return 1;
    }
    std::vector<cl_platform_id> platforms(numPlatforms);
    status = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    checkOrExit(status, "clGetPlatformIDs list");

    // Pick first platform with a GPU, otherwise CPU
    cl_device_id device = nullptr;
    cl_platform_id chosenPlatform = nullptr;
    for (cl_platform_id plat : platforms) {
        cl_uint numDevices = 0;
        status = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
        if (status == CL_SUCCESS && numDevices > 0) {
            std::vector<cl_device_id> devices(numDevices);
            status = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
            checkOrExit(status, "clGetDeviceIDs GPU list");
            device = devices[0];
            chosenPlatform = plat;
            break;
        }
    }
    if (!device) {
        // Fallback to CPU
        for (cl_platform_id plat : platforms) {
            cl_uint numDevices = 0;
            status = clGetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, 0, nullptr, &numDevices);
            if (status == CL_SUCCESS && numDevices > 0) {
                std::vector<cl_device_id> devices(numDevices);
                status = clGetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, numDevices, devices.data(), nullptr);
                checkOrExit(status, "clGetDeviceIDs CPU list");
                device = devices[0];
                chosenPlatform = plat;
                break;
            }
        }
    }
    if (!device) {
        std::cerr << "No suitable OpenCL device found." << std::endl;
        return 1;
    }

    // Create context
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(chosenPlatform), 0
    };
    cl_context context = clCreateContext(props, 1, &device, nullptr, nullptr, &status);
    checkOrExit(status, "clCreateContext");

    // Create command queue (OpenCL 1.2)
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &status);
    checkOrExit(status, "clCreateCommandQueue");

    // Create buffers
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_A.data(), &status);
    checkOrExit(status, "clCreateBuffer A");
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_B.data(), &status);
    checkOrExit(status, "clCreateBuffer B");
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &status);
    checkOrExit(status, "clCreateBuffer C");

    // Build program
    const size_t srcLen = std::strlen(kernelSrc);
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSrc, &srcLen, &status);
    checkOrExit(status, "clCreateProgramWithSource");
    status = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
    if (status != CL_SUCCESS) {
        // Print build log
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr);
        std::cerr << "Build failed:\n" << log << std::endl;
        checkOrExit(status, "clBuildProgram");
    }

    cl_kernel kernel = clCreateKernel(program, "matmul", &status);
    checkOrExit(status, "clCreateKernel");

    // Set kernel args
    checkOrExit(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A), "clSetKernelArg 0");
    checkOrExit(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B), "clSetKernelArg 1");
    checkOrExit(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C), "clSetKernelArg 2");
    checkOrExit(clSetKernelArg(kernel, 3, sizeof(int), &N), "clSetKernelArg 3");

    // Launch configuration
    const size_t local[2] = {16, 16};
    const size_t global[2] = {
        static_cast<size_t>((N + local[0] - 1) / local[0]) * local[0],
        static_cast<size_t>((N + local[1] - 1) / local[1]) * local[1]
    };

    // Enqueue kernel
    checkOrExit(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel");

    // Read back
    checkOrExit(clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, bytes, h_C.data(), 0, nullptr, nullptr),
                "clEnqueueReadBuffer");

    // CPU reference and comparison
    cpuMatMul(h_A, h_B, h_C_ref, N);
    double maxAbsDiff = 0.0;
    double sumAbsDiff = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double diff = std::fabs(static_cast<double>(h_C[i]) - static_cast<double>(h_C_ref[i]));
        if (diff > maxAbsDiff) maxAbsDiff = diff;
        sumAbsDiff += diff;
    }
    std::cout << "Max abs diff: " << maxAbsDiff
              << ", Avg abs diff: " << (sumAbsDiff / (static_cast<double>(N) * N)) << std::endl;
    std::cout << "Sample C[0]: GPU=" << h_C[0] << ", CPU=" << h_C_ref[0] << std::endl;

    // Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    std::cout << "Done." << std::endl;
    return 0;
}


