#include <iostream>
#include <vector>
#include <iomanip>

class Matrix {
public:
    float* data;
    int rows, cols;
    bool on_gpu;
    
    // Constructor
    Matrix(int r, int c, bool gpu = false) : rows(r), cols(c), on_gpu(gpu) {
        if (on_gpu) {
            cudaMalloc(&data, rows * cols * sizeof(float));
        } else {
            data = new float[rows * cols];
        }
    }
    
    // Destructor
    ~Matrix() {
        if (on_gpu) {
            cudaFree(data);
        } else {
            delete[] data;
        }
    }
    
    // Get element (only for CPU matrices)
    float& operator()(int r, int c) {
        return data[r * cols + c];
    }
    
    // Copy to GPU
    Matrix* toGPU() {
        Matrix* gpu_mat = new Matrix(rows, cols, true);
        cudaMemcpy(gpu_mat->data, data, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
        return gpu_mat;
    }
    
    // Copy from GPU
    Matrix* toCPU() {
        Matrix* cpu_mat = new Matrix(rows, cols, false);
        cudaMemcpy(cpu_mat->data, data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
        return cpu_mat;
    }
    
    // Print matrix (only for CPU matrices)
    void print() {
        if (on_gpu) {
            std::cout << "Cannot print GPU matrix directly!" << std::endl;
            return;
        }
        
        std::cout << std::fixed << std::setprecision(2);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << std::setw(8) << (*this)(i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    // Fill with random values
    void randomFill() {
        if (on_gpu) {
            std::cout << "Cannot fill GPU matrix directly!" << std::endl;
            return;
        }
        
        for (int i = 0; i < rows * cols; i++) {
            data[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
        }
    }
};

// CUDA Kernel for Matrix Multiplication
__global__ void matrixMultiply(float* A, float* B, float* C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (row >= rowsA || col >= colsB) return;
    
    float sum = 0.0f;
    for (int k = 0; k < colsA; k++) {
        sum += A[row * colsA + k] * B[k * colsB + col];
    }
    
    C[row * colsB + col] = sum;
}

// CUDA Kernel for Matrix Addition
__global__ void matrixAdd(float* A, float* B, float* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= rows * cols) return;
    
    C[idx] = A[idx] + B[idx];
}

// Wrapper functions for easier use
Matrix* multiply(Matrix* A, Matrix* B) {
    if (A->cols != B->rows) {
        std::cout << "Matrix dimensions don't match for multiplication!" << std::endl;
        return nullptr;
    }
    
    Matrix* C = new Matrix(A->rows, B->cols, true);
    
    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (B->cols + blockSize.x - 1) / blockSize.x,
        (A->rows + blockSize.y - 1) / blockSize.y
    );
    
    matrixMultiply<<<gridSize, blockSize>>>(A->data, B->data, C->data, A->rows, A->cols, B->cols);
    cudaDeviceSynchronize();
    
    return C;
}

Matrix* add(Matrix* A, Matrix* B) {
    if (A->rows != B->rows || A->cols != B->cols) {
        std::cout << "Matrix dimensions don't match for addition!" << std::endl;
        return nullptr;
    }
    
    Matrix* C = new Matrix(A->rows, A->cols, true);
    
    int blockSize = 256;
    int gridSize = (A->rows * A->cols + blockSize - 1) / blockSize;
    
    matrixAdd<<<gridSize, blockSize>>>(A->data, B->data, C->data, A->rows, A->cols);
    cudaDeviceSynchronize();
    
    return C;
}

int main() {
    std::cout << "=== Simple CUDA Matrix Operations ===" << std::endl << std::endl;
    
    // Create matrices on CPU
    Matrix A(3, 4, false);  // 3x4 matrix
    Matrix B(4, 3, false);  // 4x3 matrix
    Matrix D(3, 4, false);  // 3x4 matrix for addition
    
    // Fill with sample data
    A.randomFill();
    B.randomFill();
    D.randomFill();
    
    std::cout << "Matrix A (3x4):" << std::endl;
    A.print();
    
    std::cout << "Matrix B (4x3):" << std::endl;
    B.print();
    
    // Copy to GPU
    Matrix* A_gpu = A.toGPU();
    Matrix* B_gpu = B.toGPU();
    Matrix* D_gpu = D.toGPU();
    
    // Matrix multiplication: A * B = C (3x4 * 4x3 = 3x3)
    Matrix* C_gpu = multiply(A_gpu, B_gpu);
    Matrix* C = C_gpu->toCPU();
    
    std::cout << "Result C = A * B (3x3):" << std::endl;
    C->print();
    
    // Matrix addition: A + D = E (3x4 + 3x4 = 3x4)
    Matrix* E_gpu = add(A_gpu, D_gpu);
    Matrix* E = E_gpu->toCPU();
    
    std::cout << "Matrix D (3x4):" << std::endl;
    D.print();
    
    std::cout << "Result E = A + D (3x4):" << std::endl;
    E->print();
    
    // Cleanup
    delete A_gpu;
    delete B_gpu;
    delete C_gpu;
    delete D_gpu;
    delete E_gpu;
    delete C;
    delete E;
    
    std::cout << "Done!" << std::endl;
    
    return 0;
}