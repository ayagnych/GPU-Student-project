#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cuda_runtime.h>
#include <random>
#include <cmath>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

float trace(const std::vector<float>& matrix, int N) {
    float tr = 0.0f;
    for (int i = 0; i < N; ++i) {
        tr += matrix[i * N + i];
    }
    return tr;
}

__global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
__global__ void matrixAddKernel(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < N && j < N) {
        C[j * N + i] = A[j * N + i] + B[j * N + i];
    }
}

__global__ void matrixSubKernel(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < N && j < N) {
        C[j * N + i] = A[j * N + i] - B[j * N + i];
    }
}

void strassenMultiply(float* A, float* B, float* C, int N, int dim);

void strassenRecursive(float* A, float* B, float* C, int N) {
    if (N <= 64) {
        dim3 blockSize(16, 16);
        dim3 gridSize((N + 15) / 16, (N + 15) / 16);
        matrixMulKernel<<<gridSize, blockSize>>>(A, B, C, N);
        return;
    }
    
    int newSize = N / 2;
    int bytes = newSize * newSize * sizeof(float);
    
    float *A11, *A12, *A21, *A22;
    float *B11, *B12, *B21, *B22;
    float *C11, *C12, *C21, *C22;
    float *M1, *M2, *M3, *M4, *M5, *M6, *M7;
    float *T1, *T2;
    
    CHECK_CUDA(cudaMalloc(&A11, bytes));
    CHECK_CUDA(cudaMalloc(&A12, bytes));
    CHECK_CUDA(cudaMalloc(&A21, bytes));
    CHECK_CUDA(cudaMalloc(&A22, bytes));
    
    CHECK_CUDA(cudaMalloc(&B11, bytes));
    CHECK_CUDA(cudaMalloc(&B12, bytes));
    CHECK_CUDA(cudaMalloc(&B21, bytes));
    CHECK_CUDA(cudaMalloc(&B22, bytes));
    
    CHECK_CUDA(cudaMalloc(&C11, bytes));
    CHECK_CUDA(cudaMalloc(&C12, bytes));
    CHECK_CUDA(cudaMalloc(&C21, bytes));
    CHECK_CUDA(cudaMalloc(&C22, bytes));
    
    CHECK_CUDA(cudaMalloc(&M1, bytes));
    CHECK_CUDA(cudaMalloc(&M2, bytes));
    CHECK_CUDA(cudaMalloc(&M3, bytes));
    CHECK_CUDA(cudaMalloc(&M4, bytes));
    CHECK_CUDA(cudaMalloc(&M5, bytes));
    CHECK_CUDA(cudaMalloc(&M6, bytes));
    CHECK_CUDA(cudaMalloc(&M7, bytes));
    
    CHECK_CUDA(cudaMalloc(&T1, bytes));
    CHECK_CUDA(cudaMalloc(&T2, bytes));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((newSize + 15) / 16, (newSize + 15) / 16);
    
    // разобьем матрицы на подматрицы
    for (int i = 0; i < newSize; i++) {
        CHECK_CUDA(cudaMemcpy(A11 + i * newSize, A + i * N, newSize * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(A12 + i * newSize, A + i * N + newSize, newSize * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(A21 + i * newSize, A + (i + newSize) * N, newSize * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(A22 + i * newSize, A + (i + newSize) * N + newSize, newSize * sizeof(float), cudaMemcpyDeviceToDevice));
        
        CHECK_CUDA(cudaMemcpy(B11 + i * newSize, B + i * N, newSize * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(B12 + i * newSize, B + i * N + newSize, newSize * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(B21 + i * newSize, B + (i + newSize) * N, newSize * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(B22 + i * newSize, B + (i + newSize) * N + newSize, newSize * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    
    matrixAddKernel<<<gridSize, blockSize>>>(A11, A22, T1, newSize);
    matrixAddKernel<<<gridSize, blockSize>>>(B11, B22, T2, newSize);
    strassenMultiply(T1, T2, M1, newSize, newSize);
    
    matrixAddKernel<<<gridSize, blockSize>>>(A21, A22, T1, newSize);
    strassenMultiply(T1, B11, M2, newSize, newSize);
    
    matrixSubKernel<<<gridSize, blockSize>>>(B12, B22, T1, newSize);
    strassenMultiply(A11, T1, M3, newSize, newSize);
    
    matrixSubKernel<<<gridSize, blockSize>>>(B21, B11, T1, newSize);
    strassenMultiply(A22, T1, M4, newSize, newSize);
    
    matrixAddKernel<<<gridSize, blockSize>>>(A11, A12, T1, newSize);
    strassenMultiply(T1, B22, M5, newSize, newSize);
    
    matrixSubKernel<<<gridSize, blockSize>>>(A21, A11, T1, newSize);
    matrixAddKernel<<<gridSize, blockSize>>>(B11, B12, T2, newSize);
    strassenMultiply(T1, T2, M6, newSize, newSize);
    
    matrixSubKernel<<<gridSize, blockSize>>>(A12, A22, T1, newSize);
    matrixAddKernel<<<gridSize, blockSize>>>(B21, B22, T2, newSize);
    strassenMultiply(T1, T2, M7, newSize, newSize);
    
    matrixAddKernel<<<gridSize, blockSize>>>(M1, M4, T1, newSize);
    matrixSubKernel<<<gridSize, blockSize>>>(T1, M5, T2, newSize);
    matrixAddKernel<<<gridSize, blockSize>>>(T2, M7, C11, newSize);
    
    matrixAddKernel<<<gridSize, blockSize>>>(M3, M5, C12, newSize);
    
    matrixAddKernel<<<gridSize, blockSize>>>(M2, M4, C21, newSize);
    
    matrixAddKernel<<<gridSize, blockSize>>>(M1, M3, T1, newSize);
    matrixSubKernel<<<gridSize, blockSize>>>(T1, M2, T2, newSize);
    matrixAddKernel<<<gridSize, blockSize>>>(T2, M6, C22, newSize);
    
    // комбинируем подматрицы в С
    for (int i = 0; i < newSize; i++) {
        CHECK_CUDA(cudaMemcpy(C + i * N, C11 + i * newSize, newSize * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(C + i * N + newSize, C12 + i * newSize, newSize * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(C + (i + newSize) * N, C21 + i * newSize, newSize * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(C + (i + newSize) * N + newSize, C22 + i * newSize, newSize * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    
    cudaFree(A11); cudaFree(A12); cudaFree(A21); cudaFree(A22);
    cudaFree(B11); cudaFree(B12); cudaFree(B21); cudaFree(B22);
    cudaFree(C11); cudaFree(C12); cudaFree(C21); cudaFree(C22);
    cudaFree(M1); cudaFree(M2); cudaFree(M3); cudaFree(M4);
    cudaFree(M5); cudaFree(M6); cudaFree(M7);
    cudaFree(T1); cudaFree(T2);
}

void strassenMultiply(float* A, float* B, float* C, int N, int dim) {
    if (N != dim) {
        assert("Matrix size must be power of two" && false);
    }
    strassenRecursive(A, B, C, N);
}

void classicalMatrixMultiply(float* A, float* B, float* C, int N) {
    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (N + 15) / 16);
    matrixMulKernel<<<gridSize, blockSize>>>(A, B, C, N);
}


int main() {
    const int N = 1024;
    size_t bytes = N * N * sizeof(float);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = dis(gen);
        h_B[i] = dis(gen);
    }
    
    std::vector<float> h_C_classical(N * N);
    std::vector<float> h_C_strassen(N * N);

    float trace_A = trace(h_A, N);
    float trace_B = trace(h_B, N);

    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "Trace of matrix A: " << trace_A << std::endl;
    std::cout << "Trace of matrix B: " << trace_B << std::endl;


    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
    
    // обычное умножение
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    classicalMatrixMultiply(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float classicalTime;
    CHECK_CUDA(cudaEventElapsedTime(&classicalTime, start, stop));
    CHECK_CUDA(cudaMemcpy(h_C_classical.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    
    // умножение Штрассеном
    CHECK_CUDA(cudaEventRecord(start));
    strassenMultiply(d_A, d_B, d_C, N, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float strassenTime;
    CHECK_CUDA(cudaEventElapsedTime(&strassenTime, start, stop));
    CHECK_CUDA(cudaMemcpy(h_C_strassen.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    
    bool valid = true;
    float trace_classical = trace(h_C_classical, N);
    float trace_strassen = trace(h_C_strassen, N);
    if (fabs(trace_classical -  trace_strassen) < 1e-3){
	valid = false;
    }
    std::cout << "Classical time: " << classicalTime << " ms" << std::endl;
    std::cout << "Strassen time: " << strassenTime << " ms" << std::endl;
    std::cout << "Trace of classical result: " << trace_classical << std::endl;
    std::cout << "Trace of Strassen result: " << trace_strassen << std::endl;
    std::cout << "Results match: " << (valid ? "Yes" : "No") << std::endl;


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}