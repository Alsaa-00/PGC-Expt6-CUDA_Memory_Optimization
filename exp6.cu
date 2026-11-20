
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define TILE 32
#define BLK 16

#define CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// PART A: Uncoalesced Matrix Transpose
__global__ void transposeNaive(
    const float *in, float *out, int N)
{
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < N && y < N)
        out[x * N + y] = in[y * N + x];
}

// PART A: Coalesced Matrix Transpose
__global__ void transposeCoalesced(
    const float *in, float *out, int N)
{
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < N && y < N)
        tile[threadIdx.y][threadIdx.x] =
            in[y * N + x];

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    if (x < N && y < N)
        out[y * N + x] =
            tile[threadIdx.x][threadIdx.y];
}

// PART B: Naive Matrix Multiplication
__global__ void matMulNaive(
    const float *A,
    const float *B,
    float *C,
    int N)
{
    int row = blockIdx.y * BLK + threadIdx.y;
    int col = blockIdx.x * BLK + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < N; k++)
            sum += A[row * N + k] *
                   B[k * N + col];

        C[row * N + col] = sum;
    }
}

// Timing helper for transpose kernels
float timeTranspose(
    bool coalesced,
    const float *d_in,
    float *d_out,
    int N,
    dim3 grid,
    dim3 block)
{
    cudaEvent_t start, stop;

    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Warm-up
    if (coalesced)
        transposeCoalesced<<<grid, block>>>(d_in, d_out, N);
    else
        transposeNaive<<<grid, block>>>(d_in, d_out, N);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(start));

    int iterations = 10;

    for (int i = 0; i < iterations; i++) {
        if (coalesced)
            transposeCoalesced<<<grid, block>>>(d_in, d_out, N);
        else
            transposeNaive<<<grid, block>>>(d_in, d_out, N);
    }

    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return ms / iterations;
}

// Verify transpose result
bool checkTranspose(
    const std::vector<float>& input,
    const std::vector<float>& output,
    int N)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fabsf(output[j * N + i] -
                      input[i * N + j]) > 1e-5f)
                return false;
        }
    }

    return true;
}

// Verify matrix multiplication using sample elements
bool checkMatMul(
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& C,
    int N)
{
    int samples[] = {
        0, N / 4, N / 2, 3 * N / 4, N - 1
    };

    for (int r : samples) {
        for (int c : samples) {
            float sum = 0.0f;

            for (int k = 0; k < N; k++)
                sum += A[r * N + k] *
                       B[k * N + c];

            if (fabsf(C[r * N + c] - sum) >
                1e-2f * fmaxf(1.0f, fabsf(sum)))
                return false;
        }
    }

    return true;
}

// Main function
int main()
{
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU: %s\n", prop.name);

    // ==================================
    // PART A: MATRIX TRANSPOSE
    // ==================================

    const int N = 4096;
    size_t bytes = (size_t)N * N * sizeof(float);

    std::vector<float> h_in((size_t)N * N);
    std::vector<float> h_out((size_t)N * N);

    for (size_t i = 0; i < (size_t)N * N; i++)
        h_in[i] = static_cast<float>(i % 100) / 100.0f;

    float *d_in, *d_out;

    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));

    CHECK(cudaMemcpy(
        d_in, h_in.data(), bytes,
        cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid(
        (N + TILE - 1) / TILE,
        (N + TILE - 1) / TILE
    );

    printf("\n=== PART A: Matrix Transpose ===\n");

    float naive_ms = timeTranspose(
        false, d_in, d_out, N, grid, block);

    CHECK(cudaMemcpy(
        h_out.data(), d_out, bytes,
        cudaMemcpyDeviceToHost));

    bool naive_pass = checkTranspose(
        h_in, h_out, N);

    float naive_bw =
        (2.0f * N * N * sizeof(float)) /
        (naive_ms * 1e6f);

    printf("Uncoalesced: %.3f ms | %.1f GB/s | %s\n",
        naive_ms, naive_bw,
        naive_pass ? "PASS" : "FAIL");

    float coalesced_ms = timeTranspose(
        true, d_in, d_out, N, grid, block);

    CHECK(cudaMemcpy(
        h_out.data(), d_out, bytes,
        cudaMemcpyDeviceToHost));

    bool coalesced_pass = checkTranspose(
        h_in, h_out, N);

    float coalesced_bw =
        (2.0f * N * N * sizeof(float)) /
        (coalesced_ms * 1e6f);

    printf("Coalesced: %.3f ms | %.1f GB/s | %s\n",
        coalesced_ms, coalesced_bw,
        coalesced_pass ? "PASS" : "FAIL");

    printf("Speedup = %.2fx\n",
        naive_ms / coalesced_ms);

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));

    // ==================================
    // PART B: MATRIX MULTIPLICATION
    // ==================================

    const int M = 1024;
    size_t matBytes = (size_t)M * M * sizeof(float);

    std::vector<float> h_A((size_t)M * M);
    std::vector<float> h_B((size_t)M * M);
    std::vector<float> h_C((size_t)M * M);

    for (size_t i = 0; i < (size_t)M * M; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;

    CHECK(cudaMalloc(&d_A, matBytes));
    CHECK(cudaMalloc(&d_B, matBytes));
    CHECK(cudaMalloc(&d_C, matBytes));

    CHECK(cudaMemcpy(
        d_A, h_A.data(), matBytes,
        cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(
        d_B, h_B.data(), matBytes,
        cudaMemcpyHostToDevice));

    dim3 matBlock(BLK, BLK);
    dim3 matGrid(
        (M + BLK - 1) / BLK,
        (M + BLK - 1) / BLK
    );

    printf("\n=== PART B: Naive Matrix Multiplication ===\n");

    cudaEvent_t start, stop;

    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Warm-up
    matMulNaive<<<matGrid, matBlock>>>(
        d_A, d_B, d_C, M);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    int iterations = 10;

    CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        matMulNaive<<<matGrid, matBlock>>>(
            d_A, d_B, d_C, M);
    }

    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float matmul_ms = 0.0f;

    CHECK(cudaEventElapsedTime(
        &matmul_ms, start, stop));

    matmul_ms /= iterations;

    CHECK(cudaMemcpy(
        h_C.data(), d_C, matBytes,
        cudaMemcpyDeviceToHost));

    bool matmul_pass = checkMatMul(
        h_A, h_B, h_C, M);

    float gflops =
        (2.0f * M * M * M) /
        (matmul_ms * 1e6f);

    printf("Naive MatMul: %.3f ms | %.1f GFLOPS | %s\n",
        matmul_ms, gflops,
        matmul_pass ? "PASS" : "FAIL");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // ==================================
    // SUMMARY
    // ==================================

    printf("\n=== SUMMARY ===\n");

    printf("Transpose Uncoalesced: %.3f ms\n",
        naive_ms);

    printf("Transpose Coalesced: %.3f ms\n",
        coalesced_ms);

    printf("Naive MatMul: %.3f ms\n",
        matmul_ms);

    return 0;
}
