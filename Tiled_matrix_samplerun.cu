// tiled_gemm.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define TILE 32

// Kernel: C = A * B   (row-major)
__global__
void matmul_tiled(const float* __restrict__ A,
                  const float* __restrict__ B,
                  float* __restrict__ C,
                  int M, int N, int K)
{
    // Block tile indices
    int by = blockIdx.y;
    int bx = blockIdx.x;

    // Thread indices within block
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Global row/col indices
    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    // Shared-memory tiles for A and B
    __shared__ float sA[TILE][TILE+1]; // pad to avoid bank conflicts
    __shared__ float sB[TILE][TILE+1];

    float sum = 0.0f;

    // Loop over tiles of K dimension
    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;

        // Load A tile element (row, aCol)
        if (row < M && aCol < K) {
            sA[ty][tx] = A[row * K + aCol];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load B tile element (bRow, col)
        if (bRow < K && col < N) {
            sB[ty][tx] = B[bRow * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    // Write output
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// simple CPU reference for correctness
void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i=0;i<M;i++) for (int j=0;j<N;j++){
        double s=0.0;
        for (int k=0;k<K;k++) s += double(A[i*K+k]) * double(B[k*N+j]);
        C[i*N+j] = (float)s;
    }
}

int main(int argc, char** argv) {
    int M = 1024;
    int K = 1024;
    int N = 1024;
    if (argc==4) { M=atoi(argv[1]); K=atoi(argv[2]); N=atoi(argv[3]); }

    size_t szA = (size_t)M * K;
    size_t szB = (size_t)K * N;
    size_t szC = (size_t)M * N;

    float *hA = (float*)malloc(szA * sizeof(float));
    float *hB = (float*)malloc(szB * sizeof(float));
    float *hC = (float*)malloc(szC * sizeof(float));
    float *hC_ref = (float*)malloc(szC * sizeof(float));

    // Init
    for (size_t i=0;i<szA;i++) hA[i] = (float)(drand48() - 0.5);
    for (size_t i=0;i<szB;i++) hB[i] = (float)(drand48() - 0.5);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, szA * sizeof(float));
    cudaMalloc(&dB, szB * sizeof(float));
    cudaMalloc(&dC, szC * sizeof(float));

    cudaMemcpy(dA, hA, szA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, szB * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid( (N + TILE - 1)/TILE, (M + TILE - 1)/TILE );

    // Warmup
    matmul_tiled<<<grid, block>>>(dA,dB,dC,M,N,K);
    cudaDeviceSynchronize();

    // Timing
    cudaEvent_t st, ed;
    cudaEventCreate(&st); cudaEventCreate(&ed);
    cudaEventRecord(st);
    matmul_tiled<<<grid, block>>>(dA,dB,dC,M,N,K);
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms=0; cudaEventElapsedTime(&ms, st, ed);
    double gflops = 2.0 * (double)M * N * K / (ms * 1e6);

    printf("Tiled GEMM %dx%dx%d: %f ms, %f GFLOPS\n", M,K,N, ms, gflops);

    cudaMemcpy(hC, dC, szC * sizeof(float), cudaMemcpyDeviceToHost);

    // verify a few elements with CPU (or compute full small reference)
    if (M <= 1024 && N <= 1024 && K <= 1024 && (M*N) <= (1024*1024)) {
        cpu_gemm(hA,hB,hC_ref,M,N,K);
        // compute max error
        float maxerr=0, rms=0;
        for (size_t i=0;i<szC;i++){
            float e = hC_ref[i] - hC[i];
            maxerr = fmaxf(maxerr, fabsf(e));
            rms += double(e)*double(e);
        }
        rms = sqrt(rms / szC);
        printf("maxerr=%g rms=%g\n", maxerr, rms);
    } else {
        printf("Skipped full CPU verification (size large)\n");
    }

    // cleanup
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hC_ref);
    return 0;
}
