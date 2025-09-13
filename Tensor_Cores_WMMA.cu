// compile with: nvcc -arch=sm_80 -O3 tensor_wmma.cu
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda::wmma;

// Compute C = A*B with WMMA (FP16 input, FP32 accumulation)
__global__ void wmma_gemm_kernel(half *A, half *B, float *C, int M, int N, int K) {
    // assume M,N,K multiples of 16
    int tileM = (blockIdx.y * blockDim.y + threadIdx.y);
    int tileN = (blockIdx.x * blockDim.x + threadIdx.x);

    // each thread-block handles one WMMA tile (example)
    fragment<matrix_a, 16,16,16, half, row_major> a_frag;
    fragment<matrix_b, 16,16,16, half, col_major> b_frag;
    fragment<accumulator,16,16,16,float> c_frag;

    fill_fragment(c_frag, 0.0f);

    // loop over K in 16-step tiles
    for (int k = 0; k < K; k += 16) {
        // load into fragments (use load_matrix_sync)
        load_matrix_sync(a_frag, A + (tileM*16)*K + k, K);
        load_matrix_sync(b_frag, B + k*N + (tileN*16), N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // store result
    store_matrix_sync(C + (tileM*16)*N + (tileN*16), c_frag, N, mem_row_major);
}
