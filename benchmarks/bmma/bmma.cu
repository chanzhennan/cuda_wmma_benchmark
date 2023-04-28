#include <mma.h>

#include <cassert>
#include <cstdio>
#include <functional>
#include <iostream>
#include <vector>

#include "bm_lib/utils.h"
#include "bmma/bmma.cuh"
#include "cutlass/gemm/device/gemm.h"

template <typename TIN, typename TOUT, int M_TILE, int N_TILE, int K_TILE>
__global__ void bmma_kernel(TIN *a, TIN *b, TOUT *c, int M_PAD, int N_PAD,
                            int K_PAD) {
  const int nwarp = BLOCK_DIM_DEFAULT / WARP_SIZE;
  const int C_TILE_SIZE = M_TILE * N_TILE;
  __shared__ TOUT shm[M_TILE][nwarp * N_TILE];
  const int ndim = N_PAD / N_TILE;
  const int kdim = K_PAD / K_TILE;
  const int warpidx = threadIdx.x / WARP_SIZE;
  const int nidx = blockIdx.x % ndim;
  const int midx = blockIdx.x / ndim;
  // Declare the fragments
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M_TILE, N_TILE, K_TILE, TIN,
                         nvcuda::wmma::row_major>
      a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M_TILE, N_TILE, K_TILE, TIN,
                         nvcuda::wmma::row_major>
      b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M_TILE, N_TILE, K_TILE,
                         TOUT>
      c_frag;

  // Initialize the output to zero
  nvcuda::wmma::fill_fragment(c_frag, 0.0f);

  const int base = nidx * N_TILE + midx * ndim * C_TILE_SIZE;
  TOUT *c_unique = c + base;

  for (int kidx = 0; kidx < kdim; kidx++) {
    if (kidx % nwarp != warpidx) continue;
    // Load the inputs
    TIN *a_unique = a + kidx * K_TILE + midx * M_TILE * kdim * K_TILE;
    TIN *b_unique = b + nidx * N_TILE + kidx * K_TILE * ndim * N_TILE;

    nvcuda::wmma::load_matrix_sync(a_frag, a_unique, K_PAD);
    nvcuda::wmma::load_matrix_sync(b_frag, b_unique, N_PAD);

    // Perform the matrix multiplication
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  // Store the output
  nvcuda::wmma::store_matrix_sync(&shm[0][warpidx * N_TILE], c_frag,
                                  nwarp * N_TILE, nvcuda::wmma::mem_row_major);
  __syncthreads();
  for (int i = warpidx; i < C_TILE_SIZE; i += nwarp) {
    c_unique[i / N_TILE * ndim * N_TILE + i % N_TILE] = 0;
    for (int j = 0; j < nwarp; j++) {
      c_unique[i / N_TILE * ndim * N_TILE + i % N_TILE] +=
          shm[i / N_TILE][i % N_TILE + j * N_TILE];
    }
  }
}

__global__ void float32to16_b(float *d_src, half *d_dst, size_t len) {
  long int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid > len) return;
  d_dst[tid] = d_src[tid];
}

template <typename TIN, typename TGEMMIN>
void bmma_load(TIN *dA, TIN *dB, TGEMMIN *dA_f, TGEMMIN *dB_f, int m, int n,
               int k) {
  assert(m != 0 && n != 0 && k != 0);

  // padding
  // int M_PAD = PAD(m, M_TILE);
  // int N_PAD = PAD(n, N_TILE);
  // int K_PAD = PAD(k, K_TILE);

  int TPB = 256;
  int BLOCK1 = (m * k + TPB - 1) / TPB;
  int BLOCK2 = (n * k + TPB - 1) / TPB;
  float32to16_b<<<BLOCK1, TPB>>>(dA, dA_f, m * k);
  cudaDeviceSynchronize();
  float32to16_b<<<BLOCK2, TPB>>>(dB, dB_f, n * k);
  cudaDeviceSynchronize();
}

template <typename TIN, typename TOUT, typename TGEMMIN, typename TGEMMOUT,
          int M_TILE, int N_TILE, int K_TILE>
void bmma_gemm(TGEMMIN *dA_f, TGEMMIN *dB_f, TOUT *dC, int m, int n, int k) {
  assert(m != 0 && n != 0 && k != 0);

  // padding
  int M_PAD = PAD(m, M_TILE);
  int N_PAD = PAD(n, N_TILE);
  int K_PAD = PAD(k, K_TILE);

  int GRID_DIM, BLOCK_DIM;
  GRID_DIM = (M_PAD / M_TILE) * (N_PAD / N_TILE);
  BLOCK_DIM = BLOCK_DIM_DEFAULT;
  //    printf("GRID_DIM:%d BLOCK_DIM:%d\n",GRID_DIM,BLOCK_DIM);

  bmma_kernel<TGEMMIN, TGEMMOUT, M_TILE, N_TILE, K_TILE>
      <<<GRID_DIM, BLOCK_DIM>>>(dA_f, dB_f, dC, M_PAD, N_PAD, K_PAD);

  cudaDeviceSynchronize();  // sync for unify memory
}

template void bmma_gemm<float, float>(half *dA_f, half *dB_f, float *dC, int m,
                                      int n, int k);
template void bmma_load<float>(float *dA, float *dB, half *dA_f, half *dB_f,
                               int m, int n, int k);
