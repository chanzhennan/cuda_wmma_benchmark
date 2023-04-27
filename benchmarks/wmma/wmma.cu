#include <mma.h>

#include <cassert>
#include <cstdio>
#include <functional>
#include <iostream>
#include <vector>

#include "cutlass/gemm/device/gemm.h"
#include "wmma/wmma.cuh"

#define PAD(X, Y) (X % Y ? (X / Y + 1) * Y : X)
#define WARP_SIZE 32
#define BLOCK_DIM_DEFAULT 512

template <typename TIN, typename TOUT, int M_TILE, int N_TILE, int K_TILE>
__global__ void wmma_kernel(TIN *a, TIN *b, TOUT *c, int M_PAD, int N_PAD,
                            int K_PAD) {
  int idx, midx, nidx, ndim, kdim;
  ndim = N_PAD / N_TILE;
  kdim = K_PAD / K_TILE;
  idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  nidx = idx % ndim;
  midx = idx / ndim;
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

  TOUT *c_unique = c + nidx * N_TILE + midx * M_TILE * ndim * N_TILE;

  for (int kidx = 0; kidx < kdim; kidx++) {
    // Load the inputs
    TIN *a_unique = a + kidx * K_TILE + midx * M_TILE * kdim * K_TILE;
    TIN *b_unique = b + nidx * N_TILE + kidx * K_TILE * ndim * N_TILE;

    nvcuda::wmma::load_matrix_sync(a_frag, a_unique, K_PAD);
    nvcuda::wmma::load_matrix_sync(b_frag, b_unique, N_PAD);

    // Perform the matrix multiplication
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  // Store the output
  nvcuda::wmma::store_matrix_sync(c_unique, c_frag, N_PAD,
                                  nvcuda::wmma::mem_row_major);
}

__global__ void float32to16(float *d_src, half *d_dst, size_t len) {
  long int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid > len) return;

  d_dst[tid] = d_src[tid];
}

template <typename TIN, typename TGEMMIN>
void wmma_load(TIN *dA, TIN *dB, TGEMMIN *dA_f, TGEMMIN *dB_f, int m, int n,
               int k) {
  assert(m != 0 && n != 0 && k != 0);

  // padding
  // int M_PAD = PAD(m, M_TILE);
  // int N_PAD = PAD(n, N_TILE);
  // int K_PAD = PAD(k, K_TILE);

  int TPB = 256;
  int BLOCK1 = (m * k + TPB - 1) / TPB;
  int BLOCK2 = (n * k + TPB - 1) / TPB;
  float32to16<<<BLOCK1, TPB>>>(dA, dA_f, m * k);
  cudaDeviceSynchronize();
  float32to16<<<BLOCK2, TPB>>>(dB, dB_f, n * k);
  cudaDeviceSynchronize();
}

template <typename TIN, typename TOUT, typename TGEMMIN, typename TGEMMOUT,
          int M_TILE, int N_TILE, int K_TILE>
void wmma_gemm(TGEMMIN *dA_f, TGEMMIN *dB_f, TOUT *dC, int m, int n, int k) {
  assert(m != 0 && n != 0 && k != 0);

  // padding
  int M_PAD = PAD(m, M_TILE);
  int N_PAD = PAD(n, N_TILE);
  int K_PAD = PAD(k, K_TILE);

  int GRID_DIM, BLOCK_DIM, nwarp;
  nwarp = (M_PAD / M_TILE) * (N_PAD / N_TILE);
  if (nwarp * WARP_SIZE < BLOCK_DIM_DEFAULT) {
    GRID_DIM = 1;
    BLOCK_DIM = nwarp * WARP_SIZE;
  } else {
    GRID_DIM = (nwarp * WARP_SIZE) % BLOCK_DIM_DEFAULT
                   ? nwarp * WARP_SIZE / BLOCK_DIM_DEFAULT + 1
                   : nwarp * WARP_SIZE / BLOCK_DIM_DEFAULT;
    BLOCK_DIM = BLOCK_DIM_DEFAULT;
  }
  // printf("GRID_DIM:%d BLOCK_DIM:%d\n", GRID_DIM, BLOCK_DIM);
  wmma_kernel<half, float, M_TILE, N_TILE, K_TILE>
      <<<GRID_DIM, BLOCK_DIM>>>(dA_f, dB_f, dC, M_PAD, N_PAD, K_PAD);

  cudaDeviceSynchronize();  // sync for unify memory
}

template void wmma_gemm<float, float>(half *dA_f, half *dB_f, float *dC, int m,
                                      int n, int k);
template void wmma_load<float>(float *dA, float *dB, half *dA_f, half *dB_f,
                               int m, int n, int k);
