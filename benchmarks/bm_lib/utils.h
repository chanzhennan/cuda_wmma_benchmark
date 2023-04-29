// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.

// #include <cublas.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#define PAD(X, Y) (X % Y ? (X / Y + 1) * Y : X)
#define WARP_SIZE 32
#define BLOCK_DIM_DEFAULT 512

namespace cudabm {

// benchmark string helper
std::string strFormat(const char* format, ...);

void genRandom(std::vector<float>& vec);
void genRandom(float* vec, size_t len);
void Print(float* vec, size_t len);
float Sum(float* vec, size_t len);

void Gemm(float* dA, float* dB, float* dC, int m, int n, int k);

template <typename Type>
bool Equal(const unsigned int n, const Type* x, const Type* y,
           const Type tolerance);

// unify memory
template <typename T>
struct cuda_data {
  T* data;

  cuda_data(size_t n) {
    cudaMallocManaged(&data, sizeof(T) * n);
    // init to zero
    for (long i = 0; i < n; i++) {
      data[i] = 0;
    }
  }
  ~cuda_data() { cudaFree(data); }
};

}  // namespace cudabm
