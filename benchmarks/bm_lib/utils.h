// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.

#include <cublas.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>
namespace cudabm {

// benchmark string helper
std::string strFormat(const char* format, ...);

void genRandom(std::vector<float>& vec);
void genRandom(float* vec, size_t len);
void Print(float* vec, size_t len);
float Sum(float* vec, size_t len);

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
