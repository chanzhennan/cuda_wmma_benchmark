// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
#include "bmma/bmma.cuh"

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_lib/utils.h"

template <typename TIN, typename TOUT>
class Bmma : public benchmark::Fixture {
 public:
  void callKernel(benchmark::State &state) {
    // call kernel
    bmma_gemm<float, float>(dA_f, dB_f, C, M, N, K);
  }

  void SetUp(const ::benchmark::State &state) BENCHMARK_OVERRIDE {
    dataSize = state.range(0) * state.range(0);
    M = state.range(0);
    N = state.range(0);
    K = state.range(0);

    // Populate array
    cudaMallocManaged(&A, sizeof(TIN) * dataSize);
    cudaMallocManaged(&B, sizeof(TIN) * dataSize);
    cudaMallocManaged(&C, sizeof(TOUT) * dataSize);

    cudaMallocManaged(&dA_f, sizeof(half) * dataSize);
    cudaMallocManaged(&dB_f, sizeof(half) * dataSize);

    cudabm::genRandom(A, dataSize);
    cudabm::genRandom(B, dataSize);

    bmma_load<TIN>(A, B, dA_f, dB_f, M, N, K);
  }

  void TearDown(const ::benchmark::State &st) BENCHMARK_OVERRIDE {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(dA_f);
    cudaFree(dB_f);
  }

  double getDataSize() { return (double)dataSize; }

 private:
  TIN *A, *B;
  half *dA_f, *dB_f;
  TOUT *C;

  int M;
  int N;
  int K;
  long int dataSize = 0;
};

#define BENCHMARK_BMMA_OP(name, dType1, dType2)                        \
  BENCHMARK_TEMPLATE_DEFINE_F(Bmma, name, dType1, dType2)              \
  (benchmark::State & st) {                                            \
    for (auto _ : st) {                                                \
      callKernel(st);                                                  \
    }                                                                  \
    st.counters["DATASIZE"] = getDataSize();                           \
    st.counters["FLOPS"] = benchmark::Counter{                         \
        getDataSize(), benchmark::Counter::kIsIterationInvariantRate}; \
  }                                                                    \
  BENCHMARK_REGISTER_F(Bmma, name)                                     \
      ->Unit(benchmark::kMillisecond)                                  \
      ->RangeMultiplier(2)                                             \
      ->Range(1024, 2048);

#define BENCHMARK_BMMA_OP_TYPE(dType1, dType2) \
  BENCHMARK_BMMA_OP(Bmma_##dType1, dType1, dType2)

BENCHMARK_BMMA_OP_TYPE(float, float)
