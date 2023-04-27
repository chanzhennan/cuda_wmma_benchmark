// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
#include "cutlass/cutlass.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_lib/utils.h"

template <typename TIN, typename TOUT>
class Cutlass : public benchmark::Fixture {
 public:
  void callKernel(benchmark::State &state) {
    // call kernel
    cutlass_gemm(A, B, C, M, N, K);
  }

  void SetUp(const ::benchmark::State &state) BENCHMARK_OVERRIDE {
    dataSize = state.range(0) * state.range(0);
    M = state.range(0);
    N = state.range(0);
    K = state.range(0);

    // Populate array
    cudaMallocManaged((void **)&A, sizeof(TIN) * dataSize);
    cudaMallocManaged((void **)&B, sizeof(TIN) * dataSize);
    cudaMallocManaged((void **)&C, sizeof(TIN) * dataSize);

    cudabm::genRandom(A, dataSize);
    cudabm::genRandom(B, dataSize);
  }

  void TearDown(const ::benchmark::State &st) BENCHMARK_OVERRIDE {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
  }

  double getDataSize() { return (double)dataSize; }

 private:
  TIN *A, *dA;
  TIN *B, *dB;
  TIN *C, *dC;
  int M;
  int N;
  int K;
  long int dataSize;
};

#define BENCHMARK_CUTLASS_OP(name, dType1, dType2)                     \
  BENCHMARK_TEMPLATE_DEFINE_F(Cutlass, name, dType1, dType2)           \
  (benchmark::State & st) {                                            \
    for (auto _ : st) {                                                \
      callKernel(st);                                                  \
    }                                                                  \
    st.counters["DATASIZE"] = getDataSize();                           \
    st.counters["FLOPS"] = benchmark::Counter{                         \
        getDataSize(), benchmark::Counter::kIsIterationInvariantRate}; \
  }                                                                    \
  BENCHMARK_REGISTER_F(Cutlass, name)                                  \
      ->Unit(benchmark::kMillisecond)                                  \
      ->RangeMultiplier(2)                                             \
      ->Iterations(1)                                                  \
      ->Range(1024, 2048);

#define BENCHMARK_CUTLASS_OP_TYPE(dType1, dType2) \
  BENCHMARK_CUTLASS_OP(CUTLASS_##dType1, dType1, dType2)

BENCHMARK_CUTLASS_OP_TYPE(float, float)
// BENCHMARK_GEMM1_OP_TYPE(int)
