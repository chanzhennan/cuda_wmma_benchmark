#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

template <typename T>
void cutlass_gemm(T *dA, T *dB, T *dC, int m, int n, int k);
