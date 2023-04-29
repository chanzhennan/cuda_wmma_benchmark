
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename TIN, typename TOUT, typename TGEMMIN = half,
          typename TGEMMOUT = float, int M_TILE = 16, int N_TILE = 16,
          int K_TILE = 16>
void bmma_gemm(TGEMMIN *dA_f, TGEMMIN *dB_f, TOUT *dC, int m, int n, int k);

template <typename TIN, typename TGEMMIN = half>
void bmma_load(TIN *dA, TIN *dB, TGEMMIN *dA_f, TGEMMIN *dB_f, int m, int n,
               int k);
