add_subdirectory(bm_lib)

set(SOURCE_FILES cuda_benchmark.cc)

list(APPEND SOURCE_FILES cutlass/cutlass.cc cutlass/cutlass.cu
     # wmma/wmma.cc wmma/wmma.cu
)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-std=c++17 -arch=sm_89")
cuda_add_executable(cuda_benchmark ${SOURCE_FILES})
set_property(TARGET cuda_benchmark PROPERTY CUDA_ARCHITECTURES 89)
target_link_libraries(
  cuda_benchmark bm_lib nvidia::cutlass::cutlass nvidia::cutlass::library
  nvidia::cutlass::tools::util nvidia::cutlass::library_static)

install(TARGETS cuda_benchmark RUNTIME)
