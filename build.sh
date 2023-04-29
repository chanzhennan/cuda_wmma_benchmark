#rm -rf build
#mkdir build && cd build
export PATH=/localdata/zhennanc/sourceCode/cutlass/build/install/lib/cmake/NvidiaCutlass:$PATH
cd build
cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.1 ..
make -j
./cuda_benchmark
