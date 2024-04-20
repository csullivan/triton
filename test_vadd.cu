#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#define HANDLE_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

std::string load_ptx(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Cannot open PTX file: " << filename << std::endl;
        exit(1);
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

int main() {
  int N = 1024;
  float *a_dev, *b_dev, *c_dev;

  HANDLE_ERROR(cudaMalloc(&a_dev, N * sizeof(float)));
  HANDLE_ERROR(cudaMalloc(&b_dev, N * sizeof(float)));
  HANDLE_ERROR(cudaMalloc(&c_dev, N * sizeof(float)));

  std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N, 0.0f);
  HANDLE_ERROR(
      cudaMemcpy(a_dev, a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(b_dev, b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  std::string ptx = load_ptx("./vadd.ptx");
  std::cout << ptx << std::endl;
  CUmodule cuModule;
  CUfunction cuFunction;
  cuInit(0);
  CUdevice cuDevice;
  cuDeviceGet(&cuDevice, 0);
  CUcontext cuContext;
  cuCtxCreate(&cuContext, 0, cuDevice);
  CUresult res = cuModuleLoadDataEx(&cuModule, ptx.c_str(), 0, 0, 0);
  if (res != CUDA_SUCCESS) {
    std::cerr << "Failed to load module: " << res << std::endl;
    exit(1);
  }
  cuModuleGetFunction(&cuFunction, cuModule, "add_kernel");

  void *args[] = {&a_dev, &b_dev, &c_dev, reinterpret_cast<void *>(&N)};
  int threadsPerBlock = 128;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  cuLaunchKernel(cuFunction, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0,
                 NULL, args, NULL);
  cuCtxSynchronize();

  HANDLE_ERROR(
      cudaMemcpy(c.data(), c_dev, N * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i) {
    std::cout << "c[" << i << "] = " << c[i] << std::endl;
  }

  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);
  cuModuleUnload(cuModule);
  cuCtxDestroy(cuContext);

  return 0;
}
