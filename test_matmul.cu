#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

// Function to check CUDA errors
static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleCuResult(CUresult res, const char *file, int line) {
  if (res != CUDA_SUCCESS) {
    const char *errName, *errStr;
    cuGetErrorName(res, &errName);
    cuGetErrorString(res, &errStr);
    std::cerr << "CUDA Driver Error: " << errName << " - " << errStr << " in "
              << file << " at line " << line << std::endl;
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_CU_ERROR(err) (HandleCuResult(err, __FILE__, __LINE__))

// Read PTX code from a file
std::string load_ptx(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// Basic matrix multiplication for validation
void cpu_matmul(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

int main() {
  int M = 128;
  int N = 256;
  int K = 64;

  const int BLOCK_M = 64;
  const int BLOCK_N = 16;
  const int BLOCK_K = 32;

  int threadsPerBlock = 128;
  int blocksPerGrid =
      int((M + BLOCK_M - 1) / BLOCK_M) * int((N + BLOCK_N - 1) / BLOCK_N);

  // Other kernel parameters
  int stride_am = K;
  int stride_ak = 1;

  int stride_bk = N;
  int stride_bn = 1;

  int stride_zm = N;
  int stride_zn = 1;

  // Allocate and initialize matrices
  std::vector<float> a(M * K, 1.0f);
  std::vector<float> b(K * N, 1.0f);
  std::vector<float> c(M * N, 0.0f);
  std::vector<float> c_ref(M * N, 0.0f);

  // Allocate device memory
  float *a_dev, *b_dev, *c_dev;
  HANDLE_ERROR(cudaMalloc((void **)&a_dev, M * K * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&b_dev, K * N * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&c_dev, M * N * sizeof(float)));

  // Copy data to device
  HANDLE_ERROR(cudaMemcpy(a_dev, a.data(), M * K * sizeof(float),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(b_dev, b.data(), K * N * sizeof(float),
                          cudaMemcpyHostToDevice));

  // Prepare kernel arguments
  void *args[] = {&a_dev,     &b_dev, &c_dev, &stride_am, &stride_bk,
                  &stride_zm, &M,     &N,     &K};

  // Load the PTX code from file
  std::string ptx = load_ptx("./matmul.ptx");

  // Load the PTX and get the kernel function
  CUmodule module;
  CUfunction function;
  HANDLE_CU_ERROR(cuInit(0));
  CUcontext context;
  CUdevice device;
  HANDLE_CU_ERROR(cuDeviceGet(&device, 0));
  HANDLE_CU_ERROR(cuCtxCreate(&context, 0, device));
  HANDLE_CU_ERROR(
      cuModuleLoadDataEx(&module, ptx.c_str(), 0, nullptr, nullptr));
  HANDLE_CU_ERROR(cuModuleGetFunction(&function, module, "matmul_kernel"));

  // Launch the kernel
  uint32_t sharedMem = 1024 * 32;
  std::cout << "blocksPerGrid = " << blocksPerGrid
            << " threadsPerBlock = " << threadsPerBlock
            << " sharedMem = " << sharedMem << std::endl;
  HANDLE_CU_ERROR(cuLaunchKernel(function, blocksPerGrid, 1, 1, threadsPerBlock,
                                 1, 1, sharedMem, NULL, args, NULL));
  HANDLE_CU_ERROR(cuCtxSynchronize());
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Copy result back to host
  HANDLE_ERROR(cudaMemcpy(c.data(), c_dev, M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

  // Verify the result on the CPU
  cpu_matmul(a, b, c_ref, M, N, K);

  // Compare CPU and GPU results
  bool correct = true;
  for (int i = 0; i < M * N; ++i) {
    if (fabs(c_ref[i] - c[i]) > 1e-5) {
      correct = false;
      break;
    }
  }

  std::cout << "Matrix multiplication " << (correct ? "PASSED" : "FAILED")
            << std::endl;

  // Cleanup
  HANDLE_ERROR(cudaFree(a_dev));
  HANDLE_ERROR(cudaFree(b_dev));
  HANDLE_ERROR(cudaFree(c_dev));
  HANDLE_CU_ERROR(cuModuleUnload(module));
  HANDLE_CU_ERROR(cuCtxDestroy(context));

  return 0;
}
