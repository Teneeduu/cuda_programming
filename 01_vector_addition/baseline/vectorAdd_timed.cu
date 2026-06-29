// Timed version of baseline vectorAdd, using N = 1 << 26 (same size as the
// pinned memory example) so the HtoD/DtoH copy times are directly comparable.

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

__global__ void vectorAdd(const int *__restrict a, const int *__restrict b,
                           int *__restrict c, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) c[tid] = a[tid] + b[tid];
}

void verify_result(std::vector<int> &a, std::vector<int> &b,
                    std::vector<int> &c) {
  for (int i = 0; i < a.size(); i++) assert(c[i] == a[i] + b[i]);
}

int main() {
  constexpr int N = 1 << 26;
  constexpr size_t bytes = sizeof(int) * N;

  std::vector<int> a, b, c;
  a.reserve(N);
  b.reserve(N);
  c.reserve(N);
  for (int i = 0; i < N; i++) {
    a.push_back(rand() % 100);
    b.push_back(rand() % 100);
  }

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float ms = 0;

  // Time the Host -> Device copy
  cudaEventRecord(start);
  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  std::cout << "[baseline] HtoD copy time: " << ms << " ms\n";

  int NUM_THREADS = 1 << 10;
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
  vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

  // Time the Device -> Host copy
  cudaEventRecord(start);
  cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  std::cout << "[baseline] DtoH copy time: " << ms << " ms\n";

  verify_result(a, b, c);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::cout << "COMPLETED SUCCESSFULLY\n";
  return 0;
}
