// Timed version of pinned-memory vectorAdd, using CUDA events to measure
// the HtoD/DtoH copy times directly, plus one full-pipeline timer
// (HtoD + kernel + DtoH) so the total is directly comparable to
// um_prefetch's "full pipeline time" number.

#include <cassert>
#include <iostream>

__global__ void vectorAdd(int *a, int *b, int *c, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) c[tid] = a[tid] + b[tid];
}

void verify_result(int *a, int *b, int *c, int N) {
  for (int i = 0; i < N; i++) assert(c[i] == a[i] + b[i]);
}

int main() {
  constexpr int N = 1 << 26;
  size_t bytes = sizeof(int) * N;

  int *h_a, *h_b, *h_c;
  cudaMallocHost(&h_a, bytes);
  cudaMallocHost(&h_b, bytes);
  cudaMallocHost(&h_c, bytes);

  for (int i = 0; i < N; i++) {
    h_a[i] = rand() % 100;
    h_b[i] = rand() % 100;
  }

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaEvent_t start, stop, full_start, full_stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&full_start);
  cudaEventCreate(&full_stop);
  float ms = 0;

  // ---- full pipeline timing starts: HtoD + kernel + DtoH ----
  cudaEventRecord(full_start);

  // Time the Host -> Device copy
  cudaEventRecord(start);
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  std::cout << "[pinned] HtoD copy time: " << ms << " ms\n";

  int NUM_THREADS = 1 << 10;
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
  vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

  // Time the Device -> Host copy
  cudaEventRecord(start);
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  std::cout << "[pinned] DtoH copy time: " << ms << " ms\n";

  cudaEventRecord(full_stop);
  cudaEventSynchronize(full_stop);
  // ---- full pipeline timing ends ----

  float full_ms = 0;
  cudaEventElapsedTime(&full_ms, full_start, full_stop);
  std::cout << "[pinned] full pipeline time (HtoD+kernel+DtoH): " << full_ms
            << " ms\n";

  verify_result(h_a, h_b, h_c, N);

  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaEventDestroy(full_start);
  cudaEventDestroy(full_stop);

  std::cout << "COMPLETED SUCCESSFULLY\n";
  return 0;
}
