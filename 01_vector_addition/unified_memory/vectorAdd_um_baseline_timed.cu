// Timed version of unified memory baseline. N bumped up to 1<<26 (same as the
// pinned memory benchmark) so the fault-driven migration overhead is large
// enough to actually show up in the timing, instead of being lost in noise.
//
// We time exactly the kernel launch + cudaDeviceSynchronize() region, because
// that is where the GPU faults on pages still resident on the host and pulls
// them over on demand.

#include <cassert>
#include <iostream>

using std::cout;

__global__ void vectorAdd(int *a, int *b, int *c, int N) {
  int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
  if (tid < N) c[tid] = a[tid] + b[tid];
}

int main() {
  const int N = 1 << 26;
  size_t bytes = N * sizeof(int);

  int *a, *b, *c;
  cudaMallocManaged(&a, bytes);
  cudaMallocManaged(&b, bytes);
  cudaMallocManaged(&c, bytes);

  for (int i = 0; i < N; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
  }

  int BLOCK_SIZE = 1 << 10;
  int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, N);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  std::cout << "[um_baseline] kernel+migration time: " << ms << " ms\n";

  for (int i = 0; i < N; i++) assert(c[i] == a[i] + b[i]);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cout << "COMPLETED SUCCESSFULLY!\n";
  return 0;
}
