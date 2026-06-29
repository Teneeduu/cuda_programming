// Timed version of unified memory + prefetch. Two numbers are printed:
//
// 1. [um_prefetch] kernel+migration time — same narrow window as the
//    um_baseline file (kernel launch + cudaDeviceSynchronize only). This is
//    fast, but NOT a fair total-cost number: the real data movement
//    (cudaMemPrefetchAsync) happens *before* this timer starts, so its cost
//    is invisible here.
//
// 2. [um_prefetch] full pipeline time — wraps prefetch-to-GPU + kernel +
//    prefetch-back-to-host in one timer. This is the apples-to-apples number
//    to compare against pinned memory's HtoD+DtoH total, since it includes
//    all the real transfer work, not just the post-migration kernel run.
//    (No extra cudaDeviceSynchronize needed before recording full_stop: the
//    prefetch-back calls and the event are all enqueued on the same default
//    stream, and a stream guarantees in-order execution, so full_stop only
//    fires after they finish.)

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

  int id = cudaGetDevice(&id);

  cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  cudaMemPrefetchAsync(c, bytes, id);

  for (int i = 0; i < N; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
  }

  int BLOCK_SIZE = 1 << 10;
  int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  cudaEvent_t start, stop, full_start, full_stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&full_start);
  cudaEventCreate(&full_stop);

  // ---- full pipeline timing starts: prefetch to GPU + kernel + prefetch back ----
  cudaEventRecord(full_start);

  cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, id);
  cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, id);
  cudaMemPrefetchAsync(a, bytes, id);
  cudaMemPrefetchAsync(b, bytes, id);

  cudaEventRecord(start);
  vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, N);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);
  cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);
  cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

  cudaEventRecord(full_stop);
  cudaEventSynchronize(full_stop);
  // ---- full pipeline timing ends ----

  float ms = 0, full_ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventElapsedTime(&full_ms, full_start, full_stop);
  std::cout << "[um_prefetch] kernel+migration time: " << ms << " ms\n";
  std::cout << "[um_prefetch] full pipeline time (prefetch+kernel+prefetch-back): "
            << full_ms << " ms\n";

  for (int i = 0; i < N; i++) assert(c[i] == a[i] + b[i]);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaEventDestroy(full_start);
  cudaEventDestroy(full_stop);

  cout << "COMPLETED SUCCESSFULLY!\n";
  return 0;
}
