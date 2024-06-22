#include<stdio.h>

inline cudaError_t checkCuda(cudaError_t result) {
    if(result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    }
    return result;
}

// 访问偏移量s的kernel
template<typename T>
__global__ void offset(T *a, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] = a[i] + s;
}

// 访问偏移量s的kernel
template<typename T>
__global__ void stride(T *a, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i*s] = a[i*s] + 1;
}

template <typename T>
void runTest(int deviceId, int nMB)
{
  int blockSize = 256;
  float ms;

  T *d_a;
  cudaEvent_t startEvent, stopEvent;
    
  int n = nMB*1024*1024/sizeof(T);

  // NB:  d_a(33*nMB) for stride case
  checkCuda( cudaMalloc(&d_a, n * 33 * sizeof(T)) );

  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );

  printf("Offset, Bandwidth (GB/s):\n");
  
  offset<<<n/blockSize, blockSize>>>(d_a, 0); // warm up

    // 访问偏移0-32的，这个速度很快，8bit 16bit对其与否影响不大
  for (int i = 0; i <= 32; i++) {
    checkCuda( cudaMemset(d_a, 0, n * sizeof(T)) );

    checkCuda( cudaEventRecord(startEvent,0) );
    offset<<<n/blockSize, blockSize>>>(d_a, i);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );

    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    printf("%d, %f\n", i, 2*nMB/ms);
  }

  printf("\n");
  printf("Stride, Bandwidth (GB/s):\n");

  // 一个线程束，大步浮前进，以1-32为步长，这个越长速度越慢，很难有缓存
  stride<<<n/blockSize, blockSize>>>(d_a, 1); // warm up
  for (int i = 1; i <= 32; i++) {
    checkCuda( cudaMemset(d_a, 0, n * sizeof(T)) );

    checkCuda( cudaEventRecord(startEvent,0) );
    stride<<<n/blockSize, blockSize>>>(d_a, i);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );

    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    printf("%d, %f\n", i, 2*nMB/ms);
  }

  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  cudaFree(d_a);
}


int main() {
    runTest<int>(0, 16);
}