#include<stdio.h>
#include<math.h>

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        exit(1);
    }
    return result;
}

// 进行一个计算，将数组+1
__global__ void kernel(float *a, int offset) {
    int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
    float x = (float)i;
    float s = sinf(x); 
    float c = cosf(x);
    a[i] = a[i] + sqrtf(s*s+c*c);
}

float maxError(float *a, int n) {
    float maxE = 0;
    for (int i = 0; i < n; i++) {
        float error = fabs(a[i] - 1.0f);
        if (error > maxE) {
            maxE = error;
        }
    }

    return maxE;
}


int main() {
    const int blockSize = 256, nStreams = 4;
    // n数组长度
    const int n = 4 * 1024 * blockSize * nStreams;
    const int streamSize = n / nStreams;
    const int streamBytes = streamSize * sizeof(float);
    const int bytes = n * sizeof(float);

    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, 0));
    if (!prop.deviceOverlap) {
        printf("Device will not handle overlaps, so no speed up from streams\n");
        return 0;
    }

    float *a, *d_a;
    checkCuda(cudaMallocHost(&a, bytes));
    checkCuda(cudaMalloc(&d_a, bytes));
 
    float ms;

    cudaEvent_t startEvent, stopEvent, dummyEvent;
    cudaStream_t stream[nStreams];

    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));
    checkCuda(cudaEventCreate(&dummyEvent));

    for (int i = 0; i < nStreams; i++) {
        checkCuda(cudaStreamCreate(&stream[i]));
    }

    // baseline case - sequential transfer and execute
    memset(a, 0, bytes);
    checkCuda(cudaEventRecord(startEvent, 0));
    checkCuda(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    kernel<<<n/blockSize, blockSize>>>(d_a, 0);
    checkCuda(cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Time for sequential transfer and execute (ms): %f\n", ms);
    printf("Max error: %f\n", maxError(a, n));

    // asynchronous version 1: loop over {copy, kernel, copy}
    memset(a, 0, bytes);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < nStreams; i++) {
        int offset = i * streamSize;
        // 分到不同的流里面执行
        checkCuda(cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]));
        kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
        checkCuda(cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]));
    }

    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);
    printf("  max error: %e\n", maxError(a, n));

    memset(a, 0, bytes);
    checkCuda(cudaEventRecord(startEvent, 0));
    // 通过strema进行异步操作
    // step1: loop over copy, kernel, copy
    for (int i = 0; i < nStreams; i++) {
        int offset = i * streamSize;
        checkCuda(cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]));
    }
    // step2: loop over kernel
    for (int i = 0; i < nStreams; i++) {
        int offset = i * streamSize;
        kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
    }
    // step3: loop over copy
    for(int i = 0; i < nStreams; i++) {
        int offset = i * streamSize;
        checkCuda(cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]));
    }
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Time for asynchronous V2 transfer and execute (ms): %f\n", ms);
    printf("  max error: %e\n", maxError(a, n));

    // cleanup
    checkCuda(cudaEventDestroy(startEvent));
    checkCuda(cudaEventDestroy(stopEvent));
    checkCuda(cudaEventDestroy(dummyEvent));
    for(int i=0; i < nStreams; i++) {
        checkCuda(cudaStreamDestroy(stream[i]));
    }
    checkCuda(cudaFree(d_a));
    checkCuda(cudaFreeHost(a));

}