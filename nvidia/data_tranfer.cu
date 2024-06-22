#include<stdio.h>
#include<assert.h>

inline cudaError_t checkCuda(cudaError_t result) {
# if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
# endif
    return result;
}

void profileCopies(float *h_a, float *h_b, float *d, unsigned int n, const char *desc) {
    printf("\n%s transfers\n", desc);

    unsigned int bytes = n * sizeof(float);

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    printf("Host to Device\n");
    checkCuda(cudaEventRecord(start, 0));
    checkCuda(cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaEventRecord(stop, 0));
    checkCuda(cudaEventSynchronize(stop));
    float time;
    checkCuda(cudaEventElapsedTime(&time, start, stop));
    printf("Time: %f ms\n", time);

    printf("Device to Host\n");
    checkCuda(cudaEventRecord(start, 0));
    checkCuda(cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaEventRecord(stop, 0));
    checkCuda(cudaEventSynchronize(stop));
    checkCuda(cudaEventElapsedTime(&time, start, stop));
    printf("Time: %f ms\n", time);

    for (int i = 0; i < n; i++) {
        if (h_a[i] != h_b[i]) {
            printf("mismatch at %d: %f != %f\n", i, h_a[i], h_b[i]);
            break;
        }
    }

    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
}

int main() {
    // 0 测量传输时间
    // const unsigned int N = 1048576; // 2^20 1mB
    // const unsigned int bytes = N * sizeof(int);

    // int *h_a = (int *)malloc(bytes);
    // memset(h_a, 0, bytes);

    // int *d_a;
    // cudaMalloc(&d_a, bytes);

    // cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    // cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);

    // 1 使用固定内存
    unsigned int nElements = 4 * 1024 * 1024;
    const unsigned int bytes = nElements * sizeof(float);

    // host arrays
    float *h_aPageable, *h_bPageable;   // 普通的内存
    float *h_aPinned, *h_bPinned;       // 固定内存

    // device array
    float *d_a;

    // allocate and initialize
    h_aPageable = (float *)malloc(bytes);
    h_bPageable = (float *)malloc(bytes);
    checkCuda(cudaMallocHost((void **)&h_aPinned, bytes));
    checkCuda(cudaMallocHost((void **)&h_bPinned, bytes));
    checkCuda(cudaMalloc((void **)&d_a, bytes));
    // output device info and transfer size
    cudaDeviceProp prop;
    checkCuda( cudaGetDeviceProperties(&prop, 0) );

    printf("\nDevice: %s\n", prop.name);
    printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));

    // perform copies and report bandwidth
    profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
    profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");

    cudaFree(d_a);
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
    free(h_aPageable);
    free(h_bPageable);
}