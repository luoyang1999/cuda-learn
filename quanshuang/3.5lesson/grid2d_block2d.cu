#include<stdio.h>
#include<stdlib.h>
#include"../tool/common.cuh"

__global__ void add_g2_b2(int *a, int *b, int *c, int length) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int id = y * gridDim.x * blockDim.x + x;

    printf("blockIdx.x: %d, blockIdx.y: %d, blockDim.x: %d, blockDim.y: %d, threadIdx.x: %d, threadIdx.y: %d, id: %d\n", blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y, id);
    if (id < length)
        c[id] = a[id] + b[id];
}

void initArray(int *array, int length) {
    for(int i = 0; i < length; i++) {
        array[i] = rand() & 0xff;
    }
}


int main() {
    // 1 
    setGPU(0);

    // 2 init data
    int elements = 1024;
    int *hostA = (int *)malloc(elements * sizeof(int));
    int *hostB = (int *)malloc(elements * sizeof(int));
    int *hostC = (int *)malloc(elements * sizeof(int));
    if (hostA == NULL || hostB == NULL || hostC == NULL) {
        printf("malloc failed\n");
        exit(1);
    }
    srand(0);
    initArray(hostA, elements);
    initArray(hostB, elements);

    // for(int i = 0; i < 10; i++) {
    //     printf("%d + %d = %d\n", hostA[i], hostB[i], hostA[i] + hostB[i]);
    // }

    // 3 malloc memory on device
    int *deviceA, *deviceB, *deviceC;
    ErrorCheck(cudaMalloc((int **)&deviceA, elements * sizeof(int)), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((int **)&deviceB, elements * sizeof(int)), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((int **)&deviceC, elements * sizeof(int)), __FILE__, __LINE__);
    if (deviceA == NULL || deviceB == NULL || deviceC == NULL) {
        printf("cudaMalloc failed\n");
        exit(1);
    }

    // 4 copy data from host to device
    ErrorCheck(cudaMemcpy(deviceA, hostA, elements * sizeof(int), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(deviceB, hostB, elements * sizeof(int), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    // 5 launch kernel
    // 5.1 g1 b1
    // dim3 block(32);
    // dim3 grid((elements + block.x - 1) / block.x);

    // 5.2 g2 b2
    dim3 block(8, 8);
    dim3 grid(((int)sqrt(elements) + block.x - 1) / block.x, ((int)sqrt(elements) + block.y - 1) / block.y);
    
    add_g2_b2<<<grid, block>>>(deviceA, deviceB, deviceC, elements);

    // 6 copy data from device to host
    ErrorCheck(cudaMemcpy(hostC, deviceC, elements * sizeof(int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    // 7 check result
    for(int i = 0; i < 10; i++) {
        // if (hostC[i] != hostA[i] + hostB[i]) {
        //     printf("check failed\n");
        //     // exit(1);
        // }
        printf("%d + %d = %d\n", hostA[i], hostB[i], hostC[i]);
    }

    // 8 free memory
    free(hostA);
    free(hostB);
    free(hostC);
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;

}