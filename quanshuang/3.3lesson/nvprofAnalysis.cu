#include<stdio.h>
#include<stdlib.h>
#include"../tool/common.cuh"

void initData(float *p, int n) {
    for (int i = 0; i < n; i++) {
        p[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

__device__ float add(float a, float b) {
    return a + b;
}
__global__ void matirx1dAdd(float *A, float *B, float *C, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        C[id] = add(A[id], B[id]);
    }
}

int main() {
    // 1 setgpu
    const int dev = 0;
    setGPU(dev);

    // 2 分配主机内存 设备内存
    int iElements = 4096;    // 元素数量
    size_t nBytes = iElements * sizeof(float);

    // 2.1 分配主机内存并初始化
    float *fpHost_A, *fpHost_B, *fpHost_C;
    fpHost_A = (float *)malloc(nBytes);
    fpHost_B = (float *)malloc(nBytes);
    fpHost_C = (float *)malloc(nBytes);

    if (fpHost_A == NULL || fpHost_B == NULL || fpHost_C == NULL) {
        printf("Host malloc failed\n");
        free(fpHost_A);
        free(fpHost_B);
        free(fpHost_C);
        return 1;
    }
    memset(fpHost_A, 0, nBytes);
    memset(fpHost_B, 0, nBytes);
    memset(fpHost_C, 0, nBytes);

    // 2.2 分配设备内存
    float *fpDevice_A, *fpDevice_B, *fpDevice_C;

    cudaMalloc((float**)&fpDevice_A, nBytes);
    cudaMalloc((float**)&fpDevice_B, nBytes);
    cudaMalloc((float**)&fpDevice_C, nBytes);

    if (fpDevice_A == NULL || fpDevice_B == NULL || fpDevice_C == NULL) {
        printf("Device malloc failed\n");
        cudaFree(fpDevice_A);
        cudaFree(fpDevice_B);
        cudaFree(fpDevice_C);
        return 1;
    }

    // 3 初始化主机数据
    srand(666);
    initData(fpHost_A, iElements);
    initData(fpHost_B, iElements);
    
    // 4 将主机数据拷贝到设备
    cudaMemcpy(fpDevice_A, fpHost_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_B, fpHost_B, nBytes, cudaMemcpyHostToDevice);

    // 5 执行核函数
    dim3 grid(32); // 32个网格
    dim3 block(iElements / 32); // 512 / 32 = 16个线程块
    
    matirx1dAdd<<<grid, block>>> (fpDevice_A, fpDevice_B, fpDevice_C, iElements);
    // cudaDeviceSynchronize();

    // 6 将设备数据拷贝到主机
    cudaMemcpy(fpHost_C, fpDevice_C, nBytes, cudaMemcpyDeviceToHost);   // 拥有同步功能
    for(int i = 0; i < 10; i ++) {
        printf("%f + %f = %f\n", fpHost_A[i], fpHost_B[i], fpHost_C[i]);
    }

    // 7 释放内存
    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);
    cudaFree(fpDevice_A);
    cudaFree(fpDevice_B);
    cudaFree(fpDevice_C);

    return 0;

}