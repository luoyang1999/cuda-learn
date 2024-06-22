#include<stdio.h>

int main() {
    int iDeviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&iDeviceCount);

    if (error != cudaSuccess || iDeviceCount == 0) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        return 1;
    }
    printf("Device count: %d\n", iDeviceCount);

    // 设置执行
    int iDev = 0;
    error = cudaSetDevice(iDev);
    if (error != cudaSuccess) {
        printf("cudaSetDevice returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        return 1;
    }
    printf("Set device: %d\n", iDev);
}