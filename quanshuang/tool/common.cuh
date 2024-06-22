#include<stdio.h>

void setGPU(int index) {
    int iDeviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&iDeviceCount);

    if (error != cudaSuccess || iDeviceCount == 0) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        exit(-1);
    }
    printf("Device count: %d\n", iDeviceCount);

    if (index < 0 || index >= iDeviceCount) {
        printf("Invalid device index\n");
        exit(-1);
    }
    // 设置执行
    int iDev = index;
    error = cudaSetDevice(iDev);
    if (error != cudaSuccess) {
        printf("cudaSetDevice returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        exit(-1);
    }
}

/*
filename: __FILE__
lineNumber: __LINE__
*/
cudaError_t ErrorCheck(cudaError_t code, const char *filename, int lineNumber) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s line:%d\n", cudaGetErrorString(code), filename, lineNumber);
    }
    return code;
}