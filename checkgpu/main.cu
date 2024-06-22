#include<stdio.h>

static void HandleError(cudaError_t err, const char *file, int line)
{
    if(err != cudaSuccess){
        printf("%s in %s at line%d \n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

int main(void)
{
    int dev = 0;
    cudaDeviceProp dev_prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&dev_prop, dev));
    printf("device: %s\n", dev_prop.name);
    printf("流处理器SM数量: %d\n", dev_prop.multiProcessorCount);
    printf("每个线程块的共享内存大小:%zu KB\n", dev_prop.sharedMemPerBlock / 1024);
    printf("每个线程块的最大线程数量：%d\n", dev_prop.maxThreadsPerBlock);
    return 0;
    // device: NVIDIA GeForce RTX 3090
    // 流处理器SM数量: 82
    // 每个线程块的共享内存大小:48 KB
    // 每个线程块的最大线程数量：1024
}