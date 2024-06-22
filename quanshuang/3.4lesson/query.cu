#include<stdio.h>
#include"../tool/common.cuh"


int main() {
    int device_id = 0;
    ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);

    cudaDeviceProp prop;
    ErrorCheck(cudaGetDeviceProperties(&prop, device_id), __FILE__, __LINE__);

    printf("Device %d: %s\n", device_id, prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total global memory: %lu\n", prop.totalGlobalMem);
    printf("Shared memory per block: %lu\n", prop.sharedMemPerBlock);
    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Clock rate: %d\n", prop.clockRate);
    printf("Total constant memory: %lu\n", prop.totalConstMem);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
    printf("L2 cache size: %d\n", prop.l2CacheSize);
    printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Compute mode: %d\n", prop.computeMode);
    printf("Concurrent kernels: %d\n", prop.concurrentKernels);
    printf("PCI bus ID: %d\n", prop.pciBusID);
    printf("PCI device ID: %d\n", prop.pciDeviceID);
    printf("Memory clock rate: %d\n", prop.memoryClockRate);
    printf("Memory bus width: %d\n", prop.memoryBusWidth);
    printf("Peak memory bandwidth: %f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    return 0;
}