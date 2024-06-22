#include<stdio.h>
#include"../tool/common.cuh"

int main() {
    int *ipHost_A;
    ipHost_A = (int*)malloc(sizeof(int));

    memset(ipHost_A, 0, sizeof(int));

    int *ipDev_A;
    ErrorCheck(cudaMalloc((int**)&ipDev_A, sizeof(int)), __FILE__, __LINE__);

    ErrorCheck(cudaMemcpy(ipHost_A, ipDev_A, sizeof(int), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    free(ipHost_A);
    ErrorCheck(cudaFree(ipDev_A), __FILE__, __LINE__);
}