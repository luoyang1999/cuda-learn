#include<stdio.h>

__global__ void hello_from_gpu() {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    const int id = bid * blockDim.x + tid;
    printf("Hello from GPU! My block id is %d and thread id is %d and id is %d\n", bid, tid, id);
}

int main() {
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}