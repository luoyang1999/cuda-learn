#include<stdio.h>

__global__ void hello_from_gpu() {


    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;

    const int id = block_id * blockDim.x + thread_id;
    printf("hello world form block %d, thread %d, id %d\n", block_id, thread_id, id);
}

int main() {
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}