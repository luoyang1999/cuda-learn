#include<stdio.h>

__global__ void hello_from_gpu(void)
{
    const int grid_size = gridDim.x;    // 线程块的数量
    const int block_size = blockDim.x;  // 线程块中线程的数量

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    printf("Hello World from block %d thread %d \n", bid, tid);
}

int main(void)
{
    // <<<网格数, 线程数>>> means 1 block and 1 thread
    hello_from_gpu<<<2, 4>>>();
    // cudaDeviceSynchronize() is used to synchronize the host and the device
    // 可以促进缓冲区刷新，确保所有的输出都被打印出来
    cudaDeviceSynchronize();
    return 0;
}