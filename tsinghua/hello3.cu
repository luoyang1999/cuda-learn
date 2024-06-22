#include<stdio.h>

__global__ void hello_from_gpu(void)
{
    printf("Hello World from GPU! \n");
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