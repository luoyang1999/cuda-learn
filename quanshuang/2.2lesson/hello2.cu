#include<stdio.h>


__global__ void hello_from_gpu() {
    printf("Hello from GPU\n");
}

int main() {
    hello_from_gpu<<<2,5>>>(); // <<<1,1>>> is the syntax for launching a kernel on the GPU
    cudaDeviceSynchronize(); // This is needed to make sure that the kernel has finished execution before the program exits
    return 0;
}