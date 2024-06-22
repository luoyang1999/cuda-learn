#include<stdio.h>

#define N 2 << 10

__global__ void add(int *a, int*b, int *res)
{
    int bi = blockIdx.x;
    int ti = threadIdx.x;
    int idx = bi * blockDim.x + ti;
    res[idx] = a[idx] + b[idx];
    // printf("res[%d]=%d", idx, res[idx]);
}

int main(void)
{
    // 进行2个超长数组的加法，分到不同的线程块执行
    int *a = new int[N];
    int *b = new int[N];
    int *res = new int[N];

    for(int i=0;i<N;i++){
        a[i] = i;
        b[i] = 2 * i;
    }

    int *a_gpu, *b_gpu, *res_gpu;
    cudaMalloc((void **)&a_gpu, sizeof(int) * N);
    cudaMalloc((void **)&b_gpu, sizeof(int) * N);
    cudaMalloc((void **)&res_gpu, sizeof(int) * N);
    
    // copy to GPU
    cudaMemcpy(a_gpu, a, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, sizeof(int) * N, cudaMemcpyHostToDevice);

    // todo on gpu
    // dim3 block_size(256);
    // dim3 grid_size((N + block_size.x -2) / block_size.x);
    int block_num = 256;
    int thread_num = (N + block_num - 1) / block_num;

    add<<<block_num, thread_num>>>(a_gpu, b_gpu, res_gpu);

    // copy tocpu
    cudaMemcpy(res, res_gpu, sizeof(int) * N, cudaMemcpyDeviceToHost);

    //visiual
    for(int i=0;i<10;i++){
        printf("res[%d]=%d\n", i, res[i]);
    }

    return 0;
}