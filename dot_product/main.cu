#include<stdio.h>
#define LENGTH 16
#define THREADNUM 4
#define BLOCKNUM 2

__global__ void dot_product(float *a_gpu, float *b_gpu, float *r_gpu)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int global_id = tid + bid * blockDim.x;
    int total_thread_num = BLOCKNUM * THREADNUM;

    __shared__ float sData[THREADNUM];
    sData[tid] = a_gpu[tid] * b_gpu[tid];
    __syncthreads();

    while(global_id < LENGTH){
        sData[tid] += a_gpu[global_id] * b_gpu[global_id];
        global_id += total_thread_num;
    }
    __syncthreads();

    for(int i=THREADNUM/2; i>0; i=i/2){
        if(tid<i){
            sData[tid] = sData[tid] + sData[tid + i];
        }
        __syncthreads();
    }
    if(tid == 0){
        // r_gpu[bid] = sData[0];
        atomicAdd(r_gpu, sData[0]);
    }
}

int main()
{
    float *a = new float[LENGTH];
    float *b = new float[LENGTH];
    for(int i=0;i<LENGTH;i++){
        a[i] = i * (i + 1);
        b[i] = i * (i - 2);
    }

    float *a_gpu;
    cudaMalloc((void**)&a_gpu, LENGTH * sizeof(int));
    cudaMemcpy(a_gpu, a, LENGTH * sizeof(int), cudaMemcpyHostToDevice);
    float *b_gpu;
    cudaMalloc((void**)&b_gpu, LENGTH * sizeof(int));
    cudaMemcpy(b_gpu, b, LENGTH * sizeof(int), cudaMemcpyHostToDevice);

    float *r_gpu;
    cudaMalloc((void**)&r_gpu, LENGTH * sizeof(int));
    dot_product<<<BLOCKNUM, THREADNUM>>>(a_gpu, b_gpu, r_gpu);

    float r[1];
    cudaMemcpy(r, r_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    // float result = 0;
    // for(int i=0;i<BLOCKNUM;i++){
    //     result += r[i];
    // }
    // printf("b: %f\n", result);
    printf("b: %f\n", r[0]);
    return 0;
}