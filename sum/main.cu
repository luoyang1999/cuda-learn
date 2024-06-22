#include<stdio.h>
#include<time.h>
#include<sys/time.h>


__global__ void sum(float *a, float *b)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int sumNum = blockDim.x;

    __shared__ float sData[1024];
    sData[tid] = a[bid * sumNum + tid];
    __syncthreads();
    for(int i=8;i>=1;i=i/2){
        if(tid < i){
            sData[tid] = sData[tid] + sData[tid + i];
        }
        __syncthreads();
    }
    if(tid == 0)
        b[bid] = sData[0];
}

void sumNum(float *a, float *b, int len)
{
    b[0] = 0;
    for(int i=0;i<len;i++){
        b[0] += a[i];
    }
}

int main(void)
{
    int len = 1024;
    float a[len];
    for(int i=0;i<len;i++){
        a[i] = i * (i+ 1);
    }
    float *a_gpu;
    cudaMalloc((void **)&a_gpu, sizeof(float) * len);
    cudaMemcpy(a_gpu, a, sizeof(float) * len, cudaMemcpyHostToDevice);
    int block_num = 1;
    int thread_num = len;

    float *b_gpu;
    cudaMalloc((void **)&b_gpu, sizeof(float) * 1);

    float b[1];

    // measurement cuda time
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    for(int i=0;i<100000;i++)
        sum<<<thread_num, 1>>>(a_gpu, b_gpu);
    gettimeofday(&end_time, NULL);
    printf("cuda use time: %ld\n", (end_time.tv_sec-start_time.tv_sec)*1000000 + (end_time.tv_usec - start_time.tv_usec));

    // measurement cpu time
    gettimeofday(&start_time, NULL);
    for(int i=0;i<100000;i++)
        sumNum(a, b, len);
    gettimeofday(&end_time, NULL);
    printf("cpu use time: %ld\n", (end_time.tv_sec-start_time.tv_sec)*1000000 + (end_time.tv_usec - start_time.tv_usec));

    cudaMemcpy(b, b_gpu, sizeof(float) * 1, cudaMemcpyDeviceToHost);
    printf("sum=%2.1f\n", b[0]);
    return 0;
}