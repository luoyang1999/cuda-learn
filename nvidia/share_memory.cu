#include<stdio.h>

__global__ void staticReverse(int *d, int n) {
    __shared__ int s[64];
    int t = threadIdx.x;
    s[t] = d[t];
    __syncthreads();
    d[t] = s[n-t-1];
}

__global__ void dynamicReverse(int *d, int n) {

    extern __shared__ int s[];
    int t = threadIdx.x;
    s[t] = d[t];
    __syncthreads();
    d[t] = s[n-t-1];
}

int main() {
    const int n = 64;
    int a[n], d[n];
    for(int i=0; i<n; i++) {
        a[i] = i;   // 正数组
        d[i] = 0;
    }

    int *d_d;
    cudaMalloc(&d_d, n*sizeof(int));

    // 静态共享内存
    cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
    staticReverse<<<1, n>>>(d_d, n);
    cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0; i<n; i++) {
        printf("%d ", d[i]);
    }

    // 动态共享内存
    cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
    // 传递共享内存大小 单位是字节
    dynamicReverse<<<1, n, n*sizeof(int)>>>(d_d, n);
    cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0; i<n; i++) {
        printf("%d ", d[i]);
    }

}