#include<stdio.h>

// SAXPY stands for “Single-precision A*X Plus Y”
__global__ void saxpy(int n, float a, float *x, float *y) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n)
        y[id] = a * x[id] + y[id];
}


int main() {
    int N = 1 << 20;
    float *x, *y, *d_x, *d_y;
    x = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));

    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));
    
    for(int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block_size = 256;
    dim3 grid_size = (N + 255) / 256;
    
    cudaEventRecord(start);
    saxpy<<<grid_size, block_size>>>(N, 2.0f, d_x, d_y);
    cudaEventRecord(stop);

    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Ms: %f\n", milliseconds);
    printf("effective Bandwidth(gb/s): %f\n", N*4*3/milliseconds/1e6);

    float maxError = 0.0f;
    for (int i=0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 4.0f));
        if (maxError > 0.1f) {
            printf("%f %f\n", x[i], y[i]);
            break;
        }
    }

    printf("max error: %.2f\n", maxError);

    free(x);
    free(y);
    cudaFree(x);
    cudaFree(y);

}