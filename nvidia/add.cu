#include<iostream>
#include<math.h>

__global__ void add(int n, float *x, float *y) 
{
    // int t_id = threadIdx.x;
    // int b_id = blockIdx.x;
    // int id = b_id * blockDim.x + t_id;
    // if (id < n) {
    //     y[id] += x[id];
    // }
    // if (id == 0) {
    //     printf("y[0]=%f", y[0]);
    // }
    // printf("bdx=%d, bid=%d, t_id=%d\n", blockDim.x, b_id, t_id);

    // for(int i=t_id; i < n; i += blockDim.x) {
    //     y[i] += x[i];
    // }

    // for(int i=0; i<n;i++)
    //     y[i] += x[i];

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x; 
    // for (int i = index; i < n; i += stride)
    y[index] = x[index] + y[index];
}

int main() {
    float *x, *y;
    int N  = 1 << 20; // 1M elements
    // 在cpu和gpu统一存储器访问的内存空间， 不用memcpy了
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    int block_size= 256;
    int grid_size = (N + block_size - 1) / block_size;
    // run on cpu
    add<<<grid_size, block_size>>>(N, x, y);
    // 这句话是必须的，cpu和gpu是异步执行，否则导致下面的判断出错
    cudaDeviceSynchronize();

    float maxError = 0.0f;        
    for(int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
        if (maxError > 0.1f) {
            std::cout << i << " " << x[i] << " " << y[i] << std::endl;
            break;
        }
    }
    std::cout << "max error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);

    return 0;
}