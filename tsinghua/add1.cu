#include<math.h>
#include<stdio.h>

const double EPS = 1e-6;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;


void __global__ add(int n, double *x, double *y, double *z, int *min_thread_id) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = x[i] + y[i];
    } else {
        *min_thread_id = min(*min_thread_id, i);
        printf("Error: i = %d\n", i);
    }
}

void check(int n, double *z) {
    bool flag = true;
    for (int i = 0; i < n && flag; i++) {
        if (fabs(z[i] - c) > EPS) {
            flag = false;
            printf("Error: z[%d] = %lf\n", i, z[i]);
            break;
        }
    }
    printf("Check %s!\n", flag ? "pass" : "fail");
}

int main()
{
    int n = 100000;
    double *x, *y, *z;
    int *min_thread_id;
    cudaMallocManaged(&x, n * sizeof(double));
    cudaMallocManaged(&y, n * sizeof(double));
    cudaMallocManaged(&z, n * sizeof(double));
    cudaMallocManaged(&min_thread_id, sizeof(int));
    *min_thread_id = 1e8;
    for (int i = 0; i < n; i++) {
        x[i] = a;
        y[i] = b;
    }
    int blockSize = 256;
    // n=512, blockSize=256, numBlocks=2
    // n=513, blockSize=256, numBlocks=3, 超过513的情况下，会出现多余的线程块
    int numBlocks = (n + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(n, x, y, z, min_thread_id);
    cudaDeviceSynchronize();
    check(n, z);
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    printf("min_thread_id = %d\n", *min_thread_id);
    return 0;
}