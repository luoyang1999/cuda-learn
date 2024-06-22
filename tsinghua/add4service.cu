// 1 有返回值的设备函数
double __device__ add1_service(double a, double b) {
    return a + b;
}

void __global__ add1(const double *x, const double *y, double *z, const int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        z[tid] = add1_service(x[tid], y[tid]);
    }
}

// 2 无返回值的设备函数
void __device__ add2_service(const double a, const double b, double *c) {
    *c = a + b;
}

void __global__ add2(const double *x, const double *y, double *z, const int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        add2_service(x[tid], y[tid], &z[tid]);
    }
}

// 3 用引用的设备函数
void __device__ add3_service(const double a, const double b, double &c) {
    c = a + b;
}

void __global__ add3(const double *x, const double *y, double *z, const int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        add3_service(x[tid], y[tid], z[tid]);
    }
}