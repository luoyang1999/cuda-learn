#include <stdio.h>


static void HandleError(cudaError_t err, const char *file, int line)
{
    if(err != cudaSuccess){
        printf("%s in %s at line%d \n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void conv(float *img, float *kernel, float *result, int width, int height, int kernelSize)
{
    int ti = threadIdx.x;
    int bi = blockIdx.x;
    int id = (bi * blockDim.x + ti);
    if(id >= width * height)
    {
        return;
    }
    int row = id / width;
    int col = id % width;
    for(int i=0;i<kernelSize;i++)
    {
        for(int j=0;j<kernelSize;j++)
        {
            float imgValue = 0;
            int curRow = row - kernelSize / 2 + i;
            int curCol = col - kernelSize / 2 + j;
            if(curRow >= 0 && curRow < height || curCol >= 0 && curCol < width){
                imgValue = img[curRow * width + curCol];
            }
            result[id] += kernel[i * kernelSize + j] * imgValue;
        }
    }
}

# define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

int getThreadNum()
{
    cudaDeviceProp prop;
    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    printf("gpu num=%d\n", count);
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("max thread num=%d\n", prop.maxThreadsPerBlock);
    printf("max grid dimensions: %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    return prop.maxThreadsPerBlock;
}

int main(void)
{
    // init data
    int width = 1920;
    int height = 1080;

    float *img = new float[width * height];
    for(int row=0;row<height;row++)
    {
        for(int col=0;col<width;col++)
        {
            img[col + row*width] = (col + row) % 256;
        }
    }

    int kernelSize = 3;
    float *kernel = new float[kernelSize * kernelSize];
    for(int i=0; i<kernelSize*kernelSize;i++)
    {
        kernel[i] = i % kernelSize - 1;
    }

    float *imgGpu;
    float *kernelGpu;
    float *resultGpu;
    HANDLE_ERROR(cudaMalloc((void **)&imgGpu, width * height * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&kernelGpu, kernelSize * kernelSize * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&resultGpu, width * height * sizeof(int)));

    // copy to gpu
    HANDLE_ERROR(cudaMemcpy(imgGpu, img, width * height * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(kernelGpu, kernel, kernelSize * kernelSize * sizeof(int), cudaMemcpyHostToDevice));

    // get info
    int thread_num = getThreadNum();
    int block_num = (width * height - 0.5) / thread_num + 1;

    // do conv
    conv<<<block_num, thread_num>>>(imgGpu, kernelGpu, resultGpu, width, height, kernelSize);

    // get
    float *img_result = new float[width * height];
    HANDLE_ERROR(cudaMemcpy(img_result, resultGpu, width * height * sizeof(int), cudaMemcpyDeviceToHost));

    // visualizion
    // img
    printf("original image\n");
    for(int row=0;row < 10;row++)
    {
        for(int col=0;col<10;col++)
        {
            printf("%4.1f ", img[col + row*width]);
        }
        printf("\n");
    }

    // kernel
    printf("conv kernel\n");
    for(int row=0;row<kernelSize;row++)
    {
        for(int col=0;col<kernelSize;col++)
        {
            printf("%3.1f ", kernel[col + row*kernelSize]);
        }
        printf("\n");
    }

    // result
    printf("result\n");
    for(int row=0;row < 10;row++)
    {
        for(int col=0;col<10;col++)
        {
            printf("%4.1f ", img_result[col + row*width]);
        }
        printf("\n");
    }

    return 0;
}