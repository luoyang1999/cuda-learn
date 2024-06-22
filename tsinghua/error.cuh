#pragma once    // 保证头文件只被编译一次
#include<stdio.h>

#define CHECK(call)                          \
do                                           \
{                                            \
    const cudaError_t error = call;          \
    if (error != cudaSuccess)                \
    {                                        \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                             \
    }                                        \
} while(0)

