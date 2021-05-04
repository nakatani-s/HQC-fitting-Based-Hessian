/*
Error checker for cuda cuBLAS cuSOLVER function
*/
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call,str)                                                         \
{                                                                                    \
    const cudaError_t error = call;                                                  \
    if (error != cudaSuccess)                                                        \
    {                                                                                \
        printf("Cuda Error: %s : %s: %d, ", str, __FILE__, __LINE__);                \
        printf("code:%d, reason: %s\n", error,                                       \
                cudaGetErrorString(error));                                          \
        exit(1);                                                                     \
    }                                                                                \
}                                                                                    \

#define CHECK_CUBLAS(call,str)                                                        \
{                                                                                     \
    if ( call != CUBLAS_STATUS_SUCCESS)                                               \
    {                                                                                 \
        printf("CUBLAS Error: %s : %s %d\n", str, __FILE__, __LINE__);                \
        exit(1);                                                                      \
    }                                                                                 \
}                                                                                     \