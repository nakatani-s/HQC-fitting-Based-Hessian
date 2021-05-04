/*
    DataStructure.cu
    *構造体の初期化
*/
#include "../include/DataStructure.cuh"

__global__ void setupMonteCarloMPCVectors(MonteCarloMPC *Out)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    Out[id].L = 0.0f;
    Out[id].W = 0.0f;
    for(int i = 0; i < HORIZON; i++){
        for(int k = 0; k < DIM_OF_U; k++){
            Out[id].InputSeq[k][i] = 0.0f;
        }
    }
    __syncthreads();
}