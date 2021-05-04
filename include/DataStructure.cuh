#include <curand_kernel.h>
#include "params.cuh"
#ifndef DATASTRUCTUE_CUH
#define DATASTRUCTUE_CUH

typedef struct{
    float L;
    float W;
    float WHM;
    float InputSeq[DIM_OF_U][HORIZON];
}MonteCarloMPC;

typedef struct{
    float State[DIM_OF_STATE];
    float Param[NUM_OF_PARAMS];
    float Constraints[NUM_OF_CONSTRAINTS];
    float WeightMatrix[DIM_OF_WEIGHT_MATRIX];
}Controller;

typedef struct{
    float currentU[3][HORIZON];
    float pseudoGradient[HORIZON];
    float cost[HORIZON][3];

    float modified_U[3][HORIZON];
    float delta_u[3][HORIZON];

}SampleBasedHessian;

typedef struct{
    float tensor_vector[SIZE_OF_PARABOLOIDVESTOR];
    float column_vector[SIZE_OF_PARABOLOIDVESTOR];
}HyperParaboloid;

typedef struct{
    float InputSeq[DIM_OF_U];
}InputSequences;
#endif

__global__ void setupMonteCarloMPCVectors(MonteCarloMPC *Out);
