/* 
    MCMPC.cuh
*/

#include<cuda.h>
#include<curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include "params.cuh"
#include "DataStructure.cuh"
#include "dynamics.cuh"

void MCMPC_by_weighted_mean( InputSequences *Output, MonteCarloMPC *PrCtr, int uIndex);

void StateUpdate( Controller *CtrPrm, float *hSt);

__global__ void setup_kernel(curandState *state,int seed);

__global__ void MCMPC_callback_elite_sample(MonteCarloMPC *OutPtElt, MonteCarloMPC *AllSmplDt, int *indices);

__global__ void MCMPC_callback_elite_sample_by_IT(MonteCarloMPC *OutPtElt, MonteCarloMPC *AllSmplDt, int *indices);

__global__ void MCMPC_Cart_and_Single_Pole(MonteCarloMPC *PrCtr, curandState *randomSeed, Controller *Ctr,InputSequences *mean, float var, float *cost);