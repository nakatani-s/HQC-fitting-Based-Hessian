/*
    Functions for making samplebased hessian
*/ 

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "cublas_v2.h"

#include "params.cuh"
#include "DataStructure.cuh"
#include "Matrix.cuh"
#include "dynamics.cuh"


void getInvHessian( float *Hess, SampleBasedHessian *hostHess);

void GetInputByLSMfittingMethod(float *OutPut, HyperParaboloid *hostHP, int sz_QC, int sz_HESS, int sz_LSM);
void GetInputByLSMfittingMethodFromcuBLAS(float *OutPut, HyperParaboloid *hostHP, int sz_QC, int sz_HESS, int sz_LSM);

__global__ void getPseduoGradient(SampleBasedHessian  *Hess, float epsilon);
__global__ void ParallelSimForPseudoGrad(SampleBasedHessian *Hess, MonteCarloMPC *sample, InputSequences *MCresult, Controller *CtrPrm, float delta, int *indices);
__global__ void getCurrentUpdateResult( SampleBasedHessian *HessInfo, float *invHess );
__global__ void copyInpSeqFromSBH( InputSequences *Output, SampleBasedHessian *HessInfo);
__global__ void copySingleDateToInputSequences( InputSequences *Output, float *Sequences);
