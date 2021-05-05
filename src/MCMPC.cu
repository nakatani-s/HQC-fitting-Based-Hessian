/* 
MCMPC.cu 
*/
#include<stdio.h>
#include "../include/MCMPC.cuh"

void MCMPC_by_weighted_mean( InputSequences *Output, MonteCarloMPC *PrCtr, int uIndex)
{
    float totalWeight = 0.0f;
    float temp[HORIZON] = { };
    for(int i = 0; i < NUM_OF_ELITESAMPLE; i++){
        if(isnan(PrCtr[i].W))
        {
            totalWeight += 0.0f;
        }else{
            totalWeight += PrCtr[i].W;
        }
    }
    for(int i = 0; i < HORIZON; i++){
        for(int k = 0; k < NUM_OF_ELITESAMPLE; k++){
            if(isnan(PrCtr[k].W))
            {
                temp[i] += 0.0f;
            }else{
                temp[i] += (PrCtr[k].W * PrCtr[k].InputSeq[uIndex][i]) / totalWeight;
            }
        }
        if(isnan(temp[i]))
        {
            Output[i].InputSeq[uIndex] = 0.0f;
        }else{
            Output[i].InputSeq[uIndex] = temp[i];
        }
    }
}

void StateUpdate( Controller *CtrPrm, float *hSt)
{
    for(int i = 0; i < DIM_OF_STATE; i++)
    {
        CtrPrm->State[i] = hSt[i];
    }
}

__device__ void MemCpyInThread(float *prm, float *cnstrnt, float *mtrx, Controller *Ctr)
{
    for(int i = 0; i < NUM_OF_PARAMS; i++){
        prm[i] = Ctr->Param[i];
    }
    for(int i = 0; i < NUM_OF_CONSTRAINTS; i++){
        cnstrnt[i] = Ctr->Constraints[i];
    }
    for(int i = 0; i < DIM_OF_WEIGHT_MATRIX; i++){
        mtrx[i] = Ctr->WeightMatrix[i];
    }
}

__global__ void setup_kernel(curandState *state,int seed)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number,
     no offset */
    curand_init(seed, id, 0, &state[id]);
}

__device__ float gen_u(unsigned int id, curandState *state, float ave, float vr)
{
    float u;
    curandState localState = state[id];
    u = curand_normal(&localState) * vr + ave;
    return u;
}

__global__ void MCMPC_callback_elite_sample(MonteCarloMPC *OutPtElt, MonteCarloMPC *AllSmplDt, int *indices)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    OutPtElt[id] = AllSmplDt[indices[id]];
    __syncthreads();
    /*OutPtElt[id].W = AllSmplDt[indices[id]].W;
    OutPtElt[id].L = AllSmplDt[indices[id]].L;
    for(int t = 0; t < HORIZON; t++){
        for(intk = 0; k < DIM_OF_U; k++){
            OutPtElt[id].InputSeq[k][t] = AllSmplDt[indices[id]].L
        }
    }*/
}

__global__ void MCMPC_callback_elite_sample_by_IT(MonteCarloMPC *OutPtElt, MonteCarloMPC *AllSmplDt, int *indices)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    // OutPtElt[id] = AllSmplDt[indices[id]];
    // __syncthreads();
    float mdCost = 0.0f;
    float lambda = 1.0f;
    OutPtElt[id].L = AllSmplDt[indices[id]].L;
    mdCost = AllSmplDt[indices[id]].L - AllSmplDt[indices[0]].L;
    OutPtElt[id].W = exp( -mdCost / lambda );
    for(int t = 0; t < HORIZON; t++){
        for(int k = 0; k < DIM_OF_U; k++){
            OutPtElt[id].InputSeq[k][t] = AllSmplDt[indices[id]].L;
        }
    }
}

__global__ void MCMPC_Cart_and_Single_Pole(MonteCarloMPC *PrCtr, curandState *randomSeed, Controller *Ctr, InputSequences *mean, float var, float *cost)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int seq;
    seq = id;

    float stageCost = 0.0f;
    float totalCost = 0.0f;

    // float u[HORIZON] = { };
    InputSequences *InputSeqInThread;
    InputSeqInThread = (InputSequences *)malloc(sizeof(InputSeqInThread) * HORIZON);
    float stateInThisThreads[DIM_OF_STATE] = { };
    float dstateInThisThreads[DIM_OF_STATE] = { };
    float d_param[NUM_OF_PARAMS], d_constraints[NUM_OF_CONSTRAINTS], d_matrix[DIM_OF_WEIGHT_MATRIX];
    MemCpyInThread(d_param, d_constraints, d_matrix, Ctr);

    for(int i = 0; i < DIM_OF_STATE; i++){
        stateInThisThreads[i] = Ctr->State[i];
    }

    for(int t = 0; t < HORIZON; t++)
    {
        for(int uIndex = 0; uIndex < DIM_OF_U; uIndex++ ){
            if(isnan(mean[t].InputSeq[uIndex])){
                //u[t] = d_data[0].Input[t];
                if(t < HORIZON -1){
                    // u[t] = gen_u(seq, randomSeed, PrCtr[0].InputSeq[uIndex][t+1], var);
                    InputSeqInThread[t].InputSeq[uIndex] = gen_u(seq, randomSeed, PrCtr[0].InputSeq[uIndex][t+1], var);
                seq += NUM_OF_SAMPLES;
                }else{
                    // u[t] = gen_u(seq, randomSeed, PrCtr[0].InputSeq[uIndex][HORIZON - 1], var);
                    InputSeqInThread[t].InputSeq[uIndex] = gen_u(seq, randomSeed, PrCtr[0].InputSeq[uIndex][HORIZON-1], var);
                    seq += NUM_OF_SAMPLES;
                }
            }else{
                // u[t] = gen_u(seq, randomSeed, mean[t].InputSeq[uIndex], var);
                InputSeqInThread[t].InputSeq[uIndex] = gen_u(seq, randomSeed, mean[t].InputSeq[uIndex], var);
                seq += NUM_OF_SAMPLES;
            }
            if(InputSeqInThread[t].InputSeq[uIndex] < Ctr->Constraints[0]){
                InputSeqInThread[t].InputSeq[uIndex] = Ctr->Constraints[0];
            }
            if(InputSeqInThread[t].InputSeq[uIndex] > Ctr->Constraints[1]){
                InputSeqInThread[t].InputSeq[uIndex] = Ctr->Constraints[1];
            }
        }

        for(int sec = 0; sec < 2; sec++){
            dstateInThisThreads[0] = stateInThisThreads[2];
            dstateInThisThreads[1] = stateInThisThreads[3];
            /*
            dstateInThisThreads[2] = Cart_type_Pendulum_ddx(u[t], stateInThisThreads[0], stateInThisThreads[1], stateInThisThreads[2], stateInThisThreads[3], d_param); //ddx
            dstateInThisThreads[3] = Cart_type_Pendulum_ddtheta(u[t], stateInThisThreads[0], stateInThisThreads[1], stateInThisThreads[2], stateInThisThreads[3], d_param);
            */
            dstateInThisThreads[2] = Cart_type_Pendulum_ddx(InputSeqInThread[t].InputSeq[0], stateInThisThreads[0], stateInThisThreads[1], stateInThisThreads[2], stateInThisThreads[3], d_param); //ddx
            dstateInThisThreads[3] = Cart_type_Pendulum_ddtheta(InputSeqInThread[t].InputSeq[0], stateInThisThreads[0], stateInThisThreads[1], stateInThisThreads[2], stateInThisThreads[3], d_param);
            stateInThisThreads[2] = stateInThisThreads[2] + (interval * dstateInThisThreads[2]);
            stateInThisThreads[3] = stateInThisThreads[3] + (interval * dstateInThisThreads[3]);
            stateInThisThreads[0] = stateInThisThreads[0] + (interval * dstateInThisThreads[0]);
            stateInThisThreads[1] = stateInThisThreads[1] + (interval * dstateInThisThreads[1]);
#ifdef COLLISION
            if(stateInThisThreads[0] <= d_constraints[2]){
                float collide[3] = { };
                collide[0] = d_param[0] * d_param[1] * cosf(stateInThisThreads[1]);
                collide[1] = d_param[2] + d_param[0] * powf(d_param[1],2);
                collide[2] = collide[0] / collide[1];
                stateInThisThreads[3] = stateInThisThreads[3] + (1 + d_param[7]) * collide[2] * stateInThisThreads[2]; //dtheta = dtheta + (1+e) * F * dx
                stateInThisThreads[2] = -d_param[7] * stateInThisThreads[2]; // dx = -e * dx
                stateInThisThreads[0] = d_constraints[2];
            }
            if(d_constraints[3] <= stateInThisThreads[0]){
                float collide[3] = { };
                collide[0] = d_param[0] * d_param[1] * cosf(stateInThisThreads[1]);
                collide[1] = d_param[2] + d_param[0] * powf(d_param[1],2);
                collide[2] = collide[0] / collide[1];
                stateInThisThreads[3] = stateInThisThreads[3] + (1 + d_param[7]) * collide[2] * stateInThisThreads[2]; //dtheta = dtheta + (1+e) * F * dx
                stateInThisThreads[2] = -d_param[7] * stateInThisThreads[2]; // dx = -e * dx
                stateInThisThreads[0] = d_constraints[3];
            }
#endif
        }
        while(stateInThisThreads[1] > M_PI)
            stateInThisThreads[1] -= (2 * M_PI);
        while(stateInThisThreads[1] < -M_PI)
            stateInThisThreads[1] += (2 * M_PI);
        /*
        stageCost = stateInThisThreads[0] * stateInThisThreads[0] * d_matrix[0] + stateInThisThreads[1] * stateInThisThreads[1] * d_matrix[1]
            + stateInThisThreads[2] * stateInThisThreads[2] * d_matrix[2] + stateInThisThreads[3] * stateInThisThreads[3] * d_matrix[3]
            + u[t] * u[t] * d_matrix[4];
        */
        stageCost = stateInThisThreads[0] * stateInThisThreads[0] * d_matrix[0] + stateInThisThreads[1] * stateInThisThreads[1] * d_matrix[1]
            + stateInThisThreads[2] * stateInThisThreads[2] * d_matrix[2] + stateInThisThreads[3] * stateInThisThreads[3] * d_matrix[3]
            + InputSeqInThread[t].InputSeq[0] * InputSeqInThread[t].InputSeq[0] * d_matrix[4];

#ifndef COLLISION
            if(stateInThisThreads[0] <= 0){
                stageCost += 1 / (powf(stateInThisThreads[0] - d_constraints[2],2) * invBarrier);
                if(stateInThisThreads[0] < d_constraints[2]){
                    stageCost += 1000000;
                }
            }else{
                stageCost += 1 / (powf(d_constraints[3] - stateInThisThreads[0],2) * invBarrier);
                if(stateInThisThreads[0] > d_constraints[3]){
                    stageCost += 1000000;
                }
            }
#endif
            totalCost += stageCost;
            stageCost = 0.0f;
    }

    if(isnan(totalCost))
    {
        totalCost = 1000000 * (DIM_OF_STATE + DIM_OF_U);
    }

    float KL_COST, S, lambda;
    float HM_COST, HM;
    lambda = DIM_OF_STATE * HORIZON; // Using Constant Lambda
    S = totalCost / lambda;
    HM = totalCost / (0.75 * HORIZON);
    HM_COST = exp(-HM);
    if(isnan(HM_COST)){
      HM_COST = 0.0f;
    }else{
      HM_COST = HM_COST;
    }
    KL_COST = exp(-S);
    __syncthreads();
    PrCtr[id].WHM = HM_COST;
    PrCtr[id].L = totalCost;
    PrCtr[id].W = KL_COST;
    cost[id] = totalCost;
    for(int i = 0; i < HORIZON; i++){
        for(int k = 0; k < DIM_OF_U; k++)
        {
            PrCtr[id].InputSeq[k][i] = InputSeqInThread[i].InputSeq[k];
        }
    }
    free(InputSeqInThread);
    __syncthreads();
}
