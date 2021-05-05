/*
params.cuh
*/ 

#ifndef PARAMS_CUH
#define PARAMS_CUH

#define PARABOLOID_FITTING_HESSIAN

const int DIM_OF_STATE = 4;
const int DIM_OF_U = 1;
const int NUM_OF_CONSTRAINTS = 4;
const int DIM_OF_WEIGHT_MATRIX = 5;
#ifdef COLLISION
const int NUM_OF_PARAMS = 8; 
#else
const int NUM_OF_PARAMS = 7;
#endif

// GPU parameters
const int NUM_OF_SAMPLES = 2500;
const int NUM_OF_ELITESAMPLE = 13;
const int THREAD_PER_BLOCKS = 10;



// MPC parameters
const int TIME = 1000;
const int HORIZON = 30;
const int NUM_OF_RECALC = 1;
const float interval = 0.01;
const float invBarrier = 5000;
const float zeta = 0.01f;
const float iita = 0.6f;

// Sample Based Hessian Method Parameters
const int THREAD_PER_BLOCKS_SORT = 10;

// HORIZON 25 ==>  351
// HORIZON 30 ==>  496
// HORIZON 35 ==>  666
// HORIZON 40 ==>  861
const int SIZE_OF_PARABOLOIDVESTOR = (HORIZON * HORIZON + 3 * HORIZON + 2) / 2;

//  SIZE_OF_PARABOLOIDVESTOR + AdditionalSampleSize が　THREAD_PER_BLOCKS_SORTの定数倍　かつ　NUM_OF_SAMPLES以下　になるように設定
const int AdditionalSampleSize = 144; 

const int LIMIT_OF_THREAD_PER_BLOCK = 1024;


const float initVar = 2.0f;

#endif
