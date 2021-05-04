/* cost function calclated in main function */
#include <math.h>
#include <stdio.h>
#include "params.cuh"
#include "DataStructure.cuh"
#include "dynamics.cuh"

float calc_Cost_Cart_and_SinglePole_Single( Controller *CtrPrm,float *inputSeq);
float calc_Cost_Simple_NonLinear_Example( float *inputSequences, float *stateValues, float *param, float *weightMatrix);
float calc_Cost_Cart_and_SinglePole( Controller *CtrPrm, InputSequences *Input );