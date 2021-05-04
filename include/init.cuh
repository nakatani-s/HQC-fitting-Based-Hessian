/*
    Functions for matrix operations
    #using cuda function
    #using cuBLAS
    #using cuSOLVER
*/
#include <math.h>
#include "params.cuh"
#include "DataStructure.cuh"

void setInitState(float *st);
void setInitHostParam( float *Prm );
void initForSinglePendulum(Controller *ct);