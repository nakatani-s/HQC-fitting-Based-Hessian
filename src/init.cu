/*
    definition file for System & constraints parameters 
*/
#include "../include/init.cuh"

void init_state(float *a)
{
    // FOR CART AND POLE
    a[0] = 0.0f; //x
    a[1] = M_PI; //theta
    a[2] = 0.0f; //dx
    a[3] = 0.0f; //dth
}

void init_param(float *a)
{
#ifdef COLLISION
    a[0] = 0.1f;
    a[1] = 0.024f;
    a[2] = 0.2f;
    a[3] = a[1] * powf(a[2],2) /3;
    a[4] = 1.265f;
    a[5] = 0.0000001;
    a[6] = 9.81f;
    a[7] = 0.48; //反発係数
#else
    a[0] = 0.1f;
    a[1] = 0.024f;
    a[2] = 0.2f;
    a[3] = a[1] * powf(a[2],2) /3;
    a[4] = 1.265f;
    a[5] = 0.0000001;
    a[6] = 9.81f;
#endif
}

void init_constraints(float *a)
{
    // FOR CONTROL CART AND POLE
    a[0] = -2.0f;
    a[1] = 2.0f;
    a[2] = -1.0f;
    a[3] = 1.0f;
}

void init_weightMatrix(float *a)
{
    // FOR CAONTROL CART AND POLE
    a[0] = 3.0f;
    a[1] = 3.0f;
    a[2] = 0.04f;
    a[3] = 0.05f;
    a[4] = 0.5f;
}

void setInitState(float *st)
{
    init_state( st );
}

void setInitHostParam( float *Prm )
{
    init_param( Prm );
}

void initForSinglePendulum(Controller *ct)
{
    float *st, *pa, *co, *wm;
    st = (float *)malloc(sizeof(float) * DIM_OF_STATE);
    pa = (float *)malloc(sizeof(float) * NUM_OF_PARAMS);
    co = (float *)malloc(sizeof(float) * NUM_OF_CONSTRAINTS);
    wm = (float *)malloc(sizeof(float) * DIM_OF_WEIGHT_MATRIX);
    init_state( st );
    init_constraints( co );
    init_param( pa );
    init_weightMatrix( wm );

    for(int i = 0; i < DIM_OF_STATE; i++){
        ct->State[i] = st[i];
    }
    for(int i = 0; i < NUM_OF_PARAMS; i++){
        ct->Param[i] = pa[i]; 
    }
    for(int i = 0; i < NUM_OF_CONSTRAINTS; i++){
        ct->Constraints[i] = co[i];
    }
    for(int i = 0; i < DIM_OF_WEIGHT_MATRIX; i++){
        ct->WeightMatrix[i] = wm[i];
    }
}
