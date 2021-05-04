#include<math.h>
#include<cuda.h>
#include "params.cuh"


#ifndef DYNAMICS_CUH
#define DYNAMICS_CUH
// __host__ __device__ void calc_Linear_example(float *state, float input, float *param, float *ret);
__host__ __device__ void calc_nonLinear_example(float *state, float input, float *param, float *ret);
__host__ __device__ float Cart_type_Pendulum_ddx(float u, float x, float theta, float dx, float dtheta, float *state);
__host__ __device__ float Cart_type_Pendulum_ddtheta(float u, float x, float theta, float dx, float dtheta, float *state);

void get_current_diff_state(float *state, float input, float *param, float *diff_state);
void simple_integrator(float *diff_state, float c_sec, float *yp_vector);
void Runge_kutta_45_for_Secondary_system(float *state, float input, float *param, float c_sec);

#endif

