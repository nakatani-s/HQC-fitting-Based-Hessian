/*
only include "../include/dynamics.cuh"
*/
#include "../include/dynamics.cuh"



__host__ __device__ void calc_nonLinear_example(float *state, float input, float *param, float *ret)
{
    float temp[DIM_OF_STATE];
    temp[0] = state[1];
    temp[1] = ( 1- powf(state[0],2) - powf(state[1],2) ) * state[1] - state[0] + input;
    //temp[1] = -state[0] - state[1] + input;
    /*temp[0] = param[0] * state[0] + param[1] * state[1] + param[2] * state[2] + param[9] * input;
    temp[1] = param[3] * state[0] + param[4] * state[1] + param[5] * state[2] + param[10] * input;
    temp[2] = param[6] * state[0] + param[7] * state[1] + param[8] * state[2] + param[11] * input;*/
    
    for(int d = 0; d < DIM_OF_STATE; d++){
        ret[d] = temp[d];
    }
}


__host__ __device__ float Cart_type_Pendulum_ddx(float u, float x, float theta, float dx, float dtheta, float* state)
{
    float a[10];

    a[0] = state[3] + powf(state[2], 2) * state[1];		//J+l^2*mp
	a[1] = u - dx * state[4]
			+ powf(dtheta, 2) * state[2] * state[1] * sinf(theta);//u-dx*myuc+dtheta^2*l*mp*sin
	a[2] = cosf(theta) * state[2] * state[1];						//cos*l*mp
	a[3] = dtheta * state[5] - state[6] * state[2] * state[1] * sinf(theta);//dtheta*myup-g*l*mp*sin
	a[4] = -(a[0] * a[1] + a[2] * a[3]);

	a[5] = powf(cosf(theta), 2) * powf(state[2], 2) * powf(state[1], 2);//cos^2*l^2*mp^2
	a[6] = state[0] + state[1];		//mc+mp
	a[7] = state[3] + powf(state[2], 2) * state[1];		//J+l^2*mp
	a[8] = a[5] - (a[6] * a[7]);

	return a[4] / a[8];
}

__host__ __device__ float Cart_type_Pendulum_ddtheta(float u, float x, float theta, float dx, float dtheta, float* state)
{
    float a[10];
    a[0] = cosf(theta) * state[2] * state[1];		//cos*l*mp
	a[1] = u - dx * state[4]
			+ powf(dtheta, 2) * state[2] * state[1] * sinf(theta);//u-dx*myuc+dtheta^2*l*mp*sin
	a[2] = state[0] + state[1];		//mc+mp
	a[3] = dtheta * state[5] - state[6] * state[2] * state[1] * sinf(theta);//dtheta*myup-g*l*mp*sin
	a[4] = -(a[0] * a[1] + a[2] * a[3]);

	a[5] = state[3] * (state[0] + state[1]);		//J(mc+mp)
	a[6] = powf(state[2], 2) * state[1];		//l^2*mp
	a[7] = state[0] + state[1] - powf(cosf(theta), 2) * state[1];//mc+mp-cos^2*mp
	a[8] = a[5] + a[6] * a[7];

	return a[4] / a[8];
}

// Runge_Kutta_Function
void get_current_diff_state(float *state, float input, float *param, float *diff_state){
    /*-- p_state[DIM_X] = {dx, dtheta, ddx, ddtheta}--*/
    /*-- state[DIM_X] = {x, theta, dx, dtheta} --*/ 
    /*-- input[DIM_U] = { u } --*/
    /*-- param[NUM_OF_SYS_PARAMETERS] = {m_c, m_p, l_p, J, myu_c, myu_p, g}--*/
    
    // For Cart type Single inverted pendulum
    diff_state[0] = state[2];
    diff_state[1] = state[3];
    diff_state[2] = Cart_type_Pendulum_ddx(input, state[0], state[1], state[2], state[3], param);
    diff_state[3] = Cart_type_Pendulum_ddtheta(input, state[0], state[1], state[2], state[3], param);
    //calc_nonLinear_example(state, input, param, diff_state);
}

void simple_integrator(float *diff_state, float c_sec, float *yp_vector){
    for(int i = 0; i < DIM_OF_STATE; i++){
        yp_vector[i] = diff_state[i] * c_sec;
    }
}

void Runge_kutta_45_for_Secondary_system(float *state, float input, float *param, float c_sec)
{
    float diff_state[DIM_OF_STATE], yp_1[DIM_OF_STATE], next_state[DIM_OF_STATE];
    get_current_diff_state(state, input, param, diff_state);
    //calc_nonLinear_example(state, input, param, diff_state);
    simple_integrator(diff_state, c_sec, yp_1);
    for(int i = 0; i < DIM_OF_STATE; i++){
        next_state[i] = state[i] + yp_1[i] / 2;
    }
    float yp_2[DIM_OF_STATE];
    get_current_diff_state(next_state, input, param, diff_state);
    //calc_nonLinear_example(next_state, input, param, diff_state);
    simple_integrator(diff_state, c_sec, yp_2);
    for(int i = 0; i < DIM_OF_STATE; i++){
        next_state[i] = state[i] + yp_2[i] / 2;
    }
    float yp_3[DIM_OF_STATE];
    get_current_diff_state(next_state, input, param, diff_state);
    //calc_nonLinear_example(next_state, input, param, diff_state);
    simple_integrator(diff_state, c_sec, yp_3);
    for(int i = 0; i < DIM_OF_STATE; i++){
        next_state[i] = state[i] + yp_3[i];
    }

    float yp_4[DIM_OF_STATE];
    get_current_diff_state(next_state, input, param, diff_state);
    //calc_nonLinear_example(next_state, input, param, diff_state);
    simple_integrator(diff_state, c_sec, yp_4);

    for(int i = 0; i < DIM_OF_STATE; i++){
        state[i] = state[i] + (yp_1[i] + 2*yp_2[i] + 2*yp_3[i] + yp_4[i]) / 6;
    }
}