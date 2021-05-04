#include<iostream>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "cublas_v2.h"
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <iomanip>

#include "include/params.cuh"
#include "include/DataStructure.cuh"
#include "include/Matrix.cuh"
#include "include/init.cuh"
#include "include/SampleBasedHessian.cuh"
#include "include/costFunction.cuh"
#include "include/MCMPC.cuh"
   

int main(int argc, char **argv)
{
    /* 制御変数（状態・システムパラメータ・各種制約・重み行列）を定義して、デバイス変数としてコピー */ 
    Controller *hostCtr, *deviceCtr;
    hostCtr = (Controller*)malloc(sizeof(Controller));
    cudaMalloc(&deviceCtr, sizeof(Controller));
    initForSinglePendulum( hostCtr );
    CHECK_CUDA(cudaMemcpy(deviceCtr, hostCtr, sizeof(Controller), cudaMemcpyHostToDevice), "Failed to copy h to d Control parametes");
    // cudaMemcpy(&deviceCtr, &hostCtr, sizeof(Controller), cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol(deviceCtr, &hostCtr, sizeof(Controller));

    float hostState[DIM_OF_STATE], hostParam[NUM_OF_PARAMS];
    setInitState( hostState );
    setInitHostParam( hostParam );
 
    /* Setup Random sedd for Monte Carlo Method */ 
    unsigned int numBlocks, randomBlocks, randomNums;
    unsigned int qhpBlocks;
    int Blocks, sz_prm_qhp;
    randomNums = NUM_OF_SAMPLES * (DIM_OF_U + 1) * HORIZON;
    randomBlocks = countBlocks(randomNums, THREAD_PER_BLOCKS);
    numBlocks = countBlocks(NUM_OF_SAMPLES, THREAD_PER_BLOCKS);
    printf("#NumBlocks = %d\n", numBlocks);
    Blocks = numBlocks;
    sz_prm_qhp = SIZE_OF_PARABOLOIDVESTOR + AdditionalSampleSize;
    qhpBlocks = countBlocks(sz_prm_qhp, THREAD_PER_BLOCKS_SORT);
    printf("#phpBlocks = %d\n", qhpBlocks);

    curandState *devStates;
    CHECK_CUDA(cudaMalloc((void **)&devStates, randomNums * sizeof(curandState)), "Failed to setup Random seed");
    setup_kernel<<<randomBlocks, THREAD_PER_BLOCKS>>>(devStates,rand());

    /* データ書き込みファイルの定義 */ 
    FILE *fp;
    time_t timeValue;
    struct tm *timeObject;
    time( &timeValue );
    timeObject = localtime( &timeValue );
    char filename1[35];
    sprintf(filename1,"data_system_%d%d_%d%d.txt",timeObject->tm_mon + 1, timeObject->tm_mday, timeObject->tm_hour,timeObject->tm_min);
    fp = fopen(filename1,"w");

    MonteCarloMPC *device_MCMPC, *host_MCMPC, *device_Elite, *host_Elite;
    host_MCMPC = (MonteCarloMPC *)malloc(sizeof(MonteCarloMPC) * NUM_OF_SAMPLES);
    host_Elite = (MonteCarloMPC *)malloc(sizeof(MonteCarloMPC) * NUM_OF_ELITESAMPLE);
    CHECK_CUDA(cudaMalloc(&device_MCMPC, sizeof(MonteCarloMPC) * NUM_OF_SAMPLES), "Failed to allocate the device sample information");
    CHECK_CUDA(cudaMalloc(&device_Elite, sizeof(MonteCarloMPC) * NUM_OF_ELITESAMPLE), "Failed to allocate device elite sample information");
    setupMonteCarloMPCVectors<<<numBlocks, THREAD_PER_BLOCKS>>>(device_MCMPC);
    cudaDeviceSynchronize();

    SampleBasedHessian *deviceDataForHessian, *hostDataForHessian;
    hostDataForHessian = (SampleBasedHessian *)malloc(sizeof(SampleBasedHessian) * (HORIZON + 1) );
    CHECK_CUDA(cudaMalloc(&deviceDataForHessian, sizeof(SampleBasedHessian) * (HORIZON + 1) ), "Failed to allocate device Hessian information");

    HyperParaboloid *deviceParaboloid, *hostParaboloid;
    hostParaboloid = (HyperParaboloid *)malloc(sz_prm_qhp * sizeof(HyperParaboloid));
    CHECK_CUDA(cudaMalloc(&deviceParaboloid, sz_prm_qhp * sizeof(HyperParaboloid)), "Failed to allocate device Paraboloid information");
    float *hostProposedInput;
    float *deviceProposedInput;
    hostProposedInput = (float *)malloc(HORIZON * sizeof(float));
    CHECK_CUDA(cudaMalloc(&deviceProposedInput, HORIZON * sizeof(float)), "Failed to allocate deviceProposedInput");

    InputSequences *hostInputSBH, *hostInputMCMPC, *deviceInputSBH, *deviceInputMCMPC, *hostInput, *deviceInput;
    hostInputSBH = (InputSequences *)malloc(sizeof(InputSequences) * HORIZON);
    hostInputMCMPC = (InputSequences *)malloc(sizeof(InputSequences) * HORIZON);
    hostInput = (InputSequences *)malloc(sizeof(InputSequences) * HORIZON);
    cudaMalloc(&deviceInputSBH, sizeof(InputSequences) * HORIZON);
    cudaMalloc(&deviceInputMCMPC, sizeof(InputSequences) * HORIZON);
    cudaMalloc(&deviceInput, sizeof(InputSequences) * HORIZON);


    /* Hessianの逆行列を格納する配列 */
    float *invHessian, *deviceInvHessian;
    invHessian = (float *)malloc(sizeof(float) * HORIZON * HORIZON);
    CHECK_CUDA( cudaMalloc(&deviceInvHessian, HORIZON * HORIZON * sizeof(float)), "Failed to allocate device inv Hessian in main.cu");

    /* variables for sort */
    thrust::host_vector<int> indices_host_vec( NUM_OF_SAMPLES );
    thrust::device_vector<int> indices_device_vec = indices_host_vec;
    thrust::host_vector<float> sort_key_host_vec( NUM_OF_SAMPLES );
    thrust::device_vector<float> sort_key_device_vec = sort_key_host_vec;

    /* dim3 for pseudo gradient calculation */
    int nx = 3 * (HORIZON + 1);
    int ny = HORIZON;
    dim3 hess_block(3,1);
    dim3 hess_grid(( nx  + hess_block.x - 1)/ hess_block.x, ( ny + hess_block.y -1) / hess_block.y);
    printf("inx == %d  iny == %d\n",(int)(( nx  + hess_block.x - 1)/ hess_block.x),  (int)(( ny + hess_block.y -1) / hess_block.y));
    int n = 2;
    float Mat[4] = {-11.996, -32.664, -5.332, -11.332};
    float *invMat;
    invMat = (float*)malloc(n * n * sizeof(float));
    GetInvMatrix(invMat, Mat, n);

    printMatrix(n, n, invMat, n, "invMat");

    printf("done\n");

    float variance;
    float costMCMPC;
    float costSBH;
    float now_input, inferior_input;
    int counter;

    // 時間計測用
    float process_gpu_time, procedure_time;
    clock_t start_t, stop_t;
    cudaEvent_t start, stop;

    for(int t = 0; t < TIME; t++){
        shift_Input_vec( hostInput, 0 );
        StateUpdate( hostCtr, hostState); //この関数の宣言と実体の記述から　2021.4.27 10:55
        // cudaMemcpy(deviceInput, hostInput, sizeof(float) * HORIZON, cudaMemcpyHostToDevice);
        // 時間計測スタート
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        start_t = clock();
        for(int re = 0; re < NUM_OF_RECALC; re++)
        {
            // Parallel Simulation 〜 MCMPC解の推定まで
            variance = powf(0.95, re) * initVar;
            CHECK_CUDA(cudaMemcpy(deviceCtr, hostCtr, sizeof(Controller), cudaMemcpyHostToDevice), "failed to copy control information at control loop");
            MCMPC_Cart_and_Single_Pole<<<numBlocks, THREAD_PER_BLOCKS>>>(device_MCMPC, devStates, deviceCtr, deviceInput, variance, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
            cudaDeviceSynchronize();
            thrust::sequence(indices_device_vec.begin(), indices_device_vec.end());
            thrust::sort_by_key(sort_key_device_vec.begin(), sort_key_device_vec.end(), indices_device_vec.begin());
            MCMPC_callback_elite_sample<<<NUM_OF_ELITESAMPLE, 1>>>(device_Elite, device_MCMPC, thrust::raw_pointer_cast(indices_device_vec.data()));
            cudaMemcpy(host_MCMPC, device_MCMPC, sizeof(MonteCarloMPC) * NUM_OF_SAMPLES, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_Elite, device_Elite, sizeof(MonteCarloMPC) * NUM_OF_ELITESAMPLE, cudaMemcpyDeviceToHost);
            MCMPC_by_weighted_mean(hostInputMCMPC, host_Elite, 0);

            // Sampled Hessianの計算
            CHECK_CUDA(cudaMemcpy( deviceInputMCMPC, hostInputMCMPC, sizeof(InputSequences) * HORIZON, cudaMemcpyHostToDevice),"Failed befor call function -2");
            MCMPC_Cart_and_Single_Pole<<<numBlocks, THREAD_PER_BLOCKS>>>(device_MCMPC, devStates, deviceCtr, deviceInputMCMPC, iita, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
            thrust::sequence(indices_device_vec.begin(), indices_device_vec.end());
            thrust::sort_by_key(sort_key_device_vec.begin(), sort_key_device_vec.end(), indices_device_vec.begin());

            // ここから工事開始
#ifdef PARABOLOID_FITTING_HESSIAN
            make_tensor_vector<<< qhpBlocks, THREAD_PER_BLOCKS_SORT >>>(deviceParaboloid, device_MCMPC, thrust::raw_pointer_cast( indices_device_vec.data() ));
            CHECK_CUDA(cudaMemcpy(hostParaboloid, deviceParaboloid, sz_prm_qhp * sizeof(HyperParaboloid), cudaMemcpyDeviceToHost),"Failed to copy Paraboloid Info device ==> host");
            GetInputByLSMfittingMethod(hostProposedInput, hostParaboloid, SIZE_OF_PARABOLOIDVESTOR, HORIZON, sz_prm_qhp);
            // ここまで工事完了
            // 2021.05.04からの作業概要
            // コスト関数の計算　float *V を引数にしたversionの作成
            // GetInputByLSMfittingMethod　のヘシアンの逆行列の計算をcuBLASベースのものにした関数の実装
            // 計算時間の計測用　変数の宣言と計測の実装
            costSBH = calc_Cost_Cart_and_SinglePole_Single(hostCtr, hostProposedInput);
#else
            ParallelSimForPseudoGrad<<<hess_grid, hess_block>>>(deviceDataForHessian, device_MCMPC, deviceInputMCMPC, deviceCtr, zeta, thrust::raw_pointer_cast( indices_device_vec.data() ));
            CHECK_CUDA(cudaDeviceSynchronize(),"cudaSynchronize main #1\n");
            getPseduoGradient<<<HORIZON + 1, HORIZON>>>( deviceDataForHessian, zeta);
            cudaDeviceSynchronize();
            CHECK_CUDA(cudaMemcpy( hostDataForHessian, deviceDataForHessian, sizeof(SampleBasedHessian) * (HORIZON + 1), cudaMemcpyDeviceToHost ), "Failed befor call function");
            printf("hoge %d\n", re);
            getInvHessian( invHessian, hostDataForHessian);
            CHECK_CUDA( cudaMemcpy(deviceDataForHessian, hostDataForHessian, sizeof(SampleBasedHessian) * (HORIZON + 1), cudaMemcpyHostToDevice), "Failed to copy Hessian information to device vector" );
            CHECK_CUDA( cudaMemcpy(deviceInvHessian, invHessian, HORIZON * HORIZON * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy inverse Hessian to device vector");
            getCurrentUpdateResult<<<HORIZON, HORIZON + 1>>>( deviceDataForHessian,  deviceInvHessian );
            // もし、以降で、上位HORIZON+1個のサンプルのシミュレーション結果を基にシミュレーション→sortを行うなら記述
            // 2021.4.26現在は、上記を行なわないバージョン
            copyInpSeqFromSBH<<<HORIZON, 1>>>( deviceInputSBH, deviceDataForHessian);
            CHECK_CUDA( cudaMemcpy(hostInputSBH, deviceInputSBH, HORIZON * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy InputSeq From SBH method");

            // MCMPC解とSampleBasedHessian解の評価関数を計算
            costSBH = calc_Cost_Cart_and_SinglePole( hostCtr, hostInputSBH );
#endif
            costMCMPC = calc_Cost_Cart_and_SinglePole( hostCtr, hostInputMCMPC );
            if(costMCMPC < costSBH)
            {
                CHECK_CUDA( cudaMemcpy( deviceInput, hostInputMCMPC, HORIZON * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy MCMPC Input vector" );
                // printf("MCMPC input superior than SBH\n");
            }else{
#ifdef PARABOLOID_FITTING_HESSIAN
                CHECK_CUDA(cudaMemcpy(deviceProposedInput, hostProposedInput, HORIZON * sizeof(float), cudaMemcpyHostToDevice),"Failed to copy ProposedInputSeq host to device at main.cu");
                copySingleDateToInputSequences<<<HORIZON, 1>>>(deviceInput, deviceProposedInput);
                cudaDeviceSynchronize();
#else
                CHECK_CUDA( cudaMemcpy( deviceInput, hostInputSBH, HORIZON * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy SBH Input vector");
#endif
            }
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&process_gpu_time, start, stop);
        stop_t = clock();
        procedure_time = stop_t - start_t;

        if(costMCMPC < costSBH || isnan(costSBH))
        {
            now_input = hostInputMCMPC[0].InputSeq[0];
#ifdef PARABOLOID_FITTING_HESSIAN
            inferior_input = hostProposedInput[0];
#else
            inferior_input = hostInputSBH[0].InputSeq[0];
#endif
            counter = 0;
            CHECK_CUDA( cudaMemcpy( deviceInput, hostInputMCMPC, HORIZON * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy MCMPC Input vector" );
            printf("MCMPC input superior than SBH\n");
        }else{
#ifdef PARABOLOID_FITTING_HESSIAN
            now_input = hostProposedInput[0];
            inferior_input = hostInputMCMPC[0].InputSeq[0];
            counter = 1;
            CHECK_CUDA(cudaMemcpy(deviceProposedInput, hostProposedInput, HORIZON * sizeof(float), cudaMemcpyHostToDevice),"Failed to copy ProposedInputSeq host to device at main.cu");
            copySingleDateToInputSequences<<<HORIZON, 1>>>(deviceInput, deviceProposedInput);
            cudaDeviceSynchronize();
            printf("SBH input superior than MCMPC\n"); 
#else
            now_input = hostInputSBH[0].InputSeq[0];
            inferior_input = hostInputMCMPC[0].InputSeq[0];
            counter = 1;
            CHECK_CUDA( cudaMemcpy( deviceInput, hostInputSBH, HORIZON * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy SBH Input vector");
            printf("SBH input superior than MCMPC\n");
#endif
        }
        Runge_kutta_45_for_Secondary_system(hostState, now_input, hostParam, interval);
#ifdef COLLISION
        if(hostState[0] <= hostConstraint[2]){
            float hcollide[3] = { };
            hcollide[0] = hostParam[0] * hostParam[1] * cosf(hostState[1]);
            hcollide[1] = hostParam[2] + hostParam[0] * powf(hostParam[1],2);
            hcollide[2] = hcollide[0] / hcollide[1];
            hostState[3] = hostState[3] + (1 + hostParam[7]) * hcollide[2] * hostState[2]; //dtheta = dtheta + (1+e) * F * dx
            hostState[2] = -hostParam[7] * hostState[2]; // dx = -e * dx
            hostState[0] = hostConstraint[2];
        }
        if(hostConstraint[3] <=  hostState[0]){
            float hcollide[3] = { };
            hcollide[0] = hostParam[0] * hostParam[1] * cosf(hostState[1]);
            hcollide[1] = hostParam[2] + hostParam[0] * powf(hostParam[1],2);
            hcollide[2] = hcollide[0] / hcollide[1];
            hostState[3] = hostState[3] + (1 + hostParam[7]) * hcollide[2] * hostState[2]; //dtheta = dtheta + (1+e) * F * dx
            hostState[2] = -hostParam[7] * hostState[2]; // dx = -e * dx
            hostState[0] = hostConstraint[3];
        }
#endif
        fprintf(fp, "%f %f %f %f %f %f %f %f %f %d %f %f\n", interval * t, now_input, inferior_input, hostState[0], hostState[1], hostState[2], hostState[3], costMCMPC, costSBH, counter, process_gpu_time /1000, procedure_time);
        printf("%f %f %f %f\n", interval * t, now_input, costMCMPC, costSBH);
    }
    fclose(fp);
    cudaDeviceReset( );
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}


