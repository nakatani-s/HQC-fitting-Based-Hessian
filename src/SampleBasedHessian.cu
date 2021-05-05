/*
    Functions for making samplebased hessian
*/
#include "../include/SampleBasedHessian.cuh"

__global__ void makeGrad_SamplePointMatrix(float *G, float *SPM, SampleBasedHessian *Subject)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    G[id] = Subject[0].pseudoGradient[threadIdx.x] - Subject[blockIdx.x + 1].pseudoGradient[threadIdx.x];
    SPM[id] = Subject[0].currentU[0][threadIdx.x] - Subject[blockIdx.x + 1].currentU[0][threadIdx.x];
    __syncthreads();
}

void getInvHessian( float *Hess, SampleBasedHessian *hostHess)
{
    // cublasHandle_t handle_invHess;
    // CUBLAS_CHECK( cublasCreate(&cublas_status), "Failed to initialize cuBLAS");
    SampleBasedHessian *deviceSubject;
    float *deviceArrayHess;
    float *arrayGrad, *device_arrayGrad;
    float *invArrayGrad, *deviceInvArrayGrad;
    float /* *arrayVect, */ *device_arrayVect;

    size_t szMat2 = HORIZON * HORIZON * sizeof(float);

    int nx = HORIZON;
    int ny = HORIZON;
    dim3 block(1,1);
    dim3 grid(( nx  + block.x - 1)/ block.x, ( ny + block.y -1) / block.y);

    arrayGrad = (float *)malloc(HORIZON * HORIZON * sizeof(float));
    CHECK_CUDA(cudaMalloc(&device_arrayGrad,  szMat2), "Failed to allocate array Grad on device matrix" );
    CHECK_CUDA( cudaMalloc( &deviceArrayHess, HORIZON * HORIZON * sizeof(float)), "Failed to allocate array Hessian on device matrix");
    CHECK_CUDA( cudaMalloc( &device_arrayVect, szMat2), "Failed to allocate array Vect on device matrix");
    CHECK_CUDA( cudaMalloc( &deviceInvArrayGrad, szMat2), "Failed to allocate array invGrad on device matrix");
    CHECK_CUDA( cudaMalloc( &deviceSubject, (HORIZON + 1) * sizeof(SampleBasedHessian)), "Failed to allocate SampledBasedHessianVector in SBH.cu");
    CHECK_CUDA( cudaMemcpy(deviceSubject, hostHess, (HORIZON + 1) * sizeof(SampleBasedHessian), cudaMemcpyHostToDevice), "Failed to copy SampledBasedHessianVector in SBH.cu");
    invArrayGrad = (float *)malloc( szMat2 );
    // arrayVect = (float *)malloc( szMat );

    makeGrad_SamplePointMatrix<<<HORIZON, HORIZON>>>( device_arrayGrad, device_arrayVect, deviceSubject );
    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after kernel call!");
    CHECK_CUDA( cudaMemcpy( arrayGrad, device_arrayGrad, szMat2, cudaMemcpyDeviceToHost), "Failed to copy matrix G in SBH.cu");
    // CHECK_CUDA( cudaMemcpy( arrayVect, device_arrayVect, szMat, cudaMemcpyDeviceToHost), "Failed to copy matrix V in SBH.cu");

    GetInvMatrix(invArrayGrad, arrayGrad, HORIZON);
    CHECK_CUDA( cudaMemcpy(deviceInvArrayGrad, invArrayGrad, szMat2, cudaMemcpyHostToDevice), "Failed to copy inverse matrix G to device");

    GetResultMatrixProduct<<<grid, block>>>( deviceArrayHess, device_arrayVect, deviceInvArrayGrad, HORIZON );
    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after kernel call!");

    CHECK_CUDA( cudaMemcpy(Hess, deviceArrayHess, szMat2, cudaMemcpyDeviceToHost), "Failed to copy inverse Hess to Host");

    CHECK_CUDA(cudaFree(deviceArrayHess),"Failed to free deviceArrayHess");
    CHECK_CUDA(cudaFree(device_arrayGrad),"Failed to free device_arrayGrad");
    CHECK_CUDA(cudaFree(device_arrayVect), "Failed to free device_arrayVect");
    CHECK_CUDA(cudaFree(deviceInvArrayGrad), "Failed to free deviceInvArrayGrad");
    CHECK_CUDA(cudaFree(deviceSubject), "Failed to free deviceSubject");
    free(arrayGrad);
    free(invArrayGrad);

}

__device__ void readParam(float *prm, Controller *CtrPrm){
    for(int i = 0; i < NUM_OF_PARAMS; i++)
    {
        prm[i] = CtrPrm->Param[i];
    }
}

__global__ void getPseduoGradient(SampleBasedHessian  *Hess, float epsilon)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    float temp[2] = { };

    temp[0] = Hess[blockIdx.x].cost[threadIdx.x][1] - Hess[blockIdx.x].cost[threadIdx.x][2];
    temp[1] = 2 * epsilon;
    Hess[blockIdx.x].pseudoGradient[threadIdx.x] = temp[0] / temp[1];
    __syncthreads();
}


__global__ void getCurrentUpdateResult( SampleBasedHessian *HessInfo, float *invHess )
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;

    float delta_u = 0.0f;
    float result_u = 0.0f;

    // 以上なidの時は処理しないおまじない
    if(threadIdx.x < HORIZON && blockIdx.x < HORIZON)
    {
        for(int i = 0; i < HORIZON; i++)
        {
            delta_u += invHess[ blockIdx.x * HORIZON + i] * HessInfo[threadIdx.x].pseudoGradient[i];
        }
        result_u = HessInfo[threadIdx.x].currentU[0][blockIdx.x] - delta_u; // current_u - H^-1*g = tilde{u}^{*}
        HessInfo[threadIdx.x].modified_U[0][blockIdx.x] = result_u;
        HessInfo[threadIdx.x].delta_u[0][blockIdx.x] = delta_u;
    }
    __syncthreads();
}

__global__ void copyInpSeqFromSBH( InputSequences *Output, SampleBasedHessian *HessInfo)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;

    if(id < HORIZON)
    {
        Output[id].InputSeq[0] = HessInfo[0].modified_U[0][id];
    }
    __syncthreads();
}

/*__global__ void getRegularMatrix(float *outRmatrix, HyperParaboloid *elements, int sumSet)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    outRmatrix[id] = 0.0f;
    //float temp_here = 0.0f;
    for(int index = 0; index < sumSet; index++){
        outRmatrix[id] += elements[index].tensor_vector[threadIdx.x] * elements[index].tensor_vector[blockIdx.x];
        //float temp_here +=  
    }
    //printf("id==%d, ans == %f\n", id, outRmatrix[id]);
    __syncthreads();
}


__global__ void getRegularMatrix_overLimit(float *outRmatrix, HyperParaboloid *elements, int sumSet, int Ydimention)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int id = iy * Ydimention + ix;
    outRmatrix[id] = 0.0f;
    for(int index = 0; index < sumSet; index++){
        outRmatrix[id] += elements[index].tensor_vector[ix] * elements[index].tensor_vector[iy];
    }
    __syncthreads();
}*/

__global__ void getRegularVector(float *outRvector, HyperParaboloid *elements, int sumSet)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    outRvector[id] = 0.0f;
    for(int index = 0; index < sumSet; index++)
    {
        outRvector[id] += elements[index].column_vector[id];
    }
    __syncthreads();
}

__global__ void copySingleDateToInputSequences( InputSequences *Output, float *Sequences)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    Output[id].InputSeq[0] = Sequences[id];
    __syncthreads();
}

void GetInputByLSMfittingMethod(float *OutPut, HyperParaboloid *hostHP, int sz_QC, int sz_HESS, int sz_LSM)
{
   

    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t handle_cublas = NULL;

    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    size_t szMatG = sz_QC * sz_QC * sizeof(float);
    size_t szMatHESS = sz_HESS * sz_HESS * sizeof(float);

    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH), "Failed to initialize cuSOLVER Handle_t");
    CHECK_CUBLAS(cublasCreate(&handle_cublas),"Failed to initialize cuBLAS");

    HyperParaboloid *deviceHP;
    CHECK_CUDA(cudaMalloc(&deviceHP, sz_LSM * sizeof(HyperParaboloid)), "Failed to allocate device HyperParaboloid data!");
    CHECK_CUDA(cudaMemcpy(deviceHP, hostHP, sz_LSM * sizeof(HyperParaboloid), cudaMemcpyHostToDevice), "Failed to copy device HyperParaboloid data!");

    float *d_G;
    float *d_InverseG;
    float *d_Rvect;
    float *d_ansRvect;
    float *d_Hess;
    float *d_transH;
    float *d_vectorB;
    float *d_Input;

    int prmszQHP = sz_QC + AdditionalSampleSize;
    int sz_HessianElements = sz_QC - (sz_HESS + 1);

    CHECK_CUDA(cudaMalloc(&d_G, szMatG), "Failed to allocate d_G !");
    CHECK_CUDA(cudaMalloc(&d_InverseG, szMatG), "Failed to allocate d_InverseG !");
    CHECK_CUDA(cudaMalloc(&d_Rvect, sz_QC * sizeof(float)),"Failed to allocate d_Rvect !");
    CHECK_CUDA(cudaMalloc(&d_ansRvect, sz_QC * sizeof(float)),"Failed to allocate d_Rvect !");
    CHECK_CUDA(cudaMalloc(&d_vectorB, sz_HESS * sizeof(float)),"Failed to allocate d_vectorB !");
    CHECK_CUDA(cudaMalloc(&d_Input, sz_HESS * sizeof(float)), "Failed to allocate d_Input !");
    CHECK_CUDA(cudaMalloc(&d_Hess, szMatHESS), "Failed to allocate d_Hess !!");
    CHECK_CUDA(cudaMalloc(&d_transH ,szMatHESS), "Failed to allocate d_transH !!")

    dim3 Block(2,2);
    dim3 Grid((sz_QC + Block.x -1) / Block.x, (sz_QC + Block.y -1)/ Block.y);

    int work_size, work_sizeForHess;
    float *work_space, *work_spaceForHess;
    int *devInfo;
    CHECK_CUDA(cudaMalloc((void**)&devInfo, sizeof(int)), "Failed to allocate devInfo at Func # GetInputByLSMfittingMethod #");
    
    float alpha = 1.0f;
    float beta = 0.0f;
    clock_t start_t, stop_t;
    float get_time;
    start_t = clock();
    
    if(sz_QC > LIMIT_OF_THREAD_PER_BLOCK)
    {
        getRegularMatrix_overThreadLimit<<<Grid, Block>>>( d_G, deviceHP, prmszQHP, sz_QC);
        // getRegularMatrix_overThreadLimit<<<Grid, Block>>>( d_G, deviceHP, prmszQHP, sz_QC);
    }else{
        getRegularMatrix<<<sz_QC ,sz_QC>>>( d_G, deviceHP, prmszQHP);
    }

    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after setting Matrix d_G !");
    getRegularVector<<<sz_QC, 1>>>( d_Rvect, deviceHP , prmszQHP);
    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after setting Vector d_Rvect !");

    CHECK_CUSOLVER(cusolverDnSpotrf_bufferSize(cusolverH, uplo, sz_QC, d_G, sz_QC, &work_size),"Failed to get work_size !");
    CHECK_CUDA(cudaMalloc((void**)&work_space, sizeof(float)*work_size), "Failed to allocate work_space at cuSOLVER");
    CHECK_CUSOLVER(cusolverDnSpotrf(cusolverH, uplo, sz_QC, d_G, sz_QC, work_space, work_size, devInfo), "Failed to Function # cusolverDnSpotrf #");

    if(sz_QC > LIMIT_OF_THREAD_PER_BLOCK)
    {
        SetUpIdentity_Matrix_overThreadLimit<<<Grid, Block>>>( d_InverseG, sz_QC);
    }else{
        SetUpIdentity_Matrix<<<sz_QC ,sz_QC>>>( d_InverseG );
    }
    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after setting Identity Matrix !");

    CHECK_CUSOLVER(cusolverDnSpotrs(cusolverH, uplo, sz_QC, sz_QC, d_G, sz_QC, d_InverseG, sz_QC, devInfo),"Failed to perform Inverse operation at cuSOLVER!");

    CHECK_CUBLAS(cublasSgemv(handle_cublas, CUBLAS_OP_N, sz_QC, sz_QC, &alpha, d_InverseG, sz_QC, d_Rvect, 1, &beta, d_ansRvect, 1), "Failed to product G^-1 * Rv by cuBLAS");

    //CHECK_CUDA(cudaFree(d_G), "Failed to free d_G");
    CHECK_CUDA(cudaFree(d_InverseG), "Failed to free d_InverseG");
    CHECK_CUDA(cudaFree(d_Rvect), "Failed to free d_Rvect");
    CHECK_CUDA(cudaFree(work_space), "Failed to free work_space at GetInputByLSMfittingMethod");

    get_FullHessian_Elements<<<sz_HESS, sz_HESS>>>( d_Hess, d_ansRvect);
    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after setting Upper Triangle Hessian Matrix !");
    transpose_opration_Matrix<<<sz_HESS, sz_HESS>>>(d_transH, d_Hess);
    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after Transposed Upper Triangle Hessian Matrix !");
    make_symmetric_Matrix<<<sz_HESS, sz_HESS>>>(d_transH, d_Hess);
    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after setting Symmetric Hessian Matrix !");
    // ヘシアンの計算まで終了 d_transH にヘシアンの情報を格納
    //float *printHost;
    //printHost = (float*)malloc(szMatG);
    //CHECK_CUDA(cudaMemcpy(printHost, d_G, szMatG, cudaMemcpyDeviceToHost),"Failed to copy inverHess device ==> host");
    //printMatrix(sz_QC, sz_QC, printHost, sz_QC, "invHess");
    CHECK_CUDA(cudaFree(d_G), "Failed to free d_InverseG");
    //  -2*Hessian * b^T の b^Tベクトルを作成 (Hvector　←　b^T) 
    make_Vector_B<<<HORIZON, 1>>>(d_vectorB, d_ansRvect, sz_HessianElements);

    multiply_matrix<<<HORIZON, HORIZON>>>(d_Hess, 2.0f, d_transH);

    CHECK_CUSOLVER(cusolverDnSpotrf_bufferSize(cusolverH, uplo, sz_HESS, d_Hess, sz_HESS, &work_sizeForHess),"Failed to get work_sizeForHess !");
    CHECK_CUDA(cudaMalloc((void**)&work_spaceForHess, sizeof(float)*work_sizeForHess), "Failed to allocate work_spaceForHess !!!!");
    CHECK_CUSOLVER(cusolverDnSpotrf(cusolverH, uplo, sz_HESS, d_Hess, sz_HESS, work_spaceForHess, work_sizeForHess, devInfo), "Failed to Function # cusolverDnSpotrf #");

    SetUpIdentity_Matrix<<<sz_HESS ,sz_HESS>>>( d_transH );
    CHECK_CUSOLVER(cusolverDnSpotrs(cusolverH, uplo, sz_HESS, sz_HESS, d_Hess, sz_HESS, d_transH, sz_HESS, devInfo),"Failed to perform Inverse operation at # GetInputByLSMfittingMethod() #");
    multiply_matrix<<<sz_HESS, sz_HESS>>>(d_Hess, -1.0f, d_transH);
    CHECK_CUBLAS(cublasSgemv(handle_cublas, CUBLAS_OP_N, sz_HESS, sz_HESS, &alpha, d_Hess, sz_HESS, d_vectorB, 1, &beta, d_Input, 1), "Failed to calculate InputSeq by Proposed Method !!!");

    

    CHECK_CUDA(cudaFree(d_Hess),"Failed to free d_Hess");
    CHECK_CUDA(cudaFree(d_ansRvect),"Failed to free d_Hess");
    CHECK_CUDA(cudaFree(d_transH), "Failed to free d_transH");
    CHECK_CUDA(cudaFree(d_vectorB),"Failed to free d_vectorB");
    CHECK_CUDA(cudaFree(deviceHP),"Failed to free deviceHP");
    // CHECK_CUDA(cudaFree(work_sizeForHess),"Failed to free work_sizeForHess");
    CHECK_CUDA(cudaFree(devInfo), "Failed to free devInfo at GetInputByLSMfittingMethod");

    CHECK_CUDA(cudaMemcpy(OutPut, d_Input, sz_HESS * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy Result at # GetInputByLSMfittingMethod() #");

    CHECK_CUDA(cudaFree(d_Input) ,"Failed to free d_Input");
    CHECK_CUBLAS(cublasDestroy(handle_cublas), "Failed to destory cuBLAS");
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH),"Failed to destory cuSOLVER Handle_t");

    stop_t = clock();
    get_time = stop_t - start_t;
    printf("time == %f\n", get_time /CLOCKS_PER_SEC);
}

void GetInputByLSMfittingMethodFromcuBLAS(float *OutPut, HyperParaboloid *hostHP, int sz_QC, int sz_HESS, int sz_LSM)
{
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t handle_cublas = NULL;

    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    size_t szMatG = sz_QC * sz_QC * sizeof(float);
    size_t szMatHESS = sz_HESS * sz_HESS * sizeof(float);

    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH), "Failed to initialize cuSOLVER Handle_t");
    CHECK_CUBLAS(cublasCreate(&handle_cublas),"Failed to initialize cuBLAS");

    HyperParaboloid *deviceHP;
    CHECK_CUDA(cudaMalloc(&deviceHP, sz_LSM * sizeof(HyperParaboloid)), "Failed to allocate device HyperParaboloid data!");
    CHECK_CUDA(cudaMemcpy(deviceHP, hostHP, sz_LSM * sizeof(HyperParaboloid), cudaMemcpyHostToDevice), "Failed to copy device HyperParaboloid data!");

    float *d_G;
    float *d_InverseG;
    float *d_Rvect;
    float *d_ansRvect;
    float *d_Hess;
    float *d_transH;
    float *d_vectorB;
    float *d_Input;

    int prmszQHP = sz_QC + AdditionalSampleSize;
    int sz_HessianElements = sz_QC - (sz_HESS + 1);

    CHECK_CUDA(cudaMalloc(&d_G, szMatG), "Failed to allocate d_G !");
    CHECK_CUDA(cudaMalloc(&d_InverseG, szMatG), "Failed to allocate d_InverseG !");
    CHECK_CUDA(cudaMalloc(&d_Rvect, sz_QC * sizeof(float)),"Failed to allocate d_Rvect !");
    CHECK_CUDA(cudaMalloc(&d_ansRvect, sz_QC * sizeof(float)),"Failed to allocate d_Rvect !");
    CHECK_CUDA(cudaMalloc(&d_vectorB, sz_HESS * sizeof(float)),"Failed to allocate d_vectorB !");
    CHECK_CUDA(cudaMalloc(&d_Input, sz_HESS * sizeof(float)), "Failed to allocate d_Input !");
    CHECK_CUDA(cudaMalloc(&d_Hess, szMatHESS), "Failed to allocate d_Hess !!");
    CHECK_CUDA(cudaMalloc(&d_transH ,szMatHESS), "Failed to allocate d_transH !!")

    dim3 Block(2,2);
    dim3 Grid((sz_QC + Block.x -1) / Block.x, (sz_QC + Block.y -1)/ Block.y);

    int work_size/*, work_sizeForHess*/;
    float *work_space/*, *work_spaceForHess*/;
    int *devInfo;
    CHECK_CUDA(cudaMalloc((void**)&devInfo, sizeof(int)), "Failed to allocate devInfo at Func # GetInputByLSMfittingMethod #");
    
    float alpha = 1.0f;
    float beta = 0.0f;
    cudaEvent_t start_at_Inv, stop_at_Inv;
    float get_time;
    cudaEventCreate(&start_at_Inv);
    cudaEventCreate(&stop_at_Inv);
    cudaEventRecord(start_at_Inv, 0);

    if(sz_QC > LIMIT_OF_THREAD_PER_BLOCK)
    {
        getRegularMatrix_overThreadLimit<<<Grid, Block>>>( d_G, deviceHP, prmszQHP, sz_QC);
        // getRegularMatrix_overThreadLimit<<<Grid, Block>>>( d_G, deviceHP, prmszQHP, sz_QC);
    }else{
        getRegularMatrix<<<sz_QC ,sz_QC>>>( d_G, deviceHP, prmszQHP);
    }

    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after setting Matrix d_G !");
    getRegularVector<<<sz_QC, 1>>>( d_Rvect, deviceHP , prmszQHP);
    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after setting Vector d_Rvect !");

    CHECK_CUSOLVER(cusolverDnSpotrf_bufferSize(cusolverH, uplo, sz_QC, d_G, sz_QC, &work_size),"Failed to get work_size !");
    CHECK_CUDA(cudaMalloc((void**)&work_space, sizeof(float)*work_size), "Failed to allocate work_space at cuSOLVER");
    CHECK_CUSOLVER(cusolverDnSpotrf(cusolverH, uplo, sz_QC, d_G, sz_QC, work_space, work_size, devInfo), "Failed to Function # cusolverDnSpotrf #");

    if(sz_QC > LIMIT_OF_THREAD_PER_BLOCK)
    {
        SetUpIdentity_Matrix_overThreadLimit<<<Grid, Block>>>( d_InverseG, sz_QC);
    }else{
        SetUpIdentity_Matrix<<<sz_QC ,sz_QC>>>( d_InverseG );
    }
    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after setting Identity Matrix !");

    CHECK_CUSOLVER(cusolverDnSpotrs(cusolverH, uplo, sz_QC, sz_QC, d_G, sz_QC, d_InverseG, sz_QC, devInfo),"Failed to perform Inverse operation at cuSOLVER!");

    CHECK_CUBLAS(cublasSgemv(handle_cublas, CUBLAS_OP_N, sz_QC, sz_QC, &alpha, d_InverseG, sz_QC, d_Rvect, 1, &beta, d_ansRvect, 1), "Failed to product G^-1 * Rv by cuBLAS");

    //CHECK_CUDA(cudaFree(d_G), "Failed to free d_G");
    CHECK_CUDA(cudaFree(d_InverseG), "Failed to free d_InverseG");
    CHECK_CUDA(cudaFree(d_Rvect), "Failed to free d_Rvect");
    CHECK_CUDA(cudaFree(work_space), "Failed to free work_space at GetInputByLSMfittingMethod");

    get_FullHessian_Elements<<<sz_HESS, sz_HESS>>>( d_Hess, d_ansRvect);
    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after setting Upper Triangle Hessian Matrix !");
    transpose_opration_Matrix<<<sz_HESS, sz_HESS>>>(d_transH, d_Hess);
    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after Transposed Upper Triangle Hessian Matrix !");
    make_symmetric_Matrix<<<sz_HESS, sz_HESS>>>(d_transH, d_Hess);
    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after setting Symmetric Hessian Matrix !");
    // ヘシアンの計算まで終了 d_transH にヘシアンの情報を格納
    //float *printHost;
    //printHost = (float*)malloc(szMatG);
    //CHECK_CUDA(cudaMemcpy(printHost, d_G, szMatG, cudaMemcpyDeviceToHost),"Failed to copy inverHess device ==> host");
    //printMatrix(sz_QC, sz_QC, printHost, sz_QC, "invHess");
    CHECK_CUDA(cudaFree(d_G), "Failed to free d_InverseG");
    //  -2*Hessian * b^T の b^Tベクトルを作成 (Hvector　←　b^T) 
    make_Vector_B<<<HORIZON, 1>>>(d_vectorB, d_ansRvect, sz_HessianElements);

    multiply_matrix<<<HORIZON, HORIZON>>>(d_Hess, 2.0f, d_transH);

    float *h_Hess, *h_InvHess;
    h_Hess = (float *)malloc(szMatHESS);
    h_InvHess = (float *)malloc(szMatHESS);
    CHECK_CUDA(cudaMemcpy(h_Hess, d_Hess, szMatHESS, cudaMemcpyDeviceToHost), "Failed to copy device Hessian to host Hessian");
    GetInvMatrix(h_InvHess, h_Hess, sz_HESS);
    CHECK_CUDA(cudaMemcpy(d_transH, h_InvHess, szMatHESS, cudaMemcpyHostToDevice), "Failed to copy device Hessian to host Hessian");
    multiply_matrix<<<sz_HESS, sz_HESS>>>(d_Hess, -1.0f, d_transH);
    free(h_InvHess);
    free(h_Hess);
/*    CHECK_CUSOLVER(cusolverDnSpotrf_bufferSize(cusolverH, uplo, sz_HESS, d_Hess, sz_HESS, &work_sizeForHess),"Failed to get work_sizeForHess !");
    CHECK_CUDA(cudaMalloc((void**)&work_spaceForHess, sizeof(float)*work_sizeForHess), "Failed to allocate work_spaceForHess !!!!");
    CHECK_CUSOLVER(cusolverDnSpotrf(cusolverH, uplo, sz_HESS, d_Hess, sz_HESS, work_spaceForHess, work_sizeForHess, devInfo), "Failed to Function # cusolverDnSpotrf #");

    SetUpIdentity_Matrix<<<sz_HESS ,sz_HESS>>>( d_transH );
    CHECK_CUSOLVER(cusolverDnSpotrs(cusolverH, uplo, sz_HESS, sz_HESS, d_Hess, sz_HESS, d_transH, sz_HESS, devInfo),"Failed to perform Inverse operation at # GetInputByLSMfittingMethod() #");
    multiply_matrix<<<sz_HESS, sz_HESS>>>(d_Hess, -1.0f, d_transH);
*/
    CHECK_CUBLAS(cublasSgemv(handle_cublas, CUBLAS_OP_N, sz_HESS, sz_HESS, &alpha, d_Hess, sz_HESS, d_vectorB, 1, &beta, d_Input, 1), "Failed to calculate InputSeq by Proposed Method !!!");
    cudaEventRecord(stop_at_Inv, 0);
    cudaEventElapsedTime(&get_time, start_at_Inv, stop_at_Inv);
    printf("time == %f\n", get_time /1000);
    CHECK_CUDA(cudaFree(d_Hess),"Failed to free d_Hess");
    CHECK_CUDA(cudaFree(d_ansRvect),"Failed to free d_Hess");
    CHECK_CUDA(cudaFree(d_transH), "Failed to free d_transH");
    CHECK_CUDA(cudaFree(d_vectorB),"Failed to free d_vectorB");
    CHECK_CUDA(cudaFree(deviceHP),"Failed to free deviceHP");
    // CHECK_CUDA(cudaFree(work_sizeForHess),"Failed to free work_sizeForHess");
    CHECK_CUDA(cudaFree(devInfo), "Failed to free devInfo at GetInputByLSMfittingMethod");

    CHECK_CUDA(cudaMemcpy(OutPut, d_Input, sz_HESS * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy Result at # GetInputByLSMfittingMethod() #");

    CHECK_CUDA(cudaFree(d_Input) ,"Failed to free d_Input");
    CHECK_CUBLAS(cublasDestroy(handle_cublas), "Failed to destory cuBLAS");
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH),"Failed to destory cuSOLVER Handle_t");
}

__global__ void ParallelSimForPseudoGrad(SampleBasedHessian *Hess, MonteCarloMPC *sample, InputSequences *MCresult, Controller *CtrPrm, float delta, int *indices)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    // unsigned int id = iy * 3 + ix;
    unsigned int id =  iy * 3 * (HORIZON + 1) + ix;

    float stageCost = 0.0f;
    float totalCost = 0.0f;
    InputSequences *InputSeqInThread;
    InputSeqInThread = (InputSequences *)malloc(sizeof(InputSeqInThread) * HORIZON);
    float stateInThisThreads[DIM_OF_STATE] = { };
    float dstateInThisThreads[DIM_OF_STATE] = { };

    float d_param[NUM_OF_PARAMS];
    readParam(d_param, CtrPrm);

    for(int i = 0; i < DIM_OF_STATE; i++){
        stateInThisThreads[i] = CtrPrm->State[i];
    }

    for(int t = 0; t < HORIZON; t++){
        for(int uIndex = 0; uIndex < DIM_OF_U; uIndex++ )
        {
            if(blockIdx.x < 1 ){
                if( t == iy)
                {
                    if(threadIdx.x == 0)
                    {
                        InputSeqInThread[t].InputSeq[uIndex] = MCresult[t].InputSeq[uIndex];
                    }
                    if(threadIdx.x == 1)
                    {
                        InputSeqInThread[t].InputSeq[uIndex] = MCresult[t].InputSeq[uIndex] + delta;
                    }
                    if(threadIdx.x == 2)
                    {
                        InputSeqInThread[t].InputSeq[uIndex] = MCresult[t].InputSeq[uIndex] - delta;
                    }
                }else{
                    InputSeqInThread[t].InputSeq[uIndex] = MCresult[t].InputSeq[uIndex];
                }
            }else{
                if( t == iy)
                {
                    if(threadIdx.x == 0)
                    {
                        InputSeqInThread[t].InputSeq[uIndex] = sample[indices[blockIdx.x]].InputSeq[uIndex][t];
                    }
                    if(threadIdx.x == 1)
                    {
                        InputSeqInThread[t].InputSeq[uIndex] = sample[indices[blockIdx.x]].InputSeq[uIndex][t] + delta;
                    }
                    if(threadIdx.x == 2)
                    {
                        InputSeqInThread[t].InputSeq[uIndex] = sample[indices[blockIdx.x]].InputSeq[uIndex][t] - delta;
                    }
                }else{
                    InputSeqInThread[t].InputSeq[uIndex] = sample[indices[blockIdx.x]].InputSeq[uIndex][t];
                }

            }
        }
        for(int sec = 0; sec < 1; sec++){
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
            if(stateInThisThreads[0] <= CtrPrm.Constraints[2]){
                float collide[3] = { };
                collide[0] = d_param[0] * d_param[1] * cosf(stateInThisThreads[1]);
                collide[1] = d_param[2] + d_param[0] * powf(d_param[1],2);
                collide[2] = collide[0] / collide[1];
                stateInThisThreads[3] = stateInThisThreads[3] + (1 + d_param[7]) * collide[2] * stateInThisThreads[2]; //dtheta = dtheta + (1+e) * F * dx
                stateInThisThreads[2] = -d_param[7] * stateInThisThreads[2]; // dx = -e * dx
                stateInThisThreads[0] = CtrPrm.Constraints[2];
            }
            if(CtrPrm.Constraints[3] <= stateInThisThreads[0]){
                float collide[3] = { };
                collide[0] = d_param[0] * d_param[1] * cosf(stateInThisThreads[1]);
                collide[1] = d_param[2] + d_param[0] * powf(d_param[1],2);
                collide[2] = collide[0] / collide[1];
                stateInThisThreads[3] = stateInThisThreads[3] + (1 + d_param[7]) * collide[2] * stateInThisThreads[2]; //dtheta = dtheta + (1+e) * F * dx
                stateInThisThreads[2] = -d_param[7] * stateInThisThreads[2]; // dx = -e * dx
                stateInThisThreads[0] = CtrPrm.Constraints[3];
            }
#endif
        }
        while(stateInThisThreads[1] > M_PI)
            stateInThisThreads[1] -= (2 * M_PI);
        while(stateInThisThreads[1] < -M_PI)
            stateInThisThreads[1] += (2 * M_PI);
            
        stageCost = stateInThisThreads[0] * stateInThisThreads[0] * CtrPrm->WeightMatrix[0] + stateInThisThreads[1] * stateInThisThreads[1] * CtrPrm->WeightMatrix[1]
            + stateInThisThreads[2] * stateInThisThreads[2] * CtrPrm->WeightMatrix[2] + stateInThisThreads[3] * stateInThisThreads[3] * CtrPrm->WeightMatrix[3]
            + InputSeqInThread[t].InputSeq[0] * InputSeqInThread[t].InputSeq[0] * CtrPrm->WeightMatrix[4];

#ifndef COLLISION
        if(stateInThisThreads[0] <= 0){
            stageCost += 1 / (powf(stateInThisThreads[0] - CtrPrm->Constraints[2],2) * invBarrier);
            if(stateInThisThreads[0] < CtrPrm->Constraints[2]){
                stageCost += 1000000;
            }
        }else{
            stageCost += 1 / (powf(CtrPrm->Constraints[3] - stateInThisThreads[0],2) * invBarrier);
            if(stateInThisThreads[0] > CtrPrm->Constraints[3]){
                stageCost += 1000000;
            }
        }
#endif
        totalCost += stageCost;

        stageCost = 0.0f;
    }
    Hess[blockIdx.x].currentU[threadIdx.x][iy] = InputSeqInThread[iy].InputSeq[0];
    Hess[blockIdx.x].modified_U[threadIdx.x][iy] = 0.0f;
    Hess[blockIdx.x].delta_u[threadIdx.x][iy] = 0.0f;
    Hess[blockIdx.x].cost[iy][threadIdx.x] = totalCost;
    free(InputSeqInThread);
    //printf("id == %d blockIdx.x == %d   threadIdx.x == %d blockIdx.y == %d u == %f\n", id, blockIdx.x,  threadIdx.x, iy, InputSeqInThread[iy].InputSeq[0]);
    //__syncthreads();
}
