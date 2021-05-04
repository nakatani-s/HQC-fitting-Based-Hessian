/*
    Functions for matrix operations
*/
#include "../include/Matrix.cuh"

unsigned int countBlocks(unsigned int a, unsigned int b) {
	unsigned int num;
	num = a / b;
	if (a < b || a % b > 0)
		num++;
	return num;
}


void printMatrix(int m, int n, float*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            float Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
            //printf("%s[%d] = %f\n", name, row + col*lda, Areg);
        }
    }
}

void shift_Input_vec( InputSequences *inputVector, int uIndex)
{
    float temp[HORIZON]= { };
    for(int i = 0; i < HORIZON - 1; i++){
        temp[i] = inputVector[i+1].InputSeq[uIndex];
    }
    temp[HORIZON - 1] = inputVector[HORIZON - 1].InputSeq[uIndex];
    for(int i = 0; i < HORIZON; i++){
        inputVector[i].InputSeq[uIndex] = temp[i];
    }
}

__global__ void SetUpIdentity_Matrix(float *IdMat)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if(threadIdx.x == blockIdx.x)
    {
        IdMat[id] = 1.0f;
        //values[threadIdx.x] = 1.0f;
    }else{
        IdMat[id] = 0.0f;
        //values[threadIdx.x] = 0.0f;
    }
    __syncthreads();
}

__global__ void SetUpIdentity_Matrix_overThreadLimit(float *IdMat, int Ydimention)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int id = iy * Ydimention + ix;
    if(ix == iy)
    {
        IdMat[id] = 1.0f;
        //values[threadIdx.x] = 1.0f;
    }else{
        IdMat[id] = 0.0f;
        //values[threadIdx.x] = 0.0f;
    }
    __syncthreads();
}



void GetInvMatrix(float *invMat, float *originMat, int num)
{
    cublasHandle_t cublas_status;
    CHECK_CUBLAS(cublasCreate(&cublas_status),"Failed to initialize cuBLAS");

    float **arrayA;
    float **arrayC;
    float *d_arrayA;
    float *d_arrayC;
    int *d_LUPivots;
    int *d_LUInfo;

    size_t szMat = num * num * sizeof(float);

    CHECK_CUDA(cudaMalloc(&arrayA, sizeof(float*)), "Failed to allocate arrayA");
    CHECK_CUDA(cudaMalloc(&arrayC, sizeof(float*)), "Failed to allocate arrayC");
    CHECK_CUDA(cudaMalloc(&d_arrayA, szMat), "Failed to allocate d_arrayA");
    CHECK_CUDA(cudaMalloc(&d_arrayC, szMat), "Failed to allocate d_arrayC");
    CHECK_CUDA(cudaMalloc(&d_LUPivots, sizeof(int)), "Failed to allocate arrayC");
    CHECK_CUDA(cudaMalloc(&d_LUInfo, sizeof(int)), "Failed to allocate arrayC");

    CHECK_CUDA(cudaMemcpy(d_arrayA, originMat, szMat, cudaMemcpyHostToDevice), "Failed to copy Origin Matrix to d_arrayA");
    CHECK_CUDA(cudaMemcpy(arrayA, &d_arrayA, sizeof(float*), cudaMemcpyHostToDevice), "Failed to copy to arrayA");
    CHECK_CUDA(cudaMemcpy(arrayC, &d_arrayC, sizeof(float*), cudaMemcpyHostToDevice), "Failed to copy to arrayC");

    CHECK_CUBLAS(cublasSgetrfBatched(cublas_status, num, arrayA, num, d_LUPivots, d_LUInfo, 1), "Failed to perform LU decomp operation");
    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after kernel call!");

    CHECK_CUBLAS(cublasSgetriBatched(cublas_status, num, (const float **)arrayA, num, d_LUPivots, arrayC, num, d_LUInfo, 1), "Failed to perform Inverse operation!");
    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after kernel call!");

    CHECK_CUDA(cudaMemcpy(invMat, d_arrayC, szMat, cudaMemcpyDeviceToHost), "Failed to copy to invMat");

    CHECK_CUDA(cudaFree(arrayA),"Failed to free arrayA");
    CHECK_CUDA(cudaFree(arrayC),"Failed to free arrayC");
    CHECK_CUDA(cudaFree(d_arrayA),"Failed to free d_arrayA");
    CHECK_CUDA(cudaFree(d_arrayC),"Failed to free d_arrayC");
    CHECK_CUDA(cudaFree(d_LUPivots),"Failed to free d_LUPivots");
    CHECK_CUDA(cudaFree(d_LUInfo),"Failed to free d_LUInfo");

    CHECK_CUBLAS(cublasDestroy(cublas_status), "Failed to destory cuBLAS");

}

void GetInvMatrixBycuSOLVER(float *invMat, float *Mat, int sz_dim)
{
    cusolverDnHandle_t cusolverH = NULL;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH), "Failed to initialize cuSOLVER Handle_t");
    // cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; 
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    size_t szMat = sz_dim * sz_dim * sizeof(float);
    float *d_arrayA;
    float *d_arrayInv;

    CHECK_CUDA(cudaMalloc(&d_arrayA, szMat), "Failed to allocate arrayA at cuSOLVER");
    CHECK_CUDA(cudaMalloc(&d_arrayInv, szMat), "Failed to allocate arrayInv at cuSOLVER");
    
    CHECK_CUDA(cudaMemcpy(d_arrayA, Mat, szMat, cudaMemcpyHostToDevice), "Failed to copy Matrix data to d_arrayA");

    int work_size;
    float *work_space;
    int *devInfo;
    CHECK_CUDA(cudaMalloc((void**)&devInfo, sizeof(int)), "Failed to allocate devInfo at Func # GetInvMatrixBycuSOLVER #");

    dim3 cuSolverBlock(2,2);
    dim3 cuSolverGrid((sz_dim + cuSolverBlock.x -1)/ cuSolverBlock.x, (sz_dim + cuSolverBlock.y -1) / cuSolverBlock.y);

    CHECK_CUSOLVER(cusolverDnSpotrf_bufferSize(cusolverH, uplo, sz_dim, d_arrayA, sz_dim, &work_size),"Failed to get work_size at cuSOLVER");
    CHECK_CUDA(cudaMalloc((void**)&work_space, sizeof(float)*work_size), "Failed to allocate work_space at cuSOLVER");
    CHECK_CUSOLVER(cusolverDnSpotrf(cusolverH, uplo, sz_dim, d_arrayA, sz_dim, work_space, work_size, devInfo), "Failed to Function # cusolverDnSpotrf #");

    if(sz_dim > 1024){
        SetUpIdentity_Matrix_overThreadLimit<<<cuSolverGrid, cuSolverBlock>>>(d_arrayInv, sz_dim); 
    }else{
        SetUpIdentity_Matrix<<<sz_dim, sz_dim>>>( d_arrayInv );
    }
    CHECK_CUDA(cudaDeviceSynchronize(), "Failed to synchronize after kernel call at cuSOLVER!");

    CHECK_CUSOLVER(cusolverDnSpotrs(cusolverH, uplo, sz_dim, sz_dim, d_arrayA, sz_dim, d_arrayInv, sz_dim, devInfo),"Failed to perform Inverse operation at cuSOLVER!");

    CHECK_CUDA(cudaMemcpy(invMat, d_arrayInv, szMat, cudaMemcpyDeviceToHost),"Failed to copy Result at cuSOLVER");

    CHECK_CUDA(cudaFree(d_arrayA),"Failed to free arrayA at cuSOLVER");
    CHECK_CUDA(cudaFree(d_arrayInv),"Failed to free arrayI at cuSOLVER");
    CHECK_CUDA(cudaFree(work_space),"Failed to free work_space at cuSOLVER");
    CHECK_CUDA(cudaFree(devInfo),"Failed to free devInfo at cuSOLVER");

    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH),"Failed to destory cuSOLVER Handle_t");
}



__global__ void GetResultMatrixProduct( float *ans, float *lmat, float *rmat, const int nx )
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int id = iy * nx  + ix;

    if( ix < HORIZON && iy < HORIZON)
    {
        float el = 0.0f;
        for(int i = 0; i < HORIZON; i++)
        {
            el += lmat[ iy * nx + i] * rmat[ i * nx + ix ];
        }
        ans[id] = el;
    }
    __syncthreads();
}




__global__ void multiply_matrix(float *OutMatrix, float voc, float *InMatrix)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    OutMatrix[id] = voc * InMatrix[id];
    //printf("OutMatrix[%d] == %f = %f * %f\n",id, OutMatrix[id], voc, InMatrix[id]);
    __syncthreads();
}

__global__ void make_Vector_B(float *OutVector, float *Elemets, int indecies)
{
    unsigned int id =threadIdx.x + blockIdx.x * blockDim.x;
    OutVector[id] = Elemets[indecies + id];
    //printf("id = %d IN = %f\n", indecies + id, Elemets[indecies + id]);
    __syncthreads();
}

__global__ void make_symmetric_Matrix(float *Out, float *In)
{
    unsigned int id =threadIdx.x + blockIdx.x * blockDim.x;
	if( blockIdx.x > threadIdx.x)
    {
		if(!(Out[id]==In[id]))
		{
			Out[id] = In[id];
		}
	}
}

__global__ void transpose_opration_Matrix(float *Out, float *In)
{
    unsigned int id =threadIdx.x + blockIdx.x * blockDim.x;
    int In_index = blockIdx.x + threadIdx.x * blockDim.x;
    Out[id] = In[In_index];
    __syncthreads();
}

__global__ void get_FullHessian_Elements(float *outElements, float *inElements)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    float temp_here;
    /*if(blockIdx.x == 0){
        //outElements[id] = inElements[threadIdx.x];
        temp_here = inElements[threadIdx.x];
    }
    if(threadIdx.x==0){
        //outElements[id] = inElements[blockIdx.x];
        temp_here = inElements[blockIdx.x];
    }
    if(threadIdx.x * blockIdx.x != 0){
        // int i_id;
        // i_id = blockIdx.x + (HORIZON - 1) + (threadIdx.x - 1);
        //outElements[id] = inElements[blockIdx.x + (HORIZON - 1) + (threadIdx.x - 1)];
        
        temp_here = inElements[blockIdx.x + (HORIZON - 1) + (threadIdx.x - 1)];
    }*/
    int vect_id = blockIdx.x;
    if(threadIdx.x <= blockIdx.x){
		for(int t_id = 0; t_id < threadIdx.x; t_id++){
            int sum_a = t_id + 1;
			vect_id += (HORIZON - sum_a);
		}
        //outElements[id] = inElements[vect_id];
        temp_here = inElements[vect_id];
    }else{
        //outElements[id] = 0.0f;
        temp_here = 0.0f;
    }
    if(threadIdx.x != blockIdx.x){
        //outElements[id] = outElements[id] / 2;
        outElements[id] = temp_here / 2;
    }else{
        outElements[id] = temp_here;
    }
    //printf("outElements[%d] == %f\n", id, outElements[id]);
    __syncthreads();
}



__global__ void getRegularMatrix(float *outRmatrix, HyperParaboloid *elements, int sumSet)
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

__global__ void getRegularMatrix_overThreadLimit(float *outRmatrix, HyperParaboloid *elements, int sumSet, int Ydimention)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int id = iy * Ydimention + ix;
    outRmatrix[id] = 0.0f;
    for(int index = 0; index < sumSet; index++){
        outRmatrix[id] += elements[index].tensor_vector[ix] * elements[index].tensor_vector[iy];
    }
    __syncthreads();
}

__global__ void make_tensor_vector(HyperParaboloid *output, MonteCarloMPC *input, int *indices)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    int next_indices = 0;
    for(int i = 0; i < HORIZON; i++){
        for(int j = i; j < HORIZON; j++){
            // output[id].tensor_vector[next_indices] = input[indices[id]].Input[i] * input[indices[id]].Input[j] * input[indices[id]].WHM;
            output[id].tensor_vector[next_indices] = input[indices[id]].InputSeq[0][i] * input[indices[id]].InputSeq[0][j] * input[indices[id]].WHM;
	        //output[id].tensor_vector[next_indices] = i * j;
            // output[id].column_vector[next_indices] = input[indices[id]].L * input[indices[id]].Input[i] * input[indices[id]].Input[j]  * input[indices[id]].WHM;
            output[id].column_vector[next_indices] = input[indices[id]].L * input[indices[id]].InputSeq[0][i] * input[indices[id]].InputSeq[0][j]  * input[indices[id]].WHM;
            //output[id].column_vector[next_indices] = 3.0;
            next_indices += 1;
        }
    }
    for(int i = 0; i < HORIZON; i++){
        // output[id].tensor_vector[next_indices] = input[indices[id]].Input[i]  * input[indices[id]].WHM;
        output[id].tensor_vector[next_indices] = input[indices[id]].InputSeq[0][i]  * input[indices[id]].WHM;
        //output[id].tensor_vector[next_indices] = i;

        // output[id].column_vector[next_indices] = input[indices[id]].L * input[indices[id]].Input[i]  * input[indices[id]].WHM;
        output[id].column_vector[next_indices] = input[indices[id]].L * input[indices[id]].InputSeq[0][i]  * input[indices[id]].WHM;
        //output[id].column_vector[next_indices] = (1/2)*i;
        next_indices += 1;
    }
    output[id].tensor_vector[SIZE_OF_PARABOLOIDVESTOR - 1] = 1.0f  * input[indices[id]].WHM;
    output[id].column_vector[SIZE_OF_PARABOLOIDVESTOR - 1] = input[indices[id]].L  * input[indices[id]].WHM;
    __syncthreads();
}