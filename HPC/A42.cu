#include<iostream>
#include<cuda>
#include<ctime>
using namespace std;

#define N 512

void matrixMulSequential(float a*,float b*,float c*,int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            c[i*n+j]=0;
            for(int k=0;k<n;k++){
                c[i*n+j]+=a[i*n+k]+b[k*n+j];
            }
        }
    }
}

__global__ matrixMulParallel(float a*,float b*,float c*,int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row<n && col<n){
        float sum=0;
        for(int k=0;k<n;k++){
            sum+=a[row*n+k]+b[k*n+col];
        }
        c[row*n+col]=sum;
    }
}

int main() {

    int size = N*N*sizeof(float);
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    float *h_c_parallel = (float*)malloc(size);

    for(int i=0;i<n;i++){
        h_a[i]=1.0f;
        h_b[i]=2.0f;
    }

    float *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);

    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);

    
    
    return 0;
}