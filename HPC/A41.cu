#include<iostream>
#include<cuda.h>
#include<ctime>
using namespace std;

void vectorAddSequential(float *a,float *b,float *c,int n){
    for(int i=0;i<n;i++){
        c[i]=a[i]+b[i];
    }
}

__global__ void vectorAddParallel(float *a,float *b,float *c,int n){
    int idx = blockDim.x + blockIdx.x + threadIdx.x;
    if(idx < n){
        c[idx]=a[idx]+b[idx];
    }
}

int main(){
    int n=1<<20;
    size_t size=n*sizeof(float);

    float *h_a = (float)malloc(size);
    float *h_b = (float)malloc(size);
    float *h_c = (float)malloc(size);
    float *h_c_parallel = (float)malloc(size);

    for(int i=0;i<n;i++){
        h_a[i]=1.0f;
        h_b[i]=2.0f;
    }

    float *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);

    cudaMemcpy(d_a,h_a,size,cudeMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size,cudeMemcpyHostToDevice);

    clock_t start = clock();
    vectorAddSequential(h_a,h_b,h_c,n);
    clock_t end = clock();
    double seq_time = double(end-start)/CLOCKS_PER_SEC;
    cout<<"\nSequential time: "<<seq_time;

    int threads = 256;
    int blocks = (n-threads+1)/treads;

    start = clock();
    vectorAddParallel<<<blocks,threads>>>(d_a,d_b,d_c,n);
    cudaDeviceSynchronize();
    end = clock();
    double par_time = double(end-start)/CLOCKS_PER_SEC;
    cout<<"\nParallel time: "<<par_time;

    cudaMemcpy(h_c_parallel,d_c,size,cudeMemcpyDeviceToHost);

    cout<<"Result from sequential: "<<h_c[0]<<end;
    cout<<"Result from Parallel: "<<h_c_parallel[0]<<endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_parallel);

    return 0;
}