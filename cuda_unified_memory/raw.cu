#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__
void errorexit(const char *s) {
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}


__global__ 
void if_prime(unsigned long long number, unsigned long long *dresult) {
    int my_index=blockIdx.x*blockDim.x+threadIdx.x;
    if (my_index >= 2 && my_index * my_index <= number) {
      if (number % my_index == 0) {
        atomicAdd(dresult, 1);
      }
    }
}


int main(int argc,char **argv) {
    int threadsinblock=1024;
    int blocksingrid=10000;	
    unsigned long long number;
    unsigned long long hresult;
    unsigned long long *dresult;

    printf("Enter number: \n");
    scanf("%llu", &number);

    if (number < 2) {
      printf("Not prime \n");
      return 0;
    }

    // device memory allocate
    if (cudaSuccess != cudaMalloc((void **)&dresult, sizeof(unsigned long long)))
      errorexit("Error allocationg memory on the GPU \n");

    //call to GPU - kernel execution 
    if_prime<<<blocksingrid,threadsinblock>>>(number, dresult);
  
    //getting results from GPU to host memory
    if (cudaSuccess!=cudaMemcpy(&hresult, dresult, sizeof(unsigned long long),cudaMemcpyDeviceToHost))
       errorexit("Error copying results \n");

    if (hresult > 0) {
      printf("Not prime \n");
    }
    else {
      printf("Prime \n");
    }
}
