#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__
void errorexit(const char *s) {
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}

//elements generation
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
    int threadsinblock = 1024;
    int blocksingrid = 10000;	
    unsigned long long *result;
    unsigned long long number;

    printf("Enter number: \n");
    scanf("%llu", &number);

    if (number < 2) {
      printf("Not prime \n");
      return 0;
    }

    //unified memory allocation - available for host and device
    if (cudaSuccess!=cudaMallocManaged(&result, sizeof(unsigned long long)))
      errorexit("Error allocating memory on the GPU \n");

    //call to GPU - kernel execution 
    *result = 0;
    
    if_prime<<<blocksingrid,threadsinblock>>>(number, result);

    if (cudaSuccess!=cudaGetLastError())
      errorexit("Error during kernel launch");
  
    //device synchronization to ensure that data in memory is ready
    cudaDeviceSynchronize();

    if (*result > 0) {
      printf("Not prime \n");
    }
    else {
      printf("Prime \n");
    }

}
