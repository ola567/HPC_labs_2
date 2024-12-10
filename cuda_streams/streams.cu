#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


#define DEBUG 0
__host__
void errorexit(const char *s) {
    printf("\n%s\n",s); 
    exit(EXIT_FAILURE);   
}

__host__ 
void generateRandomNumbers(int *arr, int N, int A, int B) {
	srand(time(NULL));
    for (int i = 0; i < N; i++) {
        arr[i] = A + rand() % (B - A);
    }

}

__global__ 
void computeHistogram(int *randomNumbersDevice, unsigned long long *resultArrayDevice, int N, int A, int streamChunk, int streamId) {
    int my_index = blockIdx.x * blockDim.x + threadIdx.x + streamId * streamChunk;

    if(my_index < N) {
      int resultIdx = (randomNumbersDevice[my_index] - A);
      atomicAdd(&resultArrayDevice[resultIdx], 1);
    } 
}

int main(int argc, char **argv) {
  int threadsinblock = 1024;
  int blocksingrid;
  int N, A, B;
    
  cudaEvent_t start, stop;
  float milliseconds = 0;
    
  printf("Enter number of elements: \n");
  scanf("%d", &N);

  printf("Enter A value (start range): \n");
  scanf("%d", &A);

  printf("Enter B value (end range): \n");
  scanf("%d", &B);

  int *randomNumbers = (int *)malloc(N * sizeof(int));
  if (randomNumbers == NULL) {
    printf("Memory allocation failed.\n");
    return 1;
  }
  
  generateRandomNumbers(randomNumbers, N, A, B);

  unsigned long long *resultArrayHost, *resultArrayDevice;
  int *randomNumbersDevice;
  resultArrayHost = (unsigned long long *)calloc((B-A), sizeof(unsigned long long));
  if (resultArrayHost == NULL) {
    printf("Memory allocation failed.\n");
    return 1;
  }

  //Streams
  int numberOfStreams = 2;
  int streamChunk = N / numberOfStreams;
  printf("Stream chunk is %d \n", streamChunk);
  
  blocksingrid = 1 + ((streamChunk - 1) / threadsinblock); 
  printf("blocksingrid is %d \n", blocksingrid);

  //Create streams
  cudaStream_t streams[numberOfStreams];
  for(int i = 0; i < numberOfStreams; i++) {
    if (cudaSuccess!=cudaStreamCreate(&streams[i]))
      errorexit("Error creating stream");
  }

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  //Allocate memory on host
  cudaMalloc((void **)&randomNumbersDevice, N * sizeof(int));
  cudaMalloc((void **)&resultArrayDevice, (B-A) * sizeof(unsigned long long));

  // Initialize device histogram to 0
  cudaMemset(resultArrayDevice, 0, (B-A) * sizeof(unsigned long long));

  //execute operation in each stream - copy chunk of data and run calculations
  for(int i=0; i < numberOfStreams; i++) {
    cudaMemcpyAsync(&randomNumbersDevice[streamChunk * i], &randomNumbers[streamChunk * i], streamChunk * sizeof(int), cudaMemcpyHostToDevice, streams[i]);      
    computeHistogram<<<blocksingrid, threadsinblock, 0, streams[i]>>>(randomNumbersDevice, resultArrayDevice, N, A, streamChunk, i);
  }
  cudaDeviceSynchronize();
  
  // Copy the histogram result back to the host
  cudaMemcpy(resultArrayHost, resultArrayDevice, (B-A) * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop, 0);
  // Wait for the stop event to finish
  cudaEventSynchronize(stop);
  // Calculate elapsed time
  cudaEventElapsedTime(&milliseconds, start, stop);

  // Print the histogram
//   for (int i = 0; i < N; i ++){
//     printf("%d, ", randomNumbers[i]);
//   }
//   printf("\n");

  printf("Histogram:\n");
  for (int i = 0; i < B-A; i++) {
    printf("%d occures %llu\n", i + A, resultArrayHost[i]);
  }

  // Print execution time
  printf("Kernel execution time: %.3f ms\n", milliseconds);

  // Free allocated memory
  free(randomNumbers);
  free(resultArrayHost);
  cudaFree(randomNumbersDevice);
  cudaFree(resultArrayDevice);

  return 0;
}
