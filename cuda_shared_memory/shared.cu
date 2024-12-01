#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__
void errorexit(const char *s) {
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}

__global__ void computeSum(int *g_idata, int *g_odata) {
    extern __shared__ long shared_data[];
    unsigned int threadId = threadIdx.x;
    int globalThreadId = blockIdx.x * blockDim.x + threadId;

    // copy element from global to shared memory
    shared_data[threadId] = g_idata[globalThreadId];

    __syncthreads();

    // perform reduction in shared memory
    for (unsigned int i = 1; i < blockDim.x; i *= 2) {
        int index = 2 * i * threadId;
        if (index < blockDim.x) {
            shared_data[index] += shared_data[index + i];
        }
        __syncthreads();
    }

    if (threadId == 0) {
        g_odata[blockIdx.x] = shared_data[0];
    }
}

void generateRandomNumbers(int *arr, int N, int A, int B) {
	srand(time(NULL));
    for (int i = 0; i < N; i++) {
        arr[i] = A + rand() % (B - A + 1);
    }

}

int main(int argc,char **argv) {
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

	blocksingrid = ceil((double)N/threadsinblock);

	printf("The kernel will run with: %d blocks\n", blocksingrid);

	int *resultArrayHost, *resultArrayDevice, *randomNumbersDevice;

	resultArrayHost = (int *)calloc(N, sizeof(int));

	if (resultArrayHost == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	cudaMalloc((void **)&randomNumbersDevice, N * sizeof(int));
    cudaMalloc((void **)&resultArrayDevice, N * sizeof(int));

    cudaMemcpy(randomNumbersDevice, randomNumbers, N * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize device histogram to 0
    cudaMemset(resultArrayDevice, 0, N * sizeof(int));

    int sharedMemorySize = N * sizeof(int);

    computeSum<<<blocksingrid, threadsinblock, sharedMemorySize>>>(randomNumbersDevice, resultArrayDevice);

    // Copy the histogram result back to the host
    cudaMemcpy(resultArrayHost, resultArrayDevice, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);

    // Wait for the stop event to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print random numbers array and average result
    printf("Input array: \n");
    for (int i = 0; i < N; i ++) {
        printf("%d ", randomNumbers[i]);
    }
    printf("\n");
    printf("Average: %d", resultArrayHost[0] / N);

    printf("Kernel execution time: %.3f ms\n", milliseconds);
    
    // Free allocated memory
    free(randomNumbers);
    free(resultArrayHost);
    cudaFree(randomNumbersDevice);
    cudaFree(resultArrayDevice);

    return 0;

}
