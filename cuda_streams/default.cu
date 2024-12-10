#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__
void errorexit(const char *s) {
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}

__global__ void computeHistogram(int *randomNumbersDevice, unsigned long long *resultArrayDevice, int N, int A, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int resultIdx = (randomNumbersDevice[idx] - A);
        atomicAdd(&resultArrayDevice[resultIdx], 1);
    }
}

void generateRandomNumbers(int *arr, int N, int A, int B) {
	srand(time(NULL));
    for (int i = 0; i < N; i++) {
        arr[i] = A + rand() % (B - A);
    }

}

int main(int argc,char **argv) {
    int threadsinblock=1024;
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

	blocksingrid = ceil((double) N / threadsinblock);

	printf("The kernel will run with: %d blocks\n", blocksingrid);

	unsigned long long *resultArrayHost, *resultArrayDevice;
    int *randomNumbersDevice;

	resultArrayHost = (unsigned long long *)calloc((B-A), sizeof(unsigned long long));

	if (resultArrayHost == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	cudaMalloc((void **)&randomNumbersDevice, N * sizeof(int));
    cudaMalloc((void **)&resultArrayDevice, (B-A) * sizeof(unsigned long long));

    cudaMemcpy(randomNumbersDevice, randomNumbers, N * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize device histogram to 0
    cudaMemset(resultArrayDevice, 0, (B-A) * sizeof(unsigned long long));

    computeHistogram<<<blocksingrid, threadsinblock>>>(randomNumbersDevice, resultArrayDevice, N, A, B);

    // Copy the histogram result back to the host
    cudaMemcpy(resultArrayHost, resultArrayDevice, (B-A) * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);

    // Wait for the stop event to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the histogram
    // for (int i = 0; i < N; i++) {
    //     printf("%d, ", randomNumbers[i]);
    // }
    // printf("\n");
    
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