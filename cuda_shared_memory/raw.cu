#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

__host__ void errorexit(const char *s)
{
    printf("\n%s\n", s);
    exit(EXIT_FAILURE);
}

__global__ void computeSum(int *randomNumbers, unsigned long long *resultSumDevice, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        atomicAdd(resultSumDevice, (unsigned long long)randomNumbers[idx]);
    }
}

void generateRandomNumbers(int *arr, int N, int A, int B)
{
    srand(time(NULL));
    for (int i = 0; i < N; i++)
    {
        arr[i] = A + rand() % (B - A + 1);
    }
}

int main(int argc, char **argv)
{
    int threadsinblock = 1024;
    int blocksingrid;

    int N, A, B;
    float milliseconds = 0;

    // Get user input
    printf("Enter number of elements: \n");
    scanf("%d", &N);
    printf("Enter A value (start range): \n");
    scanf("%d", &A);
    printf("Enter B value (end range): \n");
    scanf("%d", &B);

    // Allocate host memory for random numbers
    int *randomNumbers = (int *)malloc(N * sizeof(int));
    if (randomNumbers == NULL)
    {
        printf("Memory allocation failed.\n");
        return 1;
    }

    // Generate random numbers
    generateRandomNumbers(randomNumbers, N, A, B);
    blocksingrid = (N + threadsinblock - 1) / threadsinblock; // Calculate number of blocks
    printf("The kernel will run with: %d blocks\n", blocksingrid);

    // Device memory pointers
    unsigned long long *resultSumDevice;
    int *randomNumbersDevice;
    unsigned long long resultSum = 0;

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Allocate device memory
    if (cudaMalloc((void **)&randomNumbersDevice, N * sizeof(int)) != cudaSuccess)
    {
        errorexit("Failed to allocate device memory for randomNumbers");
    }
    if (cudaMalloc((void **)&resultSumDevice, sizeof(unsigned long long)) != cudaSuccess)
    {
        errorexit("Failed to allocate device memory for resultSumDevice");
    }

    // Copy data to device
    cudaMemcpy(randomNumbersDevice, randomNumbers, N * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize device result sum to 0
    cudaMemset(resultSumDevice, 0, sizeof(unsigned long long));

    // Launch kernel
    computeSum<<<blocksingrid, threadsinblock>>>(randomNumbersDevice, resultSumDevice, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        errorexit(cudaGetErrorString(err));
    }

    // Copy result back to host
    cudaMemcpy(&resultSum, resultSumDevice, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate and print results
    // for (int i = 0; i < N; i++){
    //     printf("%d, ", randomNumbers[i]);
    // }
    // printf("\n");
    double average = (double)resultSum / N;
    printf("Average: %.2f\n", average);

    // Print execution time
    printf("Kernel execution time: %.3f ms\n", milliseconds);

    // Free allocated memory
    free(randomNumbers);
    cudaFree(randomNumbersDevice);
    cudaFree(resultSumDevice);

    return 0;
}
