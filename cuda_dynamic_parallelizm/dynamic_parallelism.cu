#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DEBUG 0
#define MAX_RECURSION_DEPTH 5

__host__ void errorexit(const char *s)
{
    printf("\n%s\n", s);
    exit(EXIT_FAILURE);
}

__host__ void generateRandomNumbers(int *arr, int N, int A, int B)
{
    srand(time(NULL));
    for (int i = 0; i < N; i++)
    {
        arr[i] = A + rand() % (B - A);
    }
}

__device__ int partition(int *arr, int left, int right)
{
    int pivot = arr[right];
    int i = left - 1;

    for (int j = left; j < right; j++)
    {
        if (arr[j] <= pivot)
        {
            i++;
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    int temp = arr[i + 1];
    arr[i + 1] = arr[right];
    arr[right] = temp;

    return i + 1;
}

__global__ void quicksort(int *arr, int left, int right, int depth)
{
    if (left >= right || depth >= MAX_RECURSION_DEPTH)
    {
        return;
    }

    // Partition the array
    int pivotIndex = partition(arr, left, right);

    // Launch child kernels if depth allows
    if (depth + 1 < MAX_RECURSION_DEPTH)
    {
        quicksort<<<1, 1>>>(arr, left, pivotIndex - 1, depth + 1);
        quicksort<<<1, 1>>>(arr, pivotIndex + 1, right, depth + 1);
        cudaDeviceSynchronize();
    }
    else
    {
        // If maximum depth reached, sort sequentially
        for (int i = left + 1; i <= right; i++)
        {
            int key = arr[i];
            int j = i - 1;
            while (j >= left && arr[j] > key)
            {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }
}

int main()
{
    int N;
    int A = 1;
    int B = 100000000;

    cudaEvent_t start, stop;
    float milliseconds = 0;

    printf("Enter number of elements: \n");
    scanf("%d", &N);

    int *randomNumbers = (int *)malloc(N * sizeof(int));
    if (randomNumbers == NULL)
    {
        printf("Memory allocation failed.\n");
        return 1;
    }
    generateRandomNumbers(randomNumbers, N, A, B);

    // printf("Array: \n");
    // for (int i = 0; i < N; i++) {
    //     printf("%d, ", randomNumbers[i]);
    // }
    // printf("\n");

    int *randomNumbersDevice;

    if (cudaSuccess != cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_RECURSION_DEPTH))
    {
        errorexit("Error setting depth limit");
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc((void **)&randomNumbersDevice, N * sizeof(int));
    cudaMemcpy(randomNumbersDevice, randomNumbers, N * sizeof(int), cudaMemcpyHostToDevice);

    quicksort<<<1, 1>>>(randomNumbersDevice, 0, N - 1, 0);

    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(randomNumbers, randomNumbersDevice, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // printf("Sorted array:\n");
    // for (int i = 0; i < N; i++) {
    //     printf("%d \n", randomNumbers[i]);
    // }
    // printf("\n");

    printf("Kernel execution time: %.3f ms\n", milliseconds);

    free(randomNumbers);
    cudaFree(randomNumbersDevice);

    return 0;
}
