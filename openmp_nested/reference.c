#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>

#define MAX_RECURSION_DEPTH 5

void generateRandomNumbers(int *arr, int N, int A, int B)
{
    srand(time(NULL));
    for (int i = 0; i < N; i++)
    {
        arr[i] = A + rand() % (B - A);
    }
}

int partition(int *arr, int left, int right)
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

void quicksort(int *arr, int left, int right, int depth)
{ 
    if (left >= right || depth >= MAX_RECURSION_DEPTH)
    {
        return;
    }

    // Partition the array
    int pivotIndex = partition(arr, left, right);

    if (depth + 1 < MAX_RECURSION_DEPTH)
    {
      quicksort(arr, left, pivotIndex - 1, depth + 1);
      quicksort(arr, pivotIndex + 1, right, depth + 1);
        
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


int main(int argc,char **argv) {
  int N;
  int A = 1;
  int B = 100;

  printf("Enter number of elements: \n");
  scanf("%d", &N);

  int *randomNumbers = (int *)malloc(N * sizeof(int));
  if (randomNumbers == NULL)
  {
      printf("Memory allocation failed.\n");
      return 1;
  }
  generateRandomNumbers(randomNumbers, N, A, B);
  
//   printf("Matrix: \n");
//   for(int i = 0; i < N; i++) {
//     printf("%d ", randomNumbers[i]);
//   }
//   printf("\n");
  
  clock_t start = clock();
  quicksort(randomNumbers, 0, N - 1, 0);
  clock_t end = clock();
  float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    
  printf("Time: %f\n", seconds);

//   printf("Result: \n");
//   for(int i = 0; i < N; i++) {
//     printf("%d ", randomNumbers[i]);
//   }
//   printf("\n");
  free(randomNumbers);
}
