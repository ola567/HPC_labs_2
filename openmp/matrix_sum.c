#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>

int** generateRandomMatrix(int rows, int columns) {
    int** matrix = (int**) malloc(rows * sizeof(int*));
    if (matrix == NULL) { 
        printf("Memory allocation failed.\n");
        exit(EXIT_FAILURE);  
    }

    srand(time(NULL)); 
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*) malloc(columns * sizeof(int));
        if (matrix[i] == NULL) { 
            printf("Memory allocation failed.\n");
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < columns; j++) {
            matrix[i][j] = rand() % 100;  
        }
    }
    return matrix;
}

int main() {
    int rows = 10000;
    int columns = 100000;

    int** matrix1 = generateRandomMatrix(rows, columns);
    int** matrix2 = generateRandomMatrix(rows, columns);

    double start = omp_get_wtime();
    int** result_matrix = (int**) malloc(rows * sizeof(int*));
    for(int i = 0; i < rows; i ++) {
        result_matrix[i] = (int*) malloc(columns * sizeof(int));
    }

    #pragma omp parallel for shared(result_matrix)
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            result_matrix[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
    double end = omp_get_wtime();
    double seconds = (double)(end - start);
    
    printf("Time: %f\n", seconds);


    // printf("Matrix1:\n");
    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < columns; j++) {
    //         printf("%d ", matrix1[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    
    // printf("Matrix2:\n");
    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < columns; j++) {
    //         printf("%d ", matrix2[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // printf("Result matrix:\n");
    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < columns; j++) {
    //         printf("%d ", result_matrix[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    return 0;
}
