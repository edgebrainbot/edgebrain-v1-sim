#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cvxopt.h" 
#include "misc.h"
#include <time.h>

extern spmatrix create_test_spmatrix();
extern void base_gemv( void *A, matrix *x, matrix *y, char trans, void *alpha, void *beta,
                    int n, int m, int incx, int incy, int offsetx, int offsety, int issp);

void test_sp_gemv() {
    clock_t start, end;
    double cpu_time_used;

    spmatrix A = create_test_spmatrix();

    // 向量 x = [1, 1]
    double xdata[4] = {1.0, 2.0, 3.0, 4.0};
    matrix x = {
        .mat_type = MAT_DENSE,
        .nrows = 4,
        .ncols = 1,
        .id = DOUBLE,
        .buffer = xdata
    };

    double ydata[5] = {1.0, 3.0, 0.0, 2.0, 5.0};
    matrix y = {
        .mat_type = MAT_DENSE,
        .nrows = 5,
        .ncols = 1,
        .id = DOUBLE,
        .buffer = ydata
    };

    double alpha = 1.0;
    double beta = 1.0;

    start = clock();
    base_gemv(
        (void*)&A, &x, &y, 'N',
        &alpha, &beta,
        -1, -1, 1, 1,
        0, 0, 0  // issp = 1
    );
    end = clock();

    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time used: %.6f seconds\n", cpu_time_used);


    printf("Result y = A * x:\n");
    for (int i = 0; i < 5; ++i) 
        printf("y[%d] = %f\n", i, ydata[i]);


    // 清理
    free(A.obj->values);
    free(A.obj->rowind);
    free(A.obj->colptr);
    free(A.obj);
}