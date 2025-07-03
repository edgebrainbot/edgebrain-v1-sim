#define _POSIX_C_SOURCE 199309L
#include "cvxopt.h"
#include "misc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern scaling misc_compute_scaling(matrix *s, matrix *z, matrix *lmbda, DIMs *dims, int *mnl_ptr);
extern double misc_jdot(matrix* x, matrix* y, int n, int offsetx, int offsety);
extern matrix* load_txt_as_matrix(const char* filename);

void test_misc_jdot() {
        int n = 3;
    matrix* x = Matrix_New(n, 1, DOUBLE);
    matrix* y = Matrix_New(n, 1, DOUBLE);

    // x = [5, 2, 3], y = [5, 2, 3]
    ((double*)x->buffer)[0] = 5.0;
    ((double*)x->buffer)[1] = 2.0;
    ((double*)x->buffer)[2] = 3.0;

    ((double*)y->buffer)[0] = 5.0;
    ((double*)y->buffer)[1] = 2.0;
    ((double*)y->buffer)[2] = 1.0;

    double res = misc_jdot(x, y, n, 0, 0);
    printf("jdot result = %f\n", res);  // 5*5 - (2*2 + 3*3) = 25 - 13 = 12

    free(x->buffer); free(y->buffer);
    free(x); free(y);
}

void test_computing_scale() {
    int sbuf[] = { 10, 6, 20, 16, 10, 3, 6, 6 };

    DIMs dims = {.l = 0, .q_size = 0, .s_size = 8, .q = NULL, .s = sbuf};     
    int mnl = 0;              // 起始索引为0

    // matrix *s = Matrix_New(973, 1, DOUBLE);
    // matrix *z = Matrix_New(973, 1, DOUBLE);
    matrix *lmbda = Matrix_New(78, 1, DOUBLE);  // 如果不用可忽略

    double sdata[3] = {4.0, 9.0, 16.0};
    double zdata[3] = {1.0, 1.0, 4.0};

    for (int i = 0; i < 3; i++) {
        // ((double *)s->buffer)[i] = sdata[i];
        // ((double *)z->buffer)[i] = zdata[i];
    }

    // print_matrix(s, "s");
    // print_matrix(z, "z");

    matrix *s = load_txt_as_matrix("../tests/c/input_matrix/s.txt");
    matrix *z = load_txt_as_matrix("../tests/c/input_matrix/z.txt");

    scaling result2 = misc_compute_scaling(s, z, lmbda, &dims, NULL);
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    scaling result;
    for (int i = 0; i < 1; ++i)
        result = misc_compute_scaling(s, z, lmbda, &dims, NULL);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec)
                   + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Elapsed time: %.6f seconds\n", elapsed);

    printf("Computed scaling.d = %.4f\n", result.d);
}

void test_misc_solvers() {
    printf("Running test_misc_solvers...\n");

    // 测试 jdot
    test_misc_jdot();

    // 测试计算缩放因子
    test_computing_scale();

    printf("All tests passed!\n");
}