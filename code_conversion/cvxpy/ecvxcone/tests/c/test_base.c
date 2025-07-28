#define _POSIX_C_SOURCE 199309L
#include "cvxopt.h"
#include "misc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

extern void base_gemv( void *A, matrix *x, matrix *y, char trans, void *alpha, void *beta,
                    int n, int m, int incx, int incy, int offsetx, int offsety, int issp);
extern void base_syrk(void *A, void *C, char uplo, char trans, void *alpha, void *beta, bool partial);
extern void* base_sqrt(void* A, int A_type, int A_id);
extern void* base_emul(void* A, void* B, int A_type, int B_type, int A_id, int B_id);
extern void* base_ediv(void* A, void* B, int A_type, int B_type, int A_id, int B_id);
extern void* base_pow(void* A, void* exponent, int A_type, int A_id, int exp_id);
extern void* base_exp(void* A, int A_type, int A_id);

extern void print_matrix(matrix *m);

extern int TEST_TIMES;

void test_base_syrk() {
    printf("==== Running test_base_syrk ====\n");

    // A: 3x2, column-major: A = [1 2; 3 4; 5 6]
    double Abuf[] = {
        1.0, 3.0, 5.0,  // col 0
        2.0, 4.0, 6.0   // col 1
    };
    matrix A = {
        .id = DOUBLE,
        .buffer = Abuf,
        .nrows = 3,
        .ncols = 2,
        .mat_type = MAT_DENSE
    };

    // C: 3x3, 初始为单位矩阵（只填下三角）
    double Cbuf[] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    matrix C = {
        .id = DOUBLE,
        .buffer = Cbuf,
        .nrows = 3,
        .ncols = 3,
        .mat_type = MAT_DENSE
    };

    // alpha, beta
    double alpha = 2.0;
    double beta = 1.0;

    base_syrk(&A, &C, 'L', 'N', &alpha, &beta, false);

    clock_t start, end;
    double elapsed;
    start = clock();  // 开始计时
    // 调用 base_syrk
    for(int i = 0; i < TEST_TIMES; ++i) {  // 多次调用以便测量时间
        base_syrk(&A, &C, 'L', 'N', &alpha, &beta, false);
    }
    end = clock();  // 结束计时
    elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("⏱️ Elapsed time: %.6f seconds\n", elapsed);

    int n = 3;
    for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j)
        printf("%8.3f ", Cbuf[i * n + j]);
    printf("\n");
    }
}

void test_base_emul() {
    printf("==== Running test_base_emul ====\n");
    matrix* A = Matrix_New(2, 3, DOUBLE);
    double* Abuf = MAT_BUFD(A);
    Abuf[0] = 1.0; Abuf[1] = 2.0;  // col 0
    Abuf[2] = 3.0; Abuf[3] = 4.0;  // col 1
    Abuf[4] = 5.0; Abuf[5] = 6.0;  // col 2

    // 构造 B 矩阵
    matrix* B = Matrix_New(2, 3, DOUBLE);
    double* Bbuf = MAT_BUFD(B);
    Bbuf[0] = 6.0; Bbuf[1] = 5.0;
    Bbuf[2] = 4.0; Bbuf[3] = 3.0;
    Bbuf[4] = 2.0; Bbuf[5] = 1.0;

    // 打印输入
    // printf("Matrix A:\n");
    // print_matrix(A);
    // printf("Matrix B:\n");
    // print_matrix(B);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    matrix* C;

    C = (matrix*)base_emul(A, B, 0, 0, DOUBLE, DOUBLE);
    for (size_t i = 0; i < TEST_TIMES; i++)
    {
        // 调用 base_emul
       C = (matrix*)base_emul(A, B, 0, 0, DOUBLE, DOUBLE);
    }

    gettimeofday(&end, NULL);

    double elapsed = (end.tv_sec - start.tv_sec)
                   + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Elapsed time: %.6f seconds\n", elapsed);

    // printf("Result of element-wise multiplication (A * B):\n");
    print_matrix(C);

    // 清理内存
    free(A->buffer); free(A);
    free(B->buffer); free(B);
    free(C->buffer); free(C);
}

void test_base_ediv() {
    printf("==== Running test_base_ediv ====\n");
    matrix* A = Matrix_New(2, 3, DOUBLE);
    double* Abuf = MAT_BUFD(A);
    Abuf[0] = 1.0; Abuf[1] = 2.0;  // col 0
    Abuf[2] = 3.0; Abuf[3] = 4.0;  // col 1
    Abuf[4] = 5.0; Abuf[5] = 6.0;  // col 2

    // 构造 B 矩阵
    matrix* B = Matrix_New(2, 3, DOUBLE);
    double* Bbuf = MAT_BUFD(B);
    Bbuf[0] = 6.0; Bbuf[1] = 5.0;
    Bbuf[2] = 4.0; Bbuf[3] = 3.0;
    Bbuf[4] = 2.0; Bbuf[5] = 1.0;

    // 打印输入
    // printf("Matrix A:\n");
    // print_matrix(A);
    // printf("Matrix B:\n");
    // print_matrix(B);
    struct timeval start, end;
    gettimeofday(&start, NULL);

    matrix* C;

    C = (matrix*)base_ediv(A, B, 0, 0, DOUBLE, DOUBLE);
    for (size_t i = 0; i < TEST_TIMES; i++)
    {
        // 调用 base_emul
       C = (matrix*)base_ediv(A, B, 0, 0, DOUBLE, DOUBLE);
    }

    gettimeofday(&end, NULL);

    double elapsed = (end.tv_sec - start.tv_sec)
                   + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Elapsed time: %.6f seconds\n", elapsed);

    printf("Result of element-wise multiplication (A * B):\n");
    print_matrix(C);

    // 清理内存
    free(A->buffer); free(A);
    free(B->buffer); free(B);
    free(C->buffer); free(C);
}

void test_base_sqrt() {
    printf("==== Running test_base_sqrt ====\n");
    number n1 = {.i = 16}; // INT
    number n2 = {.d = 25.0}; // DOUBLE
    number n3 = {.z = 3.0 + 4.0*I}; // COMPLEX

    number* r1 = (number*)base_sqrt(&n1, 1, INT);
    // printf("sqrt(16) = %f\n", (float)r1->i);  // 应该是 4.0

    number* r2 = (number*)base_sqrt(&n2, 1, DOUBLE);
    // printf("sqrt(25.0) = %f\n", r2->d);  // 应该是 5.0

    number* r3 = (number*)base_sqrt(&n3, 1, COMPLEX);
    // printf("sqrt(3+4j) = %f + %fi\n", creal(r3->z), cimag(r3->z));  // 应该是 2 + 1i


       // 构造矩阵数据 buffer（列主序存储）
    double Abuf[] = {
        1.0, 20.0,   // 第 1 列（按列优先）
        4.0, 16.0   // 第 2 列
    };

    matrix A = {
        .id = DOUBLE,
        .buffer = Abuf,
        .nrows = 2,
        .ncols = 2,
        .mat_type = MAT_DENSE
    };

    // 执行 base_sqrt
    matrix* ret = (matrix*)base_sqrt(&A, 0, DOUBLE);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    matrix* C;

    for (size_t i = 0; i < TEST_TIMES; i++)
    {
        // 调用 base_emul
       ret = (matrix*)base_sqrt(&A, 0, DOUBLE);
    }

    gettimeofday(&end, NULL);

    double elapsed = (end.tv_sec - start.tv_sec)
                   + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Elapsed time: %.6f seconds\n", elapsed);

    if (!ret) {
        fprintf(stderr, "sqrt failed\n");
        return;
    }

    // 打印结果
    // printf("sqrt matrix:\n");
    // for (int i = 0; i < ret->nrows; ++i) {
    //     for (int j = 0; j < ret->ncols; ++j) {
    //         int idx = j * ret->nrows + i; // 列主序
    //         printf("%6.2f ", ((double*)(ret->buffer))[idx]);
    //     }
    //     printf("\n");
    // }

    // 释放结果
    free(ret->buffer);
    free(ret);
}

void test_base_pow() {
    printf("=== Running test_base_pow ===\n");

    // 构造 2x2 矩阵 A，元素为 [1.0, 2.0; 3.0, 4.0]
    double Abuf[] = {
        1.0, 3.0,   // 第一列（列主序）
        2.0, 4.0    // 第二列
    };
    matrix A = {
        .id = DOUBLE,
        .buffer = Abuf,
        .nrows = 2,
        .ncols = 2,
        .mat_type = MAT_DENSE
    };

    // 指数 number：指数为 2.0，计算 A 的平方
    number exponent = {
        .d = -1.0,
    };

    // 调用 base_pow
    matrix* result = (matrix*) base_pow(&A, &exponent, 0, DOUBLE, DOUBLE);
    if (!result) {
        printf("base_pow returned NULL.\n");
        return;
    }

    // 打印结果
    printf("Result of A ** 2.0:\n");
    for (int i = 0; i < result->nrows; ++i) {
        for (int j = 0; j < result->ncols; ++j) {
            double val = MAT_BUFD(result)[i + j * result->nrows];  // 列主序
            printf("%8.3f ", val);
        }
        printf("\n");
    }

    // 释放结果矩阵（你需要实现或替换为你自己的释放函数）
    Matrix_Free(result);
}


void test_base() {
    printf("==== Running test_base ====\n");

    // 测试 base_syrk
    // test_base_syrk();

    // 测试 base_emul
    // test_base_emul();

    // 测试 base_ediv
    // test_base_ediv();

    // 测试 base_sqrt
    // test_base_sqrt(); 

    // test_base_exp();

    test_base_pow();
}