#define _POSIX_C_SOURCE 199309L
#include "cvxopt.h"
#include "misc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// extern void (*scal_[])(int *, void *, void *, int *);
extern void blas_scal(void*, matrix*, int, int, int);
extern double blas_nrm2(matrix*, int, int, int);
extern number blas_dot(matrix *x, matrix *y, int n, int incx, int incy, int offsetx, int offsety);
extern void blas_tbsv(matrix *A, matrix *x, char uplo, char trans, char diag,
                    int n, int k, int ldA, int ix, int oA, int ox);
extern void blas_tbmv(matrix *A, matrix *x, char uplo, char trans, char diag, 
                    int n, int k, int ldA, int incx, int offsetA, int offsetx);
extern void blas_trmm(matrix *A, matrix *B, char side, char uplo, char transA, char diag,
                    void* alpha, int m, int n, int ldA, int ldB, int offsetA, int offsetB);
extern void blas_trsm(matrix *A, matrix *B, char side, char uplo, char transA, char diag,
                    void* alpha, int m, int n, int ldA, int ldB, int offsetA, int offsetB);
extern void blas_trsv(matrix *A, matrix *x, char uplo, char trans, char diag, 
                    int n, int ldA, int ix, int offsetA, int offsetx);
extern void blas_syrk(matrix *A, matrix *C, char uplo, char trans, void* alpha, void* beta, 
                    int n, int k, int ldA, int ldC, int offsetA, int offsetC);
extern void blas_gemm(matrix *A, matrix *B, matrix *C, char transA, char transB, 
                    void* alpha, void* beta, int m, int n, int k, int ldA, int ldB, 
                    int ldC, int offsetA, int offsetB, int offsetC);
extern void blas_gemv(matrix *A, matrix *x, matrix *y, char trans,  void* alpha, void* beta, 
                    int m, int n, int ldA, int incx, int incy, int offsetA, int offsetx, int offsety);

extern test_sp_gemv();
extern void print_matrix(matrix *m);

extern int TEST_TIMES;

void test_blas_tbmv() {
    double Abuf[] = {0, 1, 2, 3, 4, 5};
    matrix A = { .id = DOUBLE, .buffer = Abuf, .nrows = 2, .ncols = 3, .mat_type = MAT_DENSE };

    // 创建 x 向量: b = A * [1,1,1]^T = [1,5,15]^T
    double xbuf[] = {1.0, 2.0, 3.0};
    matrix x = { .id = DOUBLE, .buffer = xbuf, .nrows = 3, .ncols = 1, .mat_type = MAT_DENSE };

    int n = 3;
    int k = 1;
    int ldA = 2;
    int incx = 1;
    char uplo = 'U';
    char trans = 'N';
    char diag = 'N';

    blas_tbmv(&A, &x,  uplo, trans, diag, n, k, ldA, incx, 0, 0);

    printf("Result x (should be [1,1,1]):\n");
    for (int i = 0; i < 3; ++i)
        printf("x[%d] = %.2f\n", i, xbuf[i]);
}

void test_blas_scal(){
    int n = 8;
    double alpha_val = 2.0;
    number alpha;
    alpha.d = alpha_val;

    // 分配和初始化一个 double 类型的 matrix
    matrix x;
    x.mat_type = MAT_DENSE;
    x.nrows = n;
    x.ncols = 1;
    x.id = DOUBLE;

    double *data = (double *)malloc(sizeof(double) * n);
    for (int i = 0; i < n; ++i) data[i] = i + 1;

    x.buffer = (void*)data;

    double scaler = 2.1;

    // 打印原始数据
    printf("Original x:\n");
    for (int i = 0; i < n; ++i)
        printf("%f ", data[i]);
    printf("\n");

    // 执行 scaling
    blas_scal((void*)&scaler, &x, -1, 1, 0);  

    // 打印结果
    printf("Scaled x:\n");
    for (int i = 0; i < n; ++i)
        printf("%f ", data[i]);
    printf("\n");
    printf("x.buffer:%p\n", x.buffer);

    // 清理
    free(data);
}

void test_blas_nrm2() {
    double data[] = {3.0, 4.0};
    matrix x;
    x.mat_type = MAT_DENSE;
    x.id = DOUBLE;
    x.buffer = data;
    x.nrows = 2;
    x.ncols = 1;

    double result = blas_nrm2(&x, -1, 1, 0);
    printf("nrm2 = %f (expected 5.0)\n", result);

    // 再测试一个带 offset 的情况
    double data2[] = {1.0, 0.0, 5.0, 12.0};  // offset = 2: [5.0, 12.0] => nrm = 13.0
    matrix x2 = { .id = DOUBLE, .buffer = data2, .nrows = 4, .ncols = 1, .mat_type = MAT_DENSE };
    double result2 = blas_nrm2(&x2, -1, 1, 0);
    printf("nrm2 (with offset) = %f (expected 13.0)\n", result2);
}

void test_blas_trmm() {
        // 创建一个下三角矩阵 A (3x3)，列主序
    // A =
    // [1 0 0]
    // [2 1 0]
    // [3 4 1]
    double Abuf[] = {
        1.0, 2.0, 3.0,  // 第一列
        0.0, 1.0, 4.0,  // 第二列
        0.0, 0.0, 1.0   // 第三列
    };
    matrix A = {
        .id = DOUBLE,
        .buffer = Abuf,
        .nrows = 3,
        .ncols = 3,
        .mat_type = MAT_DENSE
    };

    // 创建矩阵 B (3x2)，列主序
    // B =
    // [1 2]
    // [3 4]
    // [5 6]
    double Bbuf[] = {
        1.0, 3.0, 5.0,  // 第一列
        2.0, 4.0, 6.0   // 第二列
    };
    matrix B = {
        .id = DOUBLE,
        .buffer = Bbuf,
        .nrows = 3,
        .ncols = 2,
        .mat_type = MAT_DENSE
    };

    // 设置 alpha = 1.0
    number alpha;
    alpha.d = 1.0;

    // 调用 trmm_base，计算 B ← A * B
    blas_trmm(&A, &B, 'L', 'L', 'N', 'N', &alpha,
              -1, -1, 0, 0, 0, 0);

    // 打印结果 B
    printf("B := A * B =\n");
    for (int i = 0; i < B.nrows; ++i) {
        for (int j = 0; j < B.ncols; ++j) {
            printf("%8.3f ", ((double*)B.buffer)[i + j * B.nrows]);
        }
        printf("\n");
    }
}

void test_blas_trsm() {
    printf("==== Running test_blas_trsm ====\n");
        // 创建一个下三角矩阵 A (3x3)，列主序
    // A =
    // [1 0 0]
    // [2 1 0]
    // [3 4 1]
    double Abuf[] = {
        1.0, 2.0, 3.0,  // 第一列
        0.0, 1.0, 4.0,  // 第二列
        0.0, 0.0, 1.0   // 第三列
    };
    matrix A = {
        .id = DOUBLE,
        .buffer = Abuf,
        .nrows = 3,
        .ncols = 3,
        .mat_type = MAT_DENSE
    };

    // 创建矩阵 B (3x2)，列主序
    // B =
    // [1 2]
    // [3 4]
    // [5 6]
    double Bbuf[] = {
        1.0, 3.0, 5.0,  // 第一列
        2.0, 4.0, 6.0   // 第二列
    };
    matrix B = {
        .id = DOUBLE,
        .buffer = Bbuf,
        .nrows = 3,
        .ncols = 2,
        .mat_type = MAT_DENSE
    };

    // 设置 alpha = 1.0
    number alpha;
    alpha.d = 1.0;

    // 调用 blas_trsm，计算 B ← A * B
    blas_trsm(&A, &B, 'L', 'L', 'N', 'N', &alpha,
              -1, -1, 0, 0, 0, 0);

    clock_t start, end;
    double elapsed;
    start = clock();  // 开始计时
    // 调用 base_syrk
    for(int i = 0; i < TEST_TIMES; ++i) {  // 多次调用以便测量时间
        // y := alpha * A * x + beta * y
        blas_trsm(&A, &B, 'L', 'L', 'N', 'N', &alpha,
              -1, -1, 0, 0, 0, 0);
    }
    end = clock();  // 结束计时
    elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("⏱️ Elapsed time: %.6f seconds\n", elapsed);
    
    // 打印结果 B
    printf("B := A * B =\n");
    for (int i = 0; i < B.nrows; ++i) {
        for (int j = 0; j < B.ncols; ++j) {
            printf("%8.3f ", ((double*)B.buffer)[i + j * B.nrows]);
        }
        printf("\n");
    }
}

void test_blas_trsv() {
        // 创建一个下三角矩阵 A (3x3)，列主序
    // A =
    // [1 0 0]
    // [2 1 0]
    // [3 4 1]
    double Abuf[] = {
        1.0, 2.0, 3.0,  // 第一列
        0.0, 1.0, 4.0,  // 第二列
        0.0, 0.0, 1.0   // 第三列
    };
    matrix A = {
        .id = DOUBLE,
        .buffer = Abuf,
        .nrows = 3,
        .ncols = 3,
        .mat_type = MAT_DENSE
    };

    // 创建矩阵 B (3x2)，列主序
    // B =
    // [1]
    // [5]
    // [15]
    double Bbuf[] = {
        1.0, 5.0, 14.0,  // 第一列
    };
    matrix x = {
        .id = DOUBLE,
        .buffer = Bbuf,
        .nrows = 3,
        .ncols = 1,
        .mat_type = MAT_DENSE
    };

    // 设置 alpha = 1.0
    number alpha;
    alpha.d = 1.0;

    // 调用 blas_trsm，计算 B ← A * B
    blas_trsv(&A, &x, 'L', 'N', 'N', -1,
              0, 1, 0, 0);

    // 打印结果 B
    printf("B := A * B =\n");
    for (int i = 0; i < x.nrows; ++i) {
        for (int j = 0; j < x.ncols; ++j) {
            printf("%8.3f ", ((double*)x.buffer)[i + j * x.nrows]);
        }
        printf("\n");
    }
}

void test_blas_syrk() {
    // A is 3x2 matrix: (col-major)
    // A =
    // [1 2]
    // [3 4]
    // [5 6]
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

    // C is 3x3 symmetric output matrix (initialize to 0)
    double Cbuf[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};  // 3×3, will store lower triangle (e.g., 'L')
    matrix C = {
        .id = DOUBLE,
        .buffer = Cbuf,
        .nrows = 3,
        .ncols = 3,
        .mat_type = MAT_DENSE
    };

    // alpha = 1.0, beta = 0.0: C := alpha * A * A^T + beta * C
    number alpha, beta;
    alpha.d = 1.0;
    beta.d = 1.0;

    // 调用 blas_syrk：计算 C := A * A^T
    blas_syrk(&A, &C, 'L', 'N', &alpha, &beta,
              -1, -1, 0, 0, 0, 0);  // 默认 m, n, ldA, ldC, offset

    printf("C = A * A^T =\n");
    for (int i = 0; i < C.nrows; ++i) {
        for (int j = 0; j < C.ncols; ++j) {
            double cij;
            if (i >= j)
                cij = ((double*)C.buffer)[i + j * C.nrows]; // 下三角：实际更新过
            else
                cij = ((double*)C.buffer)[j + i * C.nrows]; // 上三角：镜像
            printf("%8.3f ", cij);
        }
        printf("\n");
    }

}

void test_blas_gemm() {
// A: 2x2
    double Abuf[] = {
        1.0, 3.0,  // col 0
        2.0, 4.0   // col 1
    };
    matrix A = {
        .id = DOUBLE,
        .buffer = Abuf,
        .nrows = 2,
        .ncols = 2,
        .mat_type = MAT_DENSE
    };

    // B: 2x3
    double Bbuf[] = {
        1.0, 4.0,   // col 0
        2.0, 5.0,   // col 1
        3.0, 6.0    // col 2
    };
    matrix B = {
        .id = DOUBLE,
        .buffer = Bbuf,
        .nrows = 2,
        .ncols = 3,
        .mat_type = MAT_DENSE
    };

    // C: 2x3, 初始为 0
    double Cbuf[6] = {0, 1, 2, 3, 4, 5};
    matrix C = {
        .id = DOUBLE,
        .buffer = Cbuf,
        .nrows = 2,
        .ncols = 3,
        .mat_type = MAT_DENSE
    };

    // alpha = 1.0, beta = 0.0
    number alpha, beta;
    alpha.d = 2.0;
    beta.d = 1.0;

    // 调用 blas_gemm：C = alpha*A*B + beta*C
    blas_gemm(&A, &B, &C, 'N', 'N', &alpha, &beta,
              -1, -1, -1, 0, 0, 0, 0, 0, 0);  // 默认 m,n,k, ldA/B/C=0, offset=0

    // 打印结果 C
    printf("C = A * B =\n");
    for (int i = 0; i < C.nrows; ++i) {
        for (int j = 0; j < C.ncols; ++j) {
            printf("%8.3f ", ((double*)C.buffer)[i + j * C.nrows]);
        }
        printf("\n");
    }
}

void test_blas_gemv() {
    printf("==== Running test_blas_gemv ====\n");

    // A: 3x2 (col-major): [1 3 5] [2 4 6]
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

    // x: 2x1 vector
    double xbuf[] = {1.0, 2.0};
    matrix x = {
        .id = DOUBLE,
        .buffer = xbuf,
        .nrows = 2,
        .ncols = 1,
        .mat_type = MAT_DENSE
    };

    // y: 3x1 vector, initialized to 0
    double ybuf[] = {1.0, 1.0, 0.0};
    matrix y = {
        .id = DOUBLE,
        .buffer = ybuf,
        .nrows = 3,
        .ncols = 1,
        .mat_type = MAT_DENSE
    };

    number alpha, beta;
    alpha.d = 2.0;
    beta.d = 1.0;


    // y := alpha * A * x + beta * y
    blas_gemv(&A, &x, &y, 'N', &alpha, &beta,
              -1, -1,0, 1, 1, 0, 0, 0);  // 默认 m,n, ldA, incX, incY
    clock_t start, end;
    double elapsed;
    start = clock();  // 开始计时
    // 调用 base_syrk
    for(int i = 0; i < TEST_TIMES; ++i) {  // 多次调用以便测量时间
            // y := alpha * A * x + beta * y
        blas_gemv(&A, &x, &y, 'N', &alpha, &beta,
              -1, -1,0, 1, 1, 0, 0, 0);  // 默认 m,n, ldA, incX, incY
    }
    end = clock();  // 结束计时
    elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("⏱️ Elapsed time: %.6f seconds\n", elapsed);


    // 打印 y 结果
    printf("y = A * x =\n");
    for (int i = 0; i < y.nrows; ++i) {
        printf("%8.3f\n", ((double*)y.buffer)[i]);
    }
}

void test_blas_copy() {
    int nrows = 3, ncols = 1;
    int size = nrows * ncols;

    // 初始化源矩阵 x
    double *xbuf = malloc(sizeof(double) * size);
    for (int i = 0; i < size; ++i) xbuf[i] = (double)(i + 1);  // 1, 2, 3
    xbuf[0]=3;

    matrix x = {.mat_type = 0, .buffer = xbuf, .nrows = nrows, .ncols = ncols, .id = DOUBLE};

    // 初始化目标矩阵 y
    double *ybuf = calloc(size, sizeof(double));
    matrix y = {.mat_type = 0, .buffer = ybuf, .nrows = nrows, .ncols = ncols, .id = DOUBLE};

    printf("Before copy:\n");
    print_matrix(&y);

    // 执行复制: copy x -> y
    blas_copy(&x, &y, -1, 1, 1, 0, 0);

    printf("After copy:\n");
    print_matrix(&y);

    // 清理
    free(xbuf);
    free(ybuf);
}

void test_blas_tbsv() {

    // 创建 A: 带宽 k=1，下三角，3x3，带状存储 ldA = k+1 = 2
    // 存储顺序 (col-major):
    // A[0] = 1 (diag), A[1] = 2 (below diag)
    // A[2] = 3, A[3] = 5
    // A[4] = 6, A[5] = -
    double Abuf[] = {0, 1, 2, 3,4, 5};
    matrix A = { .id = DOUBLE, .buffer = Abuf, .nrows = 2, .ncols = 3, .mat_type = MAT_DENSE };

    // 创建 x 向量: b = A * [1,1,1]^T = [1,5,15]^T
    double xbuf[] = {3.0, 7.0, 10.0};
    matrix x = { .id = DOUBLE, .buffer = xbuf, .nrows = 3, .ncols = 1, .mat_type = MAT_DENSE };

    blas_tbsv(&A, &x, 'U', 'N', 'N', 3, 1, 2, 1, 0, 0);

    printf("Result x (should be [1,1,1]):\n");
    for (int i = 0; i < 3; ++i)
        printf("x[%d] = %.2f\n", i, xbuf[i]);

}

void test_blas_dot(){
    double Abuf[] = {0.5, 1, 2};
    matrix x = { .id = DOUBLE, .buffer = Abuf, .nrows = 3, .ncols = 1, .mat_type = MAT_DENSE };

    // 创建 x 向量: b = A * [1,1,1]^T = [1,5,15]^T
    double xbuf[] = {3.0, 7.0, 10.0};
    matrix y = { .id = DOUBLE, .buffer = xbuf, .nrows = 3, .ncols = 1, .mat_type = MAT_DENSE };

    number a = blas_dot(&x, &y, 3, 1, 1, 0, 0);
    printf("res is: %.2f\n", a.d);
}

void test_blas() {
    printf("==== Running test_blas ====\n");

    // test_blas_scal();
    // test_blas_gemv();
    // test_sp_gemv();
    // test_blas_copy();
    // test_blas_nrm2();
    // test_blas_tbsv();
    // test_blas_dot();
    // test_blas_tbmv();
    // test_blas_trmm();
    test_blas_trsm();
    // test_blas_trsv();
    // test_blas_syrk();
    // test_blas_gemm();
    // test_blas_gemv();
}