#define _POSIX_C_SOURCE 199309L
#include "cvxopt.h"
#include "misc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void lapack_potrf(matrix* A, char uplo, int n, int ldA, int offsetA);
extern void lapack_gesvd(matrix *A, matrix *S, char jobu, char jobvt, matrix *U, matrix *Vt, int m, int n, 
                int ldA, int ldU, int ldVt, int offsetA, int offsetS, int offsetU, int offsetVt);
extern void lapack_ormqr(matrix *A, matrix *tau, matrix *C, char side, char trans, 
                int m, int n, int k, int ldA, int ldC, int offsetA, int offsetC);
extern void lapack_geqrf(matrix *A, matrix *tau, int m, int n, int ldA, int offsetA);
extern void lapack_trtrs(matrix *A, matrix *B, char uplo, char trans, char diag, 
                int n, int nrhs, int ldA, int ldB, int oA, int oB);            
extern matrix fill_random_matrix(int rows, int cols);

extern matrix generate_lower_triangular_matrix(int n);

extern int TEST_TIMES;

void test_lapack_potrf() {
    printf("==== Running test_lapack_potrf ====\n");
    // 初始化对称正定矩阵 A（3x3，列主序）
    double Abuf[] = {
         4.0, 12.0, -16.0,
        12.0, 37.0, -43.0,
       -16.0, -43.0, 98.0
    };

    matrix A = {
        .id = DOUBLE,
        .buffer = Abuf,
        .nrows = 3,
        .ncols = 3,
        .mat_type = MAT_DENSE
    };

    double Bbuf[] = {
         4.0, 12.0, -16.0,
        12.0, 37.0, -43.0,
       -16.0, -43.0, 98.0
    };

    matrix B = {
        .id = DOUBLE,
        .buffer = Bbuf,
        .nrows = 3,
        .ncols = 3,
        .mat_type = MAT_DENSE
    };

    int n = A.nrows;
    int lda = A.nrows;
    int info;
    char uplo = 'L';  // 使用下三角

    // 调用 lapack_gesvd：A = U * S * V^T
    
    lapack_potrf(&B, uplo, 3, 3, 0);  // offset=0
    clock_t start, end;
    double elapsed;
    start = clock();  // 开始计时
    lapack_potrf(&A, uplo, 3, 3, 0);  // offset=0
    end = clock();  // 结束计时
    elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("⏱️ Elapsed time: %.6f seconds\n", elapsed);
    
    if (info != 0) {
        printf("dpotrf failed! info = %d\n", info);
        return;
    }

    // 打印结果矩阵（只打印 L，填充对称）
    printf("Cholesky factor L (A = L * L^T):\n");
    for (int i = 0; i < A.nrows; ++i) {
        for (int j = 0; j < A.ncols; ++j) {
            double *buf = (double *)A.buffer;
            double val = (i >= j) ? buf[i + j * A.nrows] : 0.0;
            printf("%8.3f ", val);
        }
        printf("\n");
    }
}

void test_lapack_gesvd() {
    printf("==== Running test_lapack_gesvd ====\n");
    double Abuf[] = {
        1.0, 2.0, 3.0,  // col 0
        4.0, 5.0, 6.0   // col 1
    };
    matrix A = {
        .id = DOUBLE,
        .buffer = Abuf,
        .nrows = 3,
        .ncols = 2,
        .mat_type = MAT_DENSE
    };

    // 奇异值向量 S (min(m,n) = 2)
    double Sbuf[2] = {0};
    matrix S = {
        .id = DOUBLE,
        .buffer = Sbuf,
        .nrows = 2,
        .ncols = 1,
        .mat_type = MAT_DENSE
    };

    // U: 3x3, Vt: 2x2
    double Ubuf[9] = {0};
    double Vtbuf[4] = {0};
    matrix U = {
        .id = DOUBLE,
        .buffer = Ubuf,
        .nrows = 3,
        .ncols = 3,
        .mat_type = MAT_DENSE
    };
    matrix Vt = {
        .id = DOUBLE,
        .buffer = Vtbuf,
        .nrows = 2,
        .ncols = 2,
        .mat_type = MAT_DENSE
    };

    // 调用 lapack_gesvd：A = U * S * V^T
    lapack_gesvd(&A, &S, 'A', 'A', &U, &Vt,
                 -1, -1, 0, 0, 0, 0, 0, 0, 0);  // 自动计算 m,n,ldX，offset=0

    struct timespec start, end;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);  // 获取开始时间

    for(int i = 0; i < TEST_TIMES; ++i) {  // 多次调用以便测量时间
        // 调用 LAPACK SVD：A = U * S * V^T
        lapack_gesvd(&A, &S, 'A', 'A', &U, &Vt,
                     -1, -1, 0, 0, 0, 0, 0, 0, 0);  // 自动计算 m,n,ldX，offset=0
    }

    clock_gettime(CLOCK_MONOTONIC, &end);  // 获取结束时间
    elapsed = (end.tv_sec - start.tv_sec)
            + (end.tv_nsec - start.tv_nsec) / 1e9;  // 计算耗时
    printf("⏱️ Elapsed time: %.6f seconds\n", elapsed);
    
    // 打印奇异值 S
    printf("Singular values:\n");
    for (int i = 0; i < 2; ++i)
        printf("  %.6f\n", Sbuf[i]);

    // 打印左奇异向量 U (列向量形式)
    printf("\nLeft singular vectors U (3x3):\n");
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            printf("%8.4f ", Ubuf[i + j * 3]);
        printf("\n");
    }

    // 打印右奇异向量 V^T (2x2)
    printf("\nRight singular vectors V^T (2x2):\n");
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j)
            printf("%8.4f ", Vtbuf[i + j * 2]);
        printf("\n");
    }
}

void test_lapack_geqrf() {
    printf("==== Running test_lapack_geqrf ====\n");
    int m = 3, n = 2, k = 2;

    // ==== 初始化 A (3x2), 列主序 ====
    double Abuf[6] = {
        1.0, 3.0, 5.0,   // col 0
        2.0, 4.0, 6.0    // col 1
    };
    matrix A = {
        .nrows = m,
        .ncols = n,
        .buffer = Abuf,
        .mat_type = MAT_DENSE,
        .id = DOUBLE
    };

    // ==== 初始化 tau (2,) ====
    double tau_buf[2] = {0.0, 0.0};
    matrix tau = {
        .nrows = k,
        .ncols = 1,
        .buffer = tau_buf,
        .mat_type = MAT_DENSE,
        .id = DOUBLE
    };

    // ==== 调用 lapack_geqrf ====
    lapack_geqrf(&A, &tau, -1, -1, 0, 0);

    // === 直接打印 A 内容 ===
    double *a = (double *) A.buffer;
    printf("QR-factorized A (R upper, reflectors lower):\n");
    for (int i = 0; i < A.nrows; ++i) {
        for (int j = 0; j < A.ncols; ++j) {
            printf("%10.6f ", a[i + j * A.nrows]);  // 列主序访问
        }
        printf("\n");
    }

    // === 打印 tau ===
    printf("tau = [ ");
    for (int i = 0; i < 2; ++i)
        printf("%g ", tau_buf[i]);
    printf("]\n");

}

void test_lapack_ormqr() {
    printf("==== Running test_lapack_ormqr ====\n");

    int m = 3, k = 2, n = 2;

    // ==== 初始化 matrix A (3x2), 列主序 ====
    double Abuf[6] = {
        1.0, 3.0, 5.0,   // col 0
        2.0, 4.0, 6.0    // col 1
    };
    matrix A = {
        .nrows = m,
        .ncols = k,
        .buffer = Abuf,
        .mat_type = MAT_DENSE,
        .id = DOUBLE
    };

    // ==== 初始化 tau (2,) ====
    double tau_buf[2] = {0.0, 0.0};
    matrix tau = {
        .nrows = 2,
        .ncols = 1,
        .buffer = tau_buf,
        .mat_type = MAT_DENSE,
        .id = DOUBLE
    };

    // ==== 初始化 C (3x2)，列主序 ====
    double Cbuf[6] = {
        7.0, 9.0, 0.0,
        8.0, 10.0, 0.0
    };
    matrix C = {
        .nrows = m,
        .ncols = n,
        .buffer = Cbuf,
        .mat_type = MAT_DENSE,
        .id = DOUBLE
    };

    // ==== QR 分解 ====
    lapack_geqrf(&A, &tau, -1, -1, 0, 0);

    // ==== Apply Q to C（左乘 Q）====
    lapack_ormqr(&A, &tau, &C, 'L', 'N', -1, -1, -1, 0, 0, 0, 0);

    // ==== 打印 C 的结果 ====
    double *c = (double *)C.buffer;
    printf("Q @ C =\n");
    for (int i = 0; i < C.nrows; ++i) {
        for (int j = 0; j < C.ncols; ++j) {
            printf("%10.6f ", c[i + j * C.nrows]);  // 列主序访问
        }
        printf("\n");
    }
}

void test_lapack_trtrs() {
    printf("==== Running test_lapack_trtrs ====\n");

    // 解 A X = B，其中 A 为 3x3 下三角，B 为 3x2
    // A = [1 0 0; 2 3 0; 4 5 6]
    // B = [1 2; 3 4; 5 6]
    double Abuf[9] = {
        1.0, 2.0, 4.0,  // col 0
        0.0, 3.0, 5.0,  // col 1
        0.0, 0.0, 6.0   // col 2
    };
    double Bbuf[6] = {
        1.0, 8.0, 32.0,  // col 0 (rhs 1)
        2.0, 13.0, 47.0   // col 1 (rhs 2)
    };

    // 构造 matrix 结构体
    // matrix A = {.nrows = 3, .ncols = 3, .buffer = Abuf, .mat_type = MAT_DENSE, .id = DOUBLE};
    // matrix B = {.nrows = 3, .ncols = 2, .buffer = Bbuf, .mat_type = MAT_DENSE, .id = DOUBLE};

    matrix A = generate_lower_triangular_matrix(600);
    matrix B = fill_random_matrix(600, 100);  // 生成随机矩阵 B (900x2)

    // 调用 LAPACK trtrs 解决 AX = B
    lapack_trtrs(&A, &B, 'L', 'N', 'N', 3, 2, 3, 3, 0, 0);

    struct timespec start, end;

    // 获取开始时间
    clock_gettime(CLOCK_MONOTONIC, &start);

    // 运行你要测量的函数
    for(int i = 0; i < TEST_TIMES; ++i) {  // 多次调用以便测量时间
        // 调用 LAPACK SVD：A = U * S * V^T
        lapack_trtrs(&A, &B, 'L', 'N', 'N', -1, -1, 0, 0, 0, 0);
    }

    // 获取结束时间
    clock_gettime(CLOCK_MONOTONIC, &end);

    // 计算耗时（单位：秒 + 纳秒）
    double elapsed = (end.tv_sec - start.tv_sec)
                   + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Elapsed time: %.6f seconds\n", elapsed);

    // 输出 B 中解 X
    printf("Solution X (overwritten in B):\n");
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            printf("%8.4f ", Bbuf[i + j * 3]);  // 列主序
        }
        printf("\n");
    }

}

void test_lapack() {
    printf("==== Running test_lapack ====\n");
    // test_lapack_potrf();

    // test_lapack_ormqr();

    // test_lapack_gesvd();

    // test_lapack_geqrf();

    test_lapack_trtrs();
}
