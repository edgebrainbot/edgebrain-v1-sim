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

void test_lapack_potrf() {
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

    int n = A.nrows;
    int lda = A.nrows;
    int info;
    char uplo = 'L';  // 使用下三角

    // 调用 LAPACK Cholesky 分解：A ← L（下三角）
    dpotrf_(&uplo, &n, (double *)A.buffer, &lda, &info);

    if (info != 0) {
        printf("dpotrf failed! info = %d\n", info);
        return 1;
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
    clock_t start, end;
    double elapsed;
    start = clock();  // 开始计时
    // 调用 lapack_gesvd：A = U * S * V^T
    lapack_gesvd(&A, &S, 'A', 'A', &U, &Vt,
                 -1, -1, 0, 0, 0, 0, 0, 0, 0);  // 自动计算 m,n,ldX，offset=0

    end = clock();  // 结束计时

    elapsed = (double)(end - start) / CLOCKS_PER_SEC;
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

void test_lapack() {
    test_lapack_potrf();
    test_lapack_gesvd();
}
