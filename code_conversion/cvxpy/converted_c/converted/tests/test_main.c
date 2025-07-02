#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cvxopt.h" 
#include "misc.h"
#include <time.h>

extern void base_gemv( void *A, matrix *x, matrix *y, char trans, void *alpha, void *beta,
    int n, int m, int incx, int incy, int offsetx, int offsety, int issp);
extern void base_syrk(void *A, void *C, char uplo, char trans, void *alpha, void *beta, bool partial);

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
extern void lapack_potrf(matrix* A, char uplo, int n, int ldA, int offsetA);
extern void lapack_gesvd(matrix *A, matrix *S, char jobu, char jobvt, matrix *U, matrix *Vt, int m, int n, 
                int ldA, int ldU, int ldVt, int offsetA, int offsetS, int offsetU, int offsetVt);

extern void* base_sqrt(void* A, int A_type, int A_id);

extern void* base_emul(void* A, void* B, int A_type, int B_type, int A_id, int B_id);
extern void* base_ediv(void* A, void* B, int A_type, int B_type, int A_id, int B_id);

void fill_matrix(matrix *mat, double *data) {
    memcpy(mat->buffer, data, sizeof(double) * mat->nrows * mat->ncols);
}


void test_gemv(){
    int m = 3, n = 2;

    // 初始化数据
    double A_data[] = {
        1.0, 3.0, 5.0,  // 第 1 列
        2.0, 4.0, 6.0   // 第 2 列
    }; // 3x2 矩阵，列主序


    double x_data[] = {2.0, 1.0}; // 2x1 向量
    double y_data[] = {0.0, 0.0, 0.0}; // 3x1 输出向量

    // 分配内存并构造 matrix 结构体
    matrix A = {
        .mat_type = MAT_DENSE,
        .buffer = malloc(sizeof(double) * m * n),
        .nrows = m,
        .ncols = n,
        .id = DOUBLE,
    };

    matrix x = {
        .mat_type = MAT_DENSE,
        .buffer = malloc(sizeof(double) * n),
        .nrows = n,
        .ncols = 1,
        .id = DOUBLE,
    };

    matrix y = {
        .mat_type = MAT_DENSE,
        .buffer = malloc(sizeof(double) * m),
        .nrows = m,
        .ncols = 1,
        .id = DOUBLE,
    };

    fill_matrix(&A, A_data);
    fill_matrix(&x, x_data);
    fill_matrix(&y, y_data);

    double alpha = 1;
    double beta = 0.0;

    base_gemv((void*)&A, &x, &y, 'N',
                &alpha, &beta,
                -1, -1, 1, 1,
                0, 0, 0);

    double *out = (double *)y.buffer;
    printf("y = A * x = \n");
    for (int i = 0; i < m; ++i) 
        printf("%.2f\n", out[i]);
        

    // 清理内存
    free(A.buffer);
    free(x.buffer);
    free(y.buffer);
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


// // A = [
//   1   0   0   0
//   2   3   0   0
//   0   4   0   7
//   0   0   5   0
//   0   0   0   6
// ];

spmatrix create_test_spmatrix() {
    spmatrix A;
    A.mat_type = MAT_SPARSE;
    ccs *data = malloc(sizeof(ccs));

    data->nrows = 5;
    data->ncols = 4;
    data->id = DOUBLE;

    const int nnz = 7;  // 非零元素个数
    data->values = malloc(sizeof(double) * nnz);
    data->colptr = malloc(sizeof(int_t) * (data->ncols + 1));  // 5 entries
    data->rowind = malloc(sizeof(int_t) * nnz);

    // values 按列主序填充
    double vals[] = {
        1.0, 2.0,  // col 0
        3.0, 4.0,  // col 1
        5.0,       // col 2
        7.0, 6.0   // col 3
    };
    int_t row_indices[] = {
        0, 1,  // col 0
        1, 2,  // col 1
        3,     // col 2
        2, 4   // col 3
    };
    int_t colptrs[] = {
        0, 2, 4, 5, 7
    };

    memcpy(data->values, vals, sizeof(double) * nnz);
    memcpy(data->rowind, row_indices, sizeof(int_t) * nnz);
    memcpy(data->colptr, colptrs, sizeof(int_t) * (data->ncols + 1));

    A.obj = data;
    return A;
}

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

void print_matrix(matrix *m) {
    double *buf = MAT_BUFD(m);
    printf("Matrix (%dx%d):\n", m->nrows, m->ncols);
    for (int r = 0; r < m->nrows; ++r) {
        for (int c = 0; c < m->ncols; ++c) {
            printf("%.2f ", buf[c * m->nrows + r]);  // column-major
        }
        printf("\n");
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

void test_ddot(){
    double Abuf[] = {0.5, 1, 2};
    matrix x = { .id = DOUBLE, .buffer = Abuf, .nrows = 3, .ncols = 1, .mat_type = MAT_DENSE };

    // 创建 x 向量: b = A * [1,1,1]^T = [1,5,15]^T
    double xbuf[] = {3.0, 7.0, 10.0};
    matrix y = { .id = DOUBLE, .buffer = xbuf, .nrows = 3, .ncols = 1, .mat_type = MAT_DENSE };

    number a = blas_dot(&x, &y, 3, 1, 1, 0, 0);
    printf("res is: %.2f\n", a.d);
}

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

    // 打印 y 结果
    printf("y = A * x =\n");
    for (int i = 0; i < y.nrows; ++i) {
        printf("%8.3f\n", ((double*)y.buffer)[i]);
    }
}

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


    clock_t start, end;
    double elapsed;
    start = clock();  // 开始计时

    // 调用 base_syrk
    base_syrk(&A, &C, 'L', 'N', &alpha, &beta, false);

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

void test_number_sqrt() {

    number n1 = {.i = 16}; // INT
    number n2 = {.d = 25.0}; // DOUBLE
    number n3 = {.z = 3.0 + 4.0*I}; // COMPLEX

    number* r1 = (number*)base_sqrt(&n1, 1, INT);
    printf("sqrt(16) = %f\n", (float)r1->i);  // 应该是 4.0

    number* r2 = (number*)base_sqrt(&n2, 1, DOUBLE);
    printf("sqrt(25.0) = %f\n", r2->d);  // 应该是 5.0

    number* r3 = (number*)base_sqrt(&n3, 1, COMPLEX);
    printf("sqrt(3+4j) = %f + %fi\n", creal(r3->z), cimag(r3->z));  // 应该是 2 + 1i


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
    if (!ret) {
        fprintf(stderr, "sqrt failed\n");
        return 1;
    }

    // 打印结果
    printf("sqrt matrix:\n");
    for (int i = 0; i < ret->nrows; ++i) {
        for (int j = 0; j < ret->ncols; ++j) {
            int idx = j * ret->nrows + i; // 列主序
            printf("%6.2f ", ((double*)(ret->buffer))[idx]);
        }
        printf("\n");
    }

    // 释放结果
    free(ret->buffer);
    free(ret);
}

void test_base_emul() {
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
    printf("Matrix A:\n");
    print_matrix(A);
    printf("Matrix B:\n");
    print_matrix(B);

    // 调用 base_emul
    matrix* C = (matrix*)base_emul(A, B, 0, 0, DOUBLE, DOUBLE);
    printf("Result of element-wise multiplication (A * B):\n");
    print_matrix(C);

    // 清理内存
    free(A->buffer); free(A);
    free(B->buffer); free(B);
    free(C->buffer); free(C);
}

void test_base_ediv() {
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
    printf("Matrix A:\n");
    print_matrix(A);
    printf("Matrix B:\n");
    print_matrix(B);

    // 调用 base_emul
    matrix* C = (matrix*)base_ediv(A, B, 0, 0, DOUBLE, DOUBLE);
    printf("Result of element-wise multiplication (A * B):\n");
    print_matrix(C);

    // 清理内存
    free(A->buffer); free(A);
    free(B->buffer); free(B);
    free(C->buffer); free(C);
}

void test_all() {
    test_blas_scal();
    test_blas_gemv();
    test_sp_gemv();
    test_blas_copy();
    test_blas_nrm2();
    test_blas_tbsv();
    // test_blas_ddot();
    test_blas_tbmv();
    test_blas_trmm();
    test_blas_trsm();
    test_blas_trsv();
    test_blas_syrk();
    test_blas_gemm();
    test_blas_gemv();
    test_lapack_potrf();
    test_lapack_gesvd();
    test_base_syrk();
}

int main() {

    // 开始计时
    clock_t start, end;
    double elapsed;
    // test();
    // test();
    start = clock();  // 开始计时

    // test_all();  // 执行所有测试
    // test_number_sqrt();  // 测试 number_sqrt
    test_base_emul();  // 测试 base_emul
    test_base_ediv();  // 测试 base_ediv

    // 结束计时
    end = clock();  // 结束计时

    elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("⏱️ Elapsed time: %.6f seconds\n", elapsed);
    
    return 0;
}
