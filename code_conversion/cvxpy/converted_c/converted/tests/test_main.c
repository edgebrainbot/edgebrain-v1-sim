#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cvxopt.h" 
#include "misc.h"
#include <time.h>

extern void base_gemv(
    void *A, matrix *x, matrix *y,
    char trans, void *alpha, void *beta,
    int n, int m, int incx, int incy,
    int offsetx, int offsety, int issp
);

// extern void (*scal_[])(int *, void *, void *, int *);
extern void blas_scal(void*, matrix*, int, int, int);
extern double blas_nrm2(matrix*, int, int, int);
extern number blas_dot(matrix *x, matrix *y, int n, int incx, int incy, int offsetx, int offsety);
extern void blas_tbsv(matrix *A, matrix *x, char uplo, char trans, char diag,
                int n, int k, int ldA, int ix, int oA, int ox);

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
        .shape = {m, n},
        .strides = {1, m},
        .ob_exports = 0
    };

    matrix x = {
        .mat_type = MAT_DENSE,
        .buffer = malloc(sizeof(double) * n),
        .nrows = n,
        .ncols = 1,
        .id = DOUBLE,
        .shape = {n, 1},
        .strides = {1, n},
        .ob_exports = 0
    };

    matrix y = {
        .mat_type = MAT_DENSE,
        .buffer = malloc(sizeof(double) * m),
        .nrows = m,
        .ncols = 1,
        .id = DOUBLE,
        .shape = {m, 1},
        .strides = {1, m},
        .ob_exports = 0
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

void test_scal(){
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
    x.ob_exports = 0;

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

void test_nrm2() {
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

void test_copy() {
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

void test_tbsv() {

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


int main() {


    // 开始计时
    
    // 调用测试函数
    test_scal();
    // test_gemv();
    // test_sp_gemv();
    test_copy();
    test_nrm2();
    test_tbsv();
    test_ddot();

    // 结束计时
    
    return 0;
}
