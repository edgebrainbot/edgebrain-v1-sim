#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cvxopt.h" 
#include "misc.h"
#include <time.h>

extern int base_gemv(
    matrix *A, matrix *x, matrix *y,
    char trans, void *alpha, void *beta,
    int n, int m, int incx, int incy,
    int offsetx, int offsety, int issp
);

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
        .tag = MATRIX_TYPE_DENSE,
        .buffer = malloc(sizeof(double) * m * n),
        .nrows = m,
        .ncols = n,
        .id = DOUBLE,
        .shape = {m, n},
        .strides = {1, m},
        .ob_exports = 0
    };

    matrix x = {
        .tag = MATRIX_TYPE_DENSE,
        .buffer = malloc(sizeof(double) * n),
        .nrows = n,
        .ncols = 1,
        .id = DOUBLE,
        .shape = {n, 1},
        .strides = {1, n},
        .ob_exports = 0
    };

    matrix y = {
        .tag = MATRIX_TYPE_DENSE,
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

    int ret = base_gemv(&A, &x, &y, 'N',
                        &alpha, &beta,
                        -1, -1, 1, 1,
                        0, 0, 0);

    if (ret != 0) {
        fprintf(stderr, "base_gemv failed with code %d\n", ret);
    } else {
        double *out = (double *)y.buffer;
        printf("y = A * x = \n");
        for (int i = 0; i < m; ++i) {
            printf("%.2f\n", out[i]);
        }
    }

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
    x.nrows = n;
    x.ncols = 1;
    x.id = DOUBLE;
    x.ob_exports = 0;

    double *data = (double *)malloc(sizeof(double) * n);
    for (int i = 0; i < n; ++i) data[i] = i + 1;

    x.buffer = (void*)data;

    double scaler = -2.5;

    // 打印原始数据
    printf("Original x:\n");
    for (int i = 0; i < n; ++i)
        printf("%f ", data[i]);
    printf("\n");

    // 执行 scaling
    int ret = scal((void*)&scaler, &x, -1, 1, 0);  

    // 打印结果
    if (ret == 0) {
        printf("Scaled x:\n");
        for (int i = 0; i < n; ++i)
            printf("%f ", data[i]);
        printf("\n");
        printf("x.buffer:%p\n", x.buffer);
    } else {
        printf("scal failed with error code %d\n", ret);
    }

    // 清理
    free(data);
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
    A.tag = MATRIX_TYPE_SPARSE;
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
        .tag = MATRIX_TYPE_DENSE,
        .nrows = 4,
        .ncols = 1,
        .id = DOUBLE,
        .buffer = xdata
    };

    double ydata[5] = {1.0, 3.0, 0.0, 2.0, 5.0};
    matrix y = {
        .tag = MATRIX_TYPE_DENSE,
        .nrows = 5,
        .ncols = 1,
        .id = DOUBLE,
        .buffer = ydata
    };

    double alpha = 1.0;
    double beta = 1.0;

    start = clock();
    int ret = base_gemv(
        &A, &x, &y, 'N',
        &alpha, &beta,
        -1, -1, 1, 1,
        0, 0, 0  // issp = 1
    );
    end = clock();

    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time used: %.6f seconds\n", cpu_time_used);

    if (ret == 0) {
        printf("Result y = A * x:\n");
        for (int i = 0; i < 5; ++i) {
            printf("y[%d] = %f\n", i, ydata[i]);
        }
    } else {
        printf("base_gemv failed: %d\n", ret);
    }

    // 清理
    free(A.obj->values);
    free(A.obj->rowind);
    free(A.obj->colptr);
    free(A.obj);
}

int main() {


    // 开始计时
    

    // 调用测试函数
    // test_scal();
    // test_gemv();
    test_sp_gemv();

    // 结束计时
    



    return 0;
}
