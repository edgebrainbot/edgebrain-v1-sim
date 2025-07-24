#define _POSIX_C_SOURCE 199309L
#include "cvxopt.h"
#include "misc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern scaling* misc_compute_scaling(matrix *s, matrix *z, matrix *lmbda, DIMs *dims, int *mnl_ptr);
extern void misc_update_scaling(scaling *W, matrix *lmbda, matrix *s, matrix *z);
extern double misc_jdot(matrix* x, matrix* y, int n, int offsetx, int offsety);
extern void misc_unpack(matrix *x, matrix *y, DIMs *dims, int mnl, int offsetx, int offsety);
extern void misc_pack(matrix *x, matrix *y, DIMs *dims, int mnl, int offsetx, int offsety);
extern void misc_pack2(matrix *x, DIMs *dims, int mnl);
extern matrix* load_txt_as_matrix(const char* filename);
extern void misc_symm(matrix *x, int n, int offset);


extern int TEST_TIMES;

const char filepath1[] = "../tests/c/input_matrix/s.txt";        // For running
const char filepath2[] = "../tests/c/input_matrix/z.txt";

// const char filepath1[] = "input_matrix/s.txt";      // For testing
// const char filepath2[] = "input_matrix/z.txt";

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
    printf("==== Running test_computing_scale ====\n");
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

    matrix *s = load_txt_as_matrix(filepath1);
    matrix *z = load_txt_as_matrix(filepath2);

    scaling *result2 = misc_compute_scaling(s, z, lmbda, &dims, NULL);
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    scaling *result;
    for (int i = 0; i < TEST_TIMES; ++i)
        result = misc_compute_scaling(s, z, lmbda, &dims, NULL);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec)
                   + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Elapsed time: %.6f seconds\n", elapsed);

    printf("Computed scaling.d = %.4f\n", result->d);
}


void test_updating_scale() {
    printf("==== Running test_updating_scale ====\n");
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

    matrix *s = load_txt_as_matrix(filepath1);
    matrix *z = load_txt_as_matrix(filepath2);

    scaling *result = misc_compute_scaling(s, z, lmbda, &dims, NULL);
    misc_update_scaling(result, lmbda, s, z);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < TEST_TIMES; ++i)
        misc_update_scaling(result, lmbda, s, z);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec)
                   + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Elapsed time: %.6f seconds\n", elapsed);

    printf("Updated scaling.d = %.4f\n", result->d);
}


void test_misc_pack() {
    printf("==== Running test_misc_pack ====\n");
    // DIMs setup
    DIMs dims;
    dims.l = 2;
    dims.q_size = 1;
    int q[1] = {2};
    dims.q = q;
    dims.s_size = 1;
    int s[1] = {2}; // 2x2 symmetric matrix -> 3 packed entries
    dims.s = s;

    // Total variables:
    // l = 2
    // q = 2
    // s = 2x2 matrix = 4 full entries, 3 packed
    // total = 2 + 2 + 4 = 8 in x, 2 + 2 + 3 = 7 in y

    double xbuf[8] = {
        1.0, 2.0,     // linear
        3.0, 4.0,     // SOC
        5.0, 8.0,     // SDP full matrix row-major: [5 6; 6 7]
        8.0, 7.0
    };
    double ybuf[7] = {0};

    matrix x = {0, xbuf, 8, 1, 0};
    matrix y = {0, ybuf, 7, 1, 0};

    misc_pack(&x, &y, &dims, 0, 0, 0);

    printf("Output y:\n");
    for (int i = 0; i < 7; i++) {
        printf("y[%d] = %.6f\n", i, ybuf[i]);
    }

    // Expected y:
    // y[0] = 1.0
    // y[1] = 2.0
    // y[2] = 3.0
    // y[3] = 4.0
    // SDP part (packed):
    // unpacked 2x2: [5 6; 6 7]
    // packed: [5, 6, 7], then scale off-diags (6) by sqrt(2)
    // → result after misc_pack: [5/sqrt(2), 6, 7/sqrt(2)], then *sqrt(2)

    double expected[] = {
        1.0, 2.0, 3.0, 4.0,
        5.0,               // diagonal
        8.0 * sqrt(2.0),   // off-diagonal * sqrt(2)
        7.0                // diagonal
    };

    int pass = 1;
    for (int i = 0; i < 7; i++) {
        if (fabs(ybuf[i] - expected[i]) > 1e-6) {
            pass = 0;
            printf("Mismatch at y[%d]: got %.6f, expected %.6f\n", i, ybuf[i], expected[i]);
        }
    }

    printf("Test %s\n", pass ? "PASSED" : "FAILED");
}

void test_misc_pack2() {
    printf("==== Running test_misc_pack2 ====\n");
    DIMs dims;
    dims.l = 1;
    int qval[1] = {1}; dims.q_size = 1; dims.q = qval;
    int sval[1] = {2}; dims.s_size = 1; dims.s = sval;

    int nrow = 6; // 1 (l) + 1 (q) + 4 (2x2)
    int ncol = 1;

    double xbuf[6] = {
        1.0,     // linear
        2.0,     // q
        3.0, 4.0, // SDP full row 1
        4.0, 5.0  // SDP full row 2
    };

    matrix x = {0, xbuf, nrow, ncol, 0};

    misc_pack2(&x, &dims, 0);

    printf("Packed x:\n");
    for (int i = 0; i < 6; i++) {
        printf("x[%d] = %.6f\n", i, xbuf[i]);
    }

    // 验证输出：
    // x[0] = 1.0
    // x[1] = 2.0
    // 原始 SDP: [3 4; 4 5] → packed = [3, 4, 5]（off-diag 4*√2 ≈ 5.656854）
    double expected[] = {
        1.0, 2.0,
        3.0,
        4.0 * sqrt(2.0),
        5.0
    };

    int pass = 1;
    for (int i = 0; i < 5; i++) {
        if (fabs(xbuf[i] - expected[i]) > 1e-6) {
            printf("Mismatch at x[%d]: got %.6f, expected %.6f\n", i, xbuf[i], expected[i]);
            pass = 0;
        }
    }
    printf("Test %s\n", pass ? "PASSED" : "FAILED");
}

void test_misc_unpack() {
    printf("==== Running test_misc_unpack ====\n");
    // dims = l=1, q=[1], s=[2] ⇒ packed x = 1+1+3 = 5, unpacked y = 1+1+4 = 6
    DIMs dims;
    dims.l = 1;
    int q[1] = {1}; dims.q_size = 1; dims.q = q;
    int s[1] = {2}; dims.s_size = 1; dims.s = s;

    double sqrt2 = sqrt(2.0);

    // packed x: [1.0, 2.0, 3.0, 4*sqrt(2), 5.0]
    double xbuf[5] = {
        1.0, 2.0,
        3.0, 4.0 * sqrt2, 5.0  // SDP packed
    };

    double ybuf[6] = {0};  // unpacked target: l+q+4 unpacked s

    matrix x = {0, xbuf, 5, 1, 0};
    matrix y = {0, ybuf, 6, 1, 0};

    misc_unpack(&x, &y, &dims, 0, 0, 0);

    printf("Unpacked y:\n");
    for (int i = 0; i < 6; i++) {
        printf("y[%d] = %.6f\n", i, ybuf[i]);
    }

    // Expected unpacked y:
    // y[0] = 1.0 (l)
    // y[1] = 2.0 (q)
    // SDP = [[3.0, 4.0], [4.0, 5.0]]
    double expected[] = {
        1.0,
        2.0,
        3.0,
        4.0,
        4.0,
        5.0
    };

    int pass = 1;
    for (int i = 0; i < 6; i++) {
        if (fabs(ybuf[i] - expected[i]) > 1e-6) {
            printf("Mismatch at y[%d]: got %.6f, expected %.6f\n", i, ybuf[i], expected[i]);
            pass = 0;
        }
    }

    printf("Test %s\n", pass ? "PASSED" : "FAILED");
}


void test_misc_symm() {
    int n = 3;

    // 构造下三角矩阵（列主序）
    double buf[9] = {
        1.0, 2.0, 4.0,  // 第 0 列：A(0,0), A(1,0), A(2,0)
        0.0, 3.0, 5.0,  // 第 1 列：A(1,1), A(2,1), A(2,2)
        0.0, 0.0, 6.0   // 第 2 列（上三角）：将由函数填充
    };


    matrix X = {
        .nrows = n,
        .ncols = n,
        .buffer = buf,
        .mat_type = MAT_DENSE,
        .id = DOUBLE
    };

    // 调用函数补全对称矩阵
    misc_symm(&X, n, 0);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < TEST_TIMES; ++i)
        misc_symm(&X, n, 0);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec)
                   + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Elapsed time: %.6f seconds\n", elapsed);

    // 手动打印矩阵（列主序）
    printf("Symmetric matrix X:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            // 注意列主序访问方式
            double val = buf[i + j * n];
            printf("%8.4f ", val);
        }
        printf("\n");
    }
}


void test_misc_solvers() {
    printf("==== Running test_misc_solvers ====\n");

    // 测试 jdot
    // test_misc_jdot();

    // 测试计算缩放因子
    test_computing_scale();

    test_updating_scale();

    // test_misc_pack();

    // test_misc_pack2();

    // test_misc_unpack();

    test_misc_symm();

    // printf("All tests passed!\n");
}