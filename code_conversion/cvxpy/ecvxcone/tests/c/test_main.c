#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cvxopt.h" 
#include "misc.h"
#include "solver.h"
#include <sys/time.h>
#include <unistd.h> // 包含 getcwd() 声明
#include <limits.h> // 定义 PATH_MAX

int TEST_TIMES = 1000;

extern matrix* load_txt_as_matrix(const char* filename);
extern spmatrix* load_spmatrix_from_triplet_file(const char *filename, int nrows, int ncols);
extern ECVXConeContext* ecvxcone_init(matrix *c, void *G, matrix *h, void *A, matrix *b, DIMs *dims, 
                                ECVXConeSettings* settings);
extern void ecvxcone_free(ECVXConeContext *ecvxcone_ctx);
extern int conelp(matrix* c, void* G, matrix* h, void* A, matrix* b, DIMs* dims,
                    ECVXConeSettings* settings, ECVXConeContext* ecvxcone_ctx);

extern void print_matrix(matrix *m);
// const char mat_c[] = "../tests/c/input_matrix/c.txt";        // For running
// const char mat_h[] = "../tests/c/input_matrix/h.txt";
// const char spmat_G[] = "../tests/c/input_matrix/G_sp.txt";

// const char mat_c[] = "input_matrix/c.txt";        // For debugging
// const char mat_h[] = "input_matrix/h.txt";
// const char spmat_G[] = "input_matrix/G_sp.txt";

// const char mat_c[] = "../tests/c/input_matrix/simplified/c.txt";        // For running
// const char mat_h[] = "../tests/c/input_matrix/simplified/h.txt";
// const char spmat_G[] = "../tests/c/input_matrix/simplified/G_sp.txt";

// const char mat_c[] = "input_matrix/simplified/c.txt";        // For debugging
// const char mat_h[] = "input_matrix/simplified/h.txt";
// const char spmat_G[] = "input_matrix/simplified/G_sp.txt";

const char mat_c[] = "../tests/c/input_matrix/dpp/c.txt";        // For running
const char mat_h[] = "../tests/c/input_matrix/dpp/h.txt";
const char spmat_G[] = "../tests/c/input_matrix/dpp/G_sp.txt";

// const char mat_c[] = "input_matrix/dpp/c.txt";        // For debugging
// const char mat_h[] = "input_matrix/dpp/h.txt";
// const char spmat_G[] = "input_matrix/dpp/G_sp.txt";

void test_all() {
    // test_base();
    // test_blas();
    // test_lapack();
    // test_sparse();
    // test_misc_solvers();
    test_conelp();  // 测试 conelp

}

void test_conelp() {
    printf("==== Running test_conelp ====\n");
    // 这里可以添加对 conelp 的测试代码
    // 例如调用 conelp_solve 函数等
    matrix *c = load_txt_as_matrix(mat_c);
    matrix *h = load_txt_as_matrix(mat_h);
    spmatrix *G_sp = load_spmatrix_from_triplet_file(spmat_G, 973, 136);
    // spmatrix *G_sp = load_spmatrix_from_triplet_file(spmat_G, 89, 21);

    if (!c || !h || !G_sp) {
        fprintf(stderr, "加载矩阵失败\n");
        return;
    }

    matrix *b = Matrix_New(0, 1, DOUBLE);
    matrix *A = SpMatrix_New(0, 136, 0, DOUBLE);
    // matrix *A = SpMatrix_New(0, 21, 0, DOUBLE);

    int sbuf[] = { 10, 6, 20, 16, 10, 3, 6, 6 };
    // int sbuf[] = { 4, 2, 8, 1, 2};
    // DIMs dims = {.l = 0, .q_size = 0, .s_size = 5, .q = NULL, .s = sbuf};     
    DIMs dims = {.l = 0, .q_size = 0, .s_size = 8, .q = NULL, .s = sbuf};     
    PrimalStart *primalstart = NULL;
    DualStart *dualstart = NULL;    

    ECVXConeContext *ecvxcone_ctx = ecvxcone_init(c, G_sp, h, A, b, &dims, &ecvxcone_settings);

    int status;
    status = conelp(c, G_sp, h, A, b, &dims, &ecvxcone_settings, ecvxcone_ctx);

    print_matrix_(ecvxcone_ctx->result->x, "Result x");
    // print_matrix_(ecvxcone_ctx->result->s, "Result s");
    // print_matrix_(ecvxcone_ctx->result->y, "Result y");
    // print_matrix_(ecvxcone_ctx->result->z, "Result z");
    printf("ConeLPResult status: %s\n", ecvxcone_ctx->result->status);

    // int test_time = 1;
    int test_time = TEST_TIMES;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < test_time; ++i) {
        status = conelp(c, G_sp, h, A, b, &dims, &ecvxcone_settings, ecvxcone_ctx);
        Matrix_Free(ecvxcone_ctx->result->x);
        // Matrix_Free(ecvxcone_ctx->result->s);
        // Matrix_Free(ecvxcone_ctx->result->y);
        // Matrix_Free(ecvxcone_ctx->result->z);
    }
    gettimeofday(&end, NULL);

    double elapsed = (end.tv_sec - start.tv_sec)
                   + (end.tv_usec - start.tv_usec) / 1e6;
    double avg_time = elapsed / test_time * 1e3;
    printf("Average time per iteration (Running %d times): %.6f milliseconds\n", test_time, avg_time);
    printf("ConeLPResult status: %s\n", ecvxcone_ctx->result->status);

    Matrix_Free(c);
    Matrix_Free(h);
    Matrix_Free(b);
    SpMatrix_Free(A);           // 如果你用的是 dense matrix
    SpMatrix_Free(G_sp);      // 如果 A 是 sparse matrix，这一行换掉 Matrix_Free(A)

    // cvxopt_free(conelp_ctx);
}


void test_matrix_assign() {
    printf("==== Running test_matrix_assign ====\n");
    matrix *src = Matrix_New(4, 4, DOUBLE);
    matrix *dst = Matrix_New(4, 4, DOUBLE);

    // 填充 src 为如下内容（列主序）:
    // [ 1  5  9 13
    //   2  6 10 14
    //   3  7 11 15
    //   4  8 12 16 ]
    double *data = (double*)src->buffer;
    for (int i = 0; i < 16; ++i)
        data[i] = i + 1;

    printf("Source matrix:\n");
    print_matrix(src);

    // 执行拷贝：src 整体赋值给 dst 的 (0:4, 0:4)
    matrix_slice_assign(dst, src, 0, 4, 0, 4);

    printf("\nDestination matrix after assignment:\n");
    print_matrix(dst);

    Matrix_Free(src);
    Matrix_Free(dst);
    return 0;
}

int main() {

    char cwd[PATH_MAX]; // PATH_MAX 是系统定义的最大路径长度
    
    if (getcwd(cwd, sizeof(cwd))) {
        printf("当前工作目录: %s\n", cwd);
    } else {
        perror("getcwd() 错误");
        return 1;
    }

    // 开始计时
    // clock_t start, end;
    // double elapsed;  start = clock();  // 开始计时

    struct timeval start, end;
    gettimeofday(&start, NULL);

    test_all();  // 执行所有测试
    // test_matrix_assign();  // 测试 matrix_assign
    // test_number_sqrt();  // 测试 number_sqrt
    // test_base_emul();  // 测试 base_emul
    // test_base_ediv();  // 测试 base_ediv
    
    // 结束计时
    // end = clock();  // 结束计时

    // elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    // printf("⏱️ Elapsed time: %.6f seconds\n", elapsed);

    gettimeofday(&end, NULL);

    double elapsed = (end.tv_sec - start.tv_sec)
                   + (end.tv_usec - start.tv_usec) / 1e6;
    // printf("Elapsed time: %.6f seconds\n", elapsed);
    
    return 0;
}


