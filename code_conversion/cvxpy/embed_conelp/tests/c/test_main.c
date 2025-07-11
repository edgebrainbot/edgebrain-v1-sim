#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cvxopt.h" 
#include "misc.h"
#include "coneprog.h"
#include <sys/time.h>
#include <unistd.h> // 包含 getcwd() 声明
#include <limits.h> // 定义 PATH_MAX

int TEST_TIMES = 5;

extern matrix* load_txt_as_matrix(const char* filename);
extern spmatrix* load_spmatrix_from_triplet_file(const char *filename, int nrows, int ncols);
extern ConeLPResult* conelp(matrix* c, void* G, matrix* h, DIMs* dims, void* A, matrix* b, 
                     PrimalStart* primalstart, DualStart* dualstart, const char* kktsolver,
                     Options* custom_options);

extern void print_matrix(matrix *m);
const char mat_c[] = "../tests/c/input_matrix/c.txt";        // For running
const char mat_h[] = "../tests/c/input_matrix/h.txt";
const char spmat_G[] = "../tests/c/input_matrix/G_sp.txt";

// const char mat_c[] = "input_matrix/c.txt";        // For debugging
// const char mat_h[] = "input_matrix/h.txt";
// const char spmat_G[] = "input_matrix/G_sp.txt";

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

    if (!c || !h || !G_sp) {
        fprintf(stderr, "加载矩阵失败\n");
        return;
    }

    matrix *b = Matrix_New(0, 1, DOUBLE);
    matrix *A = SpMatrix_New(0, 136, 0, DOUBLE);

    int sbuf[] = { 10, 6, 20, 16, 10, 3, 6, 6 };
    DIMs dims = {.l = 0, .q_size = 0, .s_size = 8, .q = NULL, .s = sbuf};     
    PrimalStart *primalstart = NULL;
    DualStart *dualstart = NULL;    

    ConeLPResult *result = conelp(c, G_sp, h, &dims, A, b, primalstart, dualstart, "chol", NULL);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    result = conelp(c, G_sp, h, &dims, A, b, primalstart, dualstart, "chol", NULL);
    gettimeofday(&end, NULL);

    double elapsed = (end.tv_sec - start.tv_sec)
                   + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Elapsed time for conelp: %.6f seconds\n", elapsed);
    printf("ConeLPResult status: %s\n", result->status);
    // print_matrix(result->s);
    
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
