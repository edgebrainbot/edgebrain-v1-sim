#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cvxopt.h" 
#include "misc.h"
#include <sys/time.h>
#include <unistd.h> // 包含 getcwd() 声明
#include <limits.h> // 定义 PATH_MAX

int TEST_TIMES = 5;

void test_all() {
    // test_base();
    // test_blas();
    test_lapack();
    // test_sparse();
    // test_misc_solvers();

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
