#include "cvxopt.h"
#include "misc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

matrix* load_txt_as_matrix(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Cannot open file");
        return NULL;
    }

    int rows = 0, cols = 0;
    char line[4096];

    // First pass: determine matrix dimensions
    while (fgets(line, sizeof(line), file)) {
        if (strlen(line) <= 1) continue;
        rows++;
        if (cols == 0) {
            char* token = strtok(line, " \t\n");
            while (token) {
                cols++;
                token = strtok(NULL, " \t\n");
            }
        }
    }

    rewind(file);

    // Allocate matrix and data buffer
    matrix* mat = (matrix*)malloc(sizeof(matrix));
    mat->id = DOUBLE;
    mat->nrows = rows;
    mat->ncols = cols;
    mat->mat_type = MAT_DENSE;
    mat->buffer = calloc(rows * cols, sizeof(double));

    if (!mat->buffer) {
        free(mat);
        fclose(file);
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    // Fill buffer
    int row = 0;
    double* buf = (double*)mat->buffer;
    while (fgets(line, sizeof(line), file) && row < rows) {
        char* token = strtok(line, " \t\n");
        for (int col = 0; col < cols && token; col++) {
            buf[row * cols + col] = atof(token);
            token = strtok(NULL, " \t\n");
        }
        row++;
    }

    fclose(file);
    return mat;
}

spmatrix create_test_spmatrix() 
{
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


void print_matrix(matrix *m) 
{
    double *buf = MAT_BUFD(m);
    printf("Matrix (%dx%d):\n", m->nrows, m->ncols);
    for (int r = 0; r < m->nrows; ++r) {
        for (int c = 0; c < m->ncols; ++c) {
            printf("%.2f ", buf[c * m->nrows + r]);  // column-major
        }
        printf("\n");
    }
}