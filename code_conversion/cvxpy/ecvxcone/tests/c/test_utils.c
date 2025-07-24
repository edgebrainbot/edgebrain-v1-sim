#include "cvxopt.h"
#include "misc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int_t i, j;
    double val;
} triplet;

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

void print_spmatrix(const spmatrix *sp) {
    printf("Sparse matrix (%d x %d), printed as dense:\n", sp->obj->nrows, sp->obj->ncols);
    int m = sp->obj->nrows;
    int n = sp->obj->ncols;
    double *dense = calloc(m * n, sizeof(double));

    for (int j = 0; j < n; ++j) {
        for (int idx = sp->obj->colptr[j]; idx < sp->obj->colptr[j + 1]; ++idx) {
            int i = sp->obj->rowind[idx];
            dense[i + j * m] = ((double*)sp->obj->values)[idx];
        }
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%8.4f ", dense[i + j * m]);
        }
        printf("\n");
    }
    free(dense);
}


void print_matrix(matrix *m) 
{
    double *buf = MAT_BUFD(m);
    printf("Matrix (%dx%d):\n", m->nrows, m->ncols);
    for (int r = 0; r < m->nrows; ++r) {
        for (int c = 0; c < m->ncols; ++c) {
            printf("%.8f ", buf[c * m->nrows + r]);  // column-major
        }
        printf("\n");
    }
}

// 生成下三角矩阵
matrix generate_lower_triangular_matrix(int n) 
{
    matrix A;
    A.nrows = n;
    A.ncols = n;
    A.mat_type = MAT_DENSE;
    A.id = DOUBLE;
    A.buffer = malloc(sizeof(double) * n * n);

    if (!A.buffer) {
        fprintf(stderr, "Memory allocation failed in generate_lower_triangular_matrix.\n");
        A.nrows = A.ncols = 0;
        return A;
    }

    double *buf = (double *) A.buffer;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            buf[i + j*n] = (i >= j) ? ((double) rand() / RAND_MAX) : 0.0;

    return A;
}

// 填充随机矩阵 B
matrix fill_random_matrix(int rows, int cols) 
{
    matrix B;
    B.nrows = rows;
    B.ncols = cols;
    B.mat_type = MAT_DENSE;
    B.id = DOUBLE;
    B.buffer = malloc(sizeof(double) * rows * cols);

    if (!B.buffer) {
        fprintf(stderr, "Memory allocation failed in fill_random_matrix.\n");
        B.nrows = B.ncols = 0;
        return B;
    }

    double *buf = (double *) B.buffer;
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            buf[i + j*rows] = (double) rand() / RAND_MAX;

    return B;
}

int compare_triplet(const void *a, const void *b) {
    triplet *t1 = (triplet *)a;
    triplet *t2 = (triplet *)b;
    if (t1->j != t2->j) return (t1->j - t2->j); // sort by column
    return (t1->i - t2->i);                     // then by row
}

spmatrix* load_spmatrix_from_triplet_file(const char *filename, int nrows, int ncols) 
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("fopen");
        return NULL;
    }

    size_t cap = 1024;
    size_t nnz = 0;
    triplet *data = malloc(cap * sizeof(triplet));
    if (!data) return NULL;

    // Step 1: Read i j v triplets
    int_t i, j;
    double v;
    while (fscanf(fp, "%ld %ld %lf", &i, &j, &v) == 3) {
        if (nnz >= cap) {
            cap *= 2;
            data = realloc(data, cap * sizeof(triplet));
        }
        data[nnz++] = (triplet){i, j, v};
    }
    fclose(fp);

    // Step 2: Sort by column-major
    qsort(data, nnz, sizeof(triplet), compare_triplet);

    // Step 3: Allocate CCS
    ccs *A = malloc(sizeof(ccs));
    A->nrows = nrows;
    A->ncols = ncols;
    A->id = DOUBLE; // DOUBLE

    A->values = malloc(nnz * sizeof(double));
    A->rowind = malloc(nnz * sizeof(int_t));
    A->colptr = calloc(ncols + 1, sizeof(int_t));

    double *val = (double *)A->values;
    int_t *rowind = A->rowind;
    int_t *colptr = A->colptr;

    // Step 4: Fill CCS from sorted triplets
    int_t col = 0, idx = 0;
    colptr[0] = 0;
    for (size_t k = 0; k < nnz; ++k) {
        while (col < data[k].j) {
            ++col;
            colptr[col] = idx;
        }
        val[idx] = data[k].val;
        rowind[idx] = data[k].i;
        ++idx;
    }
    while (col < ncols) {
        ++col;
        colptr[col] = idx;
    }

    // Step 5: Wrap into spmatrix
    spmatrix *S = malloc(sizeof(spmatrix));
    S->mat_type = MAT_SPARSE;  // MAT_SPARSE
    S->obj = A;

    free(data);
    return S;
}