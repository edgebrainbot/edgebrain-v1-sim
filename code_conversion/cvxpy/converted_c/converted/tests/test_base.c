#define BASE_MODULE

#include "cvxopt.h"
#include "misc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// extern int (*sp_axpy[])(number, void *, void *, int, int, int, void **) ;

// extern int (*sp_gemm[])(char, char, number, void *, void *, number, void *,
    // int, int, int, int, void **, int, int, int);

extern int (*sp_gemv[])(char, int, int, number, void *, int, void *, int,
    number, void *, int) ;

// extern int (*sp_symv[])(char, int, number, ccs *, int, void *, int,
    // number, void *, int) ;

// extern int (*sp_syrk[])(char, char, number, void *, number,
    // void *, int, int, int, int, void **) ;

// Element size array for different types
static int E_SIZE[] = {sizeof(int), sizeof(double), sizeof(double complex)};

// Constants for different types
static number One[] = {{.i=1}, {.d=1.0}, {.z=1.0+0.0*I}};
static number Zero[] = {{.i=0}, {.d=0.0}, {.z=0.0+0.0*I}};

/* val_id: 0 = matrix, 
 */
static int
convert_inum(void *dest, void *val, int val_id, int_t offset)
{
  if (val_id==0) { /* 1x1 matrix */
    switch (MAT_ID(val)) {
    case INT:
      *(int_t *)dest = MAT_BUFI(val)[offset]; return 0;
    case DOUBLE:
      *(int_t *)dest = (int_t)round(MAT_BUFD(val)[offset]); return 0;
    default: 
      fprintf(stderr, "Error: cannot cast argument as integer\n");
      return -1;
    }
  } else { 
      *(int_t *)dest = *(int_t *)val;
      return 0;
  }
}

static int
convert_dnum(void *dest, void *val, int val_id, int_t offset)
{
  if (val_id==0) { /* matrix */
    switch (MAT_ID(val)) {
    case INT:    *(double *)dest = MAT_BUFI(val)[offset]; return 0;
    case DOUBLE: *(double *)dest = MAT_BUFD(val)[offset]; return 0;
    default: 
      fprintf(stderr, "Error: cannot cast argument as double\n");
      return -1;
    }
  } else { /* normal number */
    *(double *)dest = *(double *)val;
    return 0;
  }
}

static int
convert_znum(void *dest, void *val, int val_id, int_t offset)
{
  if (val_id==0) { /* 1x1 matrix */
    switch (MAT_ID(val)) {
    case INT:
#ifndef _MSC_VER
      *(double complex *)dest = MAT_BUFI(val)[offset]; return 0;
#else
      *(_Dcomplex *)dest = _Cbuild((double)MAT_BUFI(val)[offset],0.0); return 0;
#endif
    case DOUBLE:
#ifndef _MSC_VER
      *(double complex *)dest = MAT_BUFD(val)[offset]; return 0;
#else
      *(_Dcomplex *)dest = _Cbuild(MAT_BUFD(val)[offset],0.0); return 0;
#endif
    case COMPLEX:
#ifndef _MSC_VER
      *(double complex *)dest = MAT_BUFZ(val)[offset]; return 0;
#else
      *(_Dcomplex *)dest = MAT_BUFZ(val)[offset]; return 0;
#endif
    default: 
      fprintf(stderr, "Error: unsupported matrix type for complex conversion\n");
      return -1;
    }
  } else { 
    *(double complex *)dest = *(double complex *)val;
    return 0;
  }
}

int (*convert_num[])(void *, void *, int, int_t) = {
    convert_inum, convert_dnum, convert_znum };

extern void daxpy_(int *, void *, void *, int *, void *, int *) ;
extern void zaxpy_(int *, void *, void *, int *, void *, int *) ;

static void i_axpy(int *n, void *a, void *x, int *incx, void *y, int *incy) {
  int i;
  for (i=0; i < *n; i++) {
    ((int_t *)y)[i*(*incy)] += *((int_t *)a)*((int_t *)x)[i*(*incx)];
  }
}

void (*axpy[])(int *, void *, void *, int *, void *, int *) = {
    i_axpy, daxpy_, zaxpy_ };

extern void dscal_(int *, void *, void *, int *) ;
extern void zscal_(int *, void *, void *, int *) ;

/* we dont implement a BLAS iscal */
static void i_scal(int *n, void *a, void *x, int *incx) {
  int i;
  for (i=0; i < *n; i++) {
    ((int_t *)x)[i*(*incx)] *= *((int_t *)a);
  }
}

void (*scal[])(int *, void *, void *, int *) = { i_scal, dscal_, zscal_ };

extern void dgemm_(char *, char *, int *, int *, int *, void *, void *,
    int *, void *, int *, void *, void *, int *) ;
extern void zgemm_(char *, char *, int *, int *, int *, void *, void *,
    int *, void *, int *, void *, void *, int *) ;

/* we dont implement a BLAS igemm */
static void i_gemm(char *transA, char *transB, int *m, int *n, int *k,
    void *alpha, void *A, int *ldA, void *B, int *ldB, void *beta,
    void *C, int *ldC)
{
  int i, j, l;
  for (j=0; j<*n; j++) {
    for (i=0; i<*m; i++) {
      ((int_t *)C)[i+j*(*m)] = 0;
      for (l=0; l<*k; l++)
        ((int_t *)C)[i+j*(*m)]+=((int_t *)A)[i+l*(*m)]*((int_t *)B)[j*(*k)+l];
    }
  }
}

void (*gemm[])(char *, char *, int *, int *, int *, void *, void *, int *,
    void *, int *, void *, void *, int *) = { i_gemm, dgemm_, zgemm_ };

extern void dgemv_(char *, int *, int *, void *, void *, int *, void *,
    int *, void *, void *, int *);
extern void zgemv_(char *, int *, int *, void *, void *, int *, void *,
    int *, void *, void *, int *);
static void (*gemv[])(char *, int *, int *, void *, void *, int *, void *,
    int *, void *, void *, int *) = { NULL, dgemv_, zgemv_ };

extern void dsyrk_(char *, char *, int *, int *, void *, void *,
    int *, void *, void *, int *);
extern void zsyrk_(char *, char *, int *, int *, void *, void *,
    int *, void *, void *, int *);
void (*syrk[])(char *, char *, int *, int *, void *, void *,
    int *, void *, void *, int *) = { NULL, dsyrk_, zsyrk_ };

extern void dsymv_(char *, int *, void *, void *, int *, void *, int *,
    void *, void *, int *);
extern void zsymv_(char *, int *, void *, void *, int *, void *, int *,
    void *, void *, int *);
void (*symv[])(char *, int *, void *, void *, int *, void *, int *,
    void *, void *, int *) = { NULL, dsymv_, zsymv_ };

static void mtx_iabs(void *src, void *dest, int n) {
  int i;
  for (i=0; i<n; i++)
    ((int_t *)dest)[i] = labs(((int_t *)src)[i]);
}

static void mtx_dabs(void *src, void *dest, int n) {
  int i;
  for (i=0; i<n; i++)
    ((double *)dest)[i] = fabs(((double *)src)[i]);
}

static void mtx_zabs(void *src, void *dest, int n) {
  int i;
  for (i=0; i<n; i++)
#ifndef _MSC_VER
    ((double *)dest)[i] = cabs(((double complex *)src)[i]);
#else
    ((double *)dest)[i] = cabs(((_Dcomplex *)src)[i]);
#endif
}

void (*mtx_abs[])(void *, void *, int) = { mtx_iabs, mtx_dabs, mtx_zabs };

// Error handling
#define GEMV_SUCCESS                      0
#define GEMV_ERROR_INVALID_MATRIX        -1
#define GEMV_ERROR_INVALID_TYPE          -2
#define GEMV_ERROR_CONFLICTING_IDS       -3
#define GEMV_ERROR_INVALID_CHAR          -4
#define GEMV_ERROR_ZERO_INT              -5
#define GEMV_ERROR_NEGATIVE_INT          -6
#define GEMV_ERROR_BUFFER_LENGTH         -7
#define GEMV_ERROR_TYPE_CONVERSION       -8
#define GEMV_ERROR_MEMORY                -9


/* General matrix-vector product for sparse and dense matrices.
 *
 * gemv(A, x, y, trans, alpha, beta, m, n, incx, incy, offsetA, offsetx, offsety)
 *
 * PURPOSE
 * If trans is 'N', computes y := alpha*A*x + beta*y.
 * If trans is 'T', computes y := alpha*A^T*x + beta*y.
 * If trans is 'C', computes y := alpha*A^H*x + beta*y.
 * The matrix A is m by n.
 * Returns immediately if n=0 and trans is 'T' or 'C', or if m=0 
 * and trans is 'N'.
 * Computes y := beta*y if n=0, m>0 and trans is 'N', or if m=0, 
 * n>0 and trans is 'T' or 'C'.
 *
 * ARGUMENTS
 * A         'd' or 'z' (sp)matrix
 * x         'd' or 'z' matrix.  Must have the same type as A.
 * y         'd' or 'z' matrix.  Must have the same type as A.
 * trans     'N', 'T' or 'C'
 * alpha     number (int, float or complex).  Complex alpha is only
 *           allowed if A is complex.
 * beta      number (int, float or complex).  Complex beta is only
 *           allowed if A is complex.
 * m         integer.  If negative, the default value is used.
 * n         integer.  If negative, the default value is used.
 * incx      nonzero integer
 * incy      nonzero integer
 * offsetA   nonnegative integer
 * offsetx   nonnegative integer
 * offsety   nonnegative integer
 *
 * This sparse version of GEMV requires that
 *   m <= A.size[0] - (offsetA % A.size[0])
 */

int base_gemv(void *A, matrix *x, matrix *y, char trans, 
              void *alpha_ptr, void *beta_ptr, int m, int n,
              int incx, int incy, int offsetA, int offsetx, int offsety)
{
    number a, b;
    
    int id;
    // Input validation
    if (!is_matrix(A) && !is_spmatrix(A)) {
        return GEMV_ERROR_INVALID_MATRIX;
    }
    if (!is_matrix(x)) {
        return GEMV_ERROR_INVALID_MATRIX;
    }
    if (!is_matrix(y)) {
        return GEMV_ERROR_INVALID_MATRIX;
    }
    
    if (MAT_ID(x) == INT || MAT_ID(y) == INT || X_ID(A) == INT) {
        return GEMV_ERROR_INVALID_TYPE;
    }
    
    if (X_ID(A) != MAT_ID(x) || X_ID(A) != MAT_ID(y)) {
        return GEMV_ERROR_CONFLICTING_IDS;
    }
    
    if (trans != 'N' && trans != 'T' && trans != 'C') {
        return GEMV_ERROR_INVALID_CHAR;
    }
    
    if (incx == 0) {
        return GEMV_ERROR_ZERO_INT;
    }
    if (incy == 0) {
        return GEMV_ERROR_ZERO_INT;
    }
    
    id = MAT_ID(x);
    if (m < 0) m = X_NROWS(A);
    if (n < 0) n = X_NCOLS(A);
    if ((!m && trans == 'N') || (!n && (trans == 'T' || trans == 'C'))) {
        return GEMV_SUCCESS;
    }
    
    if (offsetA < 0) {
        return GEMV_ERROR_NEGATIVE_INT;
    }
    if (n > 0 && m > 0 && offsetA + (n-1)*MAX(1,X_NROWS(A)) + m >
        X_NROWS(A)*X_NCOLS(A)) {
        return GEMV_ERROR_BUFFER_LENGTH;
    }
    
    if (offsetx < 0) {
        return GEMV_ERROR_NEGATIVE_INT;
    }
    if ((trans == 'N' && n > 0 && offsetx + (n-1)*abs(incx) + 1 > MAT_LGT(x)) ||
        ((trans == 'T' || trans == 'C') && m > 0 &&
            offsetx + (m-1)*abs(incx) + 1 > MAT_LGT(x))) {
        return GEMV_ERROR_BUFFER_LENGTH;
    }
    
    if (offsety < 0) {
        return GEMV_ERROR_NEGATIVE_INT;
    }
    int aa = offsetx + (n-1)*abs(incx) + 1;
    int bb = offsety + (m-1)*abs(incy) + 1;
    printf("aa is: %d\n", aa);
    printf("bb is: %d\n", bb);
    printf("MAT_LGT(x): %d\n", MAT_LGT(x));
    printf("MAT_LGT(y): %d\n", MAT_LGT(y));
    fflush(stdout);
    if ((trans == 'N' && offsety + (m-1)*abs(incy) + 1 > MAT_LGT(y)) ||
        ((trans == 'T' || trans == 'C') &&
            offsety + (n-1)*abs(incy) + 1 > MAT_LGT(y))) {
        return GEMV_ERROR_BUFFER_LENGTH;
    }
    
    if (alpha_ptr && convert_num[MAT_ID(x)](&a, alpha_ptr, 1, 0)) {
        return GEMV_ERROR_TYPE_CONVERSION;
    }
    if (beta_ptr && convert_num[MAT_ID(x)](&b, beta_ptr, 1, 0)) {
        return GEMV_ERROR_TYPE_CONVERSION;
    }
    
    if (is_matrix(A)) {
        int ldA = MAX(1,X_NROWS(A));
        if (trans == 'N' && n == 0) {
            scal[id](&m, (beta_ptr ? &b : &Zero[id]), 
                    (unsigned char*)MAT_BUF(y)+offsety*E_SIZE[id], &incy);
        }
        else if ((trans == 'T' || trans == 'C') && m == 0) {
            scal[id](&n, (beta_ptr ? &b : &Zero[id]), 
                    (unsigned char*)MAT_BUF(y)+offsety*E_SIZE[id], &incy);
        }
        else {
            gemv[id](&trans, &m, &n, (alpha_ptr ? &a : &One[id]),
                    (unsigned char*)MAT_BUF(A) + offsetA*E_SIZE[id], &ldA,
                    (unsigned char*)MAT_BUF(x) + offsetx*E_SIZE[id], &incx, 
                    (beta_ptr ? &b : &Zero[id]),
                    (unsigned char*)MAT_BUF(y) + offsety*E_SIZE[id], &incy);
        }
    } else {
        if (sp_gemv[id](trans, m, n, (alpha_ptr ? a : One[id]), ((spmatrix *)A)->obj,
                       offsetA, (unsigned char*)MAT_BUF(x) + offsetx*E_SIZE[id], incx, 
                       (beta_ptr ? b : Zero[id]),
                       (unsigned char*)MAT_BUF(y) + offsety*E_SIZE[id], incy)) {
            return GEMV_ERROR_MEMORY;
        }
    }
    
    return GEMV_SUCCESS;
}



