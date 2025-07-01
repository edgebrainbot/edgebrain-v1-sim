#define BLSA_MODULE

#include "cvxopt.h"
#include "misc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#ifndef _MSC_VER
typedef complex double complex_t;
#else
typedef _Dcomplex complex_t;
#endif

#define USE_CBLAS_ZDOT 0


/* BLAS 1 prototypes */
extern void dswap_(int *n, double *x, int *incx, double *y, int *incy);
extern void zswap_(int *n, complex_t *x, int *incx, complex_t *y,
    int *incy);
extern void dscal_(int *n, double *alpha, double *x, int *incx);
extern void zscal_(int *n, complex_t *alpha, complex_t *x, int *incx);
extern void zdscal_(int *n, double *alpha, complex_t *x, int *incx);
extern void dcopy_(int *n, double *x, int *incx, double *y, int *incy);
extern void zcopy_(int *n, complex_t *x, int *incx, complex_t *y,
    int *incy);
extern void daxpy_(int *n, double *alpha, double *x, int *incx,
    double *y, int *incy);
extern void zaxpy_(int *n, complex_t *alpha, complex_t *x, int *incx,
    complex_t *y, int *incy);
extern double ddot_(int *n, double *x, int *incx, double *y, int *incy);
#if USE_CBLAS_ZDOT
extern void cblas_zdotc_sub(int n, void *x, int incx, void *y,
    int incy, void *result);
extern void cblas_zdotu_sub(int n, void *x, int incx, void *y, int incy,
    void *result);
#endif
extern double dnrm2_(int *n, double *x, int *incx);
extern double dznrm2_(int *n, complex_t *x, int *incx);
extern double dasum_(int *n, double *x, int *incx);
extern double dzasum_(int *n, complex_t *x, int *incx);
extern int idamax_(int *n, double *x, int *incx);
extern int izamax_(int *n, complex_t *x, int *incx);


/* BLAS 2 prototypes */
extern void dgemv_(char* trans, int *m, int *n, double *alpha,
    double *A, int *lda, double *x, int *incx, double *beta, double *y,
    int *incy);
extern void zgemv_(char* trans, int *m, int *n, complex_t *alpha,
    complex_t *A, int *lda, complex_t *x, int *incx, complex_t *beta,
    complex_t *y, int *incy);
extern void dgbmv_(char* trans, int *m, int *n, int *kl, int *ku,
    double *alpha, double *A, int *lda, double *x, int *incx,
    double *beta, double *y,  int *incy);
extern void zgbmv_(char* trans, int *m, int *n, int *kl, int *ku,
    complex_t *alpha, complex_t *A, int *lda, complex_t *x, int *incx,
    complex_t *beta, complex_t *y,  int *incy);
extern void dsymv_(char *uplo, int *n, double *alpha, double *A,
    int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern void zhemv_(char *uplo, int *n, complex_t *alpha, complex_t *A,
    int *lda, complex_t *x, int *incx, complex_t *beta, complex_t *y,
    int *incy);
extern void dsbmv_(char *uplo, int *n, int *k, double *alpha, double *A,
    int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern void zhbmv_(char *uplo, int *n, int *k, complex_t *alpha,
    complex_t *A, int *lda, complex_t *x, int *incx, complex_t *beta,
    complex_t *y, int *incy);
extern void dtrmv_(char *uplo, char *trans, char *diag, int *n,
    double *A, int *lda, double *x, int *incx);
extern void ztrmv_(char *uplo, char *trans, char *diag, int *n,
    complex_t *A, int *lda, complex_t *x, int *incx);
extern void dtbmv_(char *uplo, char *trans, char *diag, int *n, int *k,
    double *A, int *lda, double *x, int *incx);
extern void ztbmv_(char *uplo, char *trans, char *diag, int *n, int *k,
    complex_t *A, int *lda, complex_t *x, int *incx);
extern void dtrsv_(char *uplo, char *trans, char *diag, int *n,
    double *A, int *lda, double *x, int *incx);
extern void ztrsv_(char *uplo, char *trans, char *diag, int *n,
    complex_t *A, int *lda, complex_t *x, int *incx);
extern void dtbsv_(char *uplo, char *trans, char *diag, int *n, int *k,
    double *A, int *lda, double *x, int *incx);
extern void ztbsv_(char *uplo, char *trans, char *diag, int *n, int *k,
    complex_t *A, int *lda, complex_t *x, int *incx);
extern void dger_(int *m, int *n, double *alpha, double *x, int *incx,
    double *y, int *incy, double *A, int *lda);
extern void zgerc_(int *m, int *n, complex_t *alpha, complex_t *x,
    int *incx, complex_t *y, int *incy, complex_t *A, int *lda);
extern void zgeru_(int *m, int *n, complex_t *alpha, complex_t *x,
    int *incx, complex_t *y, int *incy, complex_t *A, int *lda);
extern void dsyr_(char *uplo, int *n, double *alpha, double *x,
    int *incx, double *A, int *lda);
extern void zher_(char *uplo, int *n, double *alpha, complex_t *x,
    int *incx, complex_t *A, int *lda);
extern void dsyr2_(char *uplo, int *n, double *alpha, double *x,
    int *incx, double *y, int *incy, double *A, int *lda);
extern void zher2_(char *uplo, int *n, complex_t *alpha, complex_t *x,
    int *incx, complex_t *y, int *incy, complex_t *A, int *lda);


/* BLAS 3 prototypes */
extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
    double *alpha, double *A, int *lda, double *B, int *ldb,
    double *beta, double *C, int *ldc);
extern void zgemm_(char *transa, char *transb, int *m, int *n, int *k,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    complex_t *beta, complex_t *C, int *ldc);
extern void dsymm_(char *side, char *uplo, int *m, int *n,
    double *alpha, double *A, int *lda, double *B, int *ldb,
    double *beta, double *C, int *ldc);
extern void zsymm_(char *side, char *uplo, int *m, int *n,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    complex_t *beta, complex_t *C, int *ldc);
extern void zhemm_(char *side, char *uplo, int *m, int *n,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    complex_t *beta, complex_t *C, int *ldc);
extern void dsyrk_(char *uplo, char *trans, int *n, int *k,
    double *alpha, double *A, int *lda, double *beta, double *B,
    int *ldb);
extern void zsyrk_(char *uplo, char *trans, int *n, int *k,
    complex_t *alpha, complex_t *A, int *lda, complex_t *beta, complex_t *B,
    int *ldb);
extern void zherk_(char *uplo, char *trans, int *n, int *k,
    double *alpha, complex_t *A, int *lda, double *beta, complex_t *B,
    int *ldb);
extern void dsyr2k_(char *uplo, char *trans, int *n, int *k,
    double *alpha, double *A, int *lda, double *B, int *ldb,
    double *beta, double *C, int *ldc);
extern void zsyr2k_(char *uplo, char *trans, int *n, int *k,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    complex_t *beta, complex_t *C, int *ldc);
extern void zher2k_(char *uplo, char *trans, int *n, int *k,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    double *beta, complex_t *C, int *ldc);
extern void dtrmm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, double *alpha, double *A, int *lda, double *B,
    int *ldb);
extern void ztrmm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, complex_t *alpha, complex_t *A, int *lda, complex_t *B,
    int *ldb);
extern void dtrsm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, double *alpha, double *A, int *lda, double *B,
    int *ldb);
extern void ztrsm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, complex_t *alpha, complex_t *A, int *lda, complex_t *B,
    int *ldb);


// BLAS function prototypes
extern void dgemv_(char* trans, int* m, int* n, double* alpha, 
                   double* A, int* lda, double* x, int* incx, 
                   double* beta, double* y, int* incy);
extern void zgemv_(char* trans, int* m, int* n, double complex* alpha, 
                   double complex* A, int* lda, double complex* x, int* incx, 
                   double complex* beta, double complex* y, int* incy);
extern void dscal_(int* n, double* alpha, double* x, int* incx);
extern void zscal_(int* n, double complex* alpha, double complex* x, int* incx);


// Microsoft Visual C++ complex number support
#ifdef _MSC_VER
#define _Cbuild(r,i) ((double complex)((r) + (i)*I))
#endif

/**
 * Interprets a void* input as either double or complex and stores the value in a `number` struct.
 *
 * Parameters:
 *   src - Pointer to a double or double complex value
 *   a   - Output pointer to the `number` struct
 *   id  - Type identifier (DOUBLE or COMPLEX)
 *
 * Returns:
 *   0 on success, -1 on invalid input or unsupported type
 */
int number_from_raw(const void *src, number *a, int id) {
    if (!src || !a) return -1;

    switch (id) {
        case DOUBLE:
            a->d = *(const double *)src;
            return 0;

        case COMPLEX:
            a->z = *(const double complex *)src;
            return 0;

        default:
            return -1;
    }
}


void blas_copy(matrix *x, matrix *y, int n, int ix, int iy, int ox, int oy)
{
    /* 
        * Copies a vector x to a vector y (y := x).
        *
        * blas_copy(x, y, n, incx, incy, offsetx, offsety)
        *
        * ARGUMENTS
        * x         'd' or 'z' matrix
        *
        * y         'd' or 'z' matrix.  Must have the same type as x.
        *
        * n         integer.  If n<0, the default value of n is used.
        *           The default value is given by
        *           (len(x)>=offsetx+1) ? 1+(len(x)-offsetx-1)/incx : 0
        *
        * incx      nonzero integer
        *
        * incy      nonzero integer
        *
        * offsetx   nonnegative integer
        *
        * offsety   nonnegative integer
    */
    // Set default values
    if (n == -1) n = -1;  // Use -1 as indicator for default
    if (ix == 0) ix = 1;  // Default increment
    if (iy == 0) iy = 1;  // Default increment
    if (ox == -1) ox = 0; // Default offset
    if (oy == -1) oy = 0; // Default offset

    // Input validation
    if (!x) err_mtrx("x");
    if (!y) err_mtrx("y");
    if (MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (ox < 0) err_nn_int("offsetx");
    if (oy < 0) err_nn_int("offsety");

    // Calculate default n if needed
    if (n < 0) n = (len(x) >= ox+1) ? 1+(len(x)-ox-1)/abs(ix) : 0;
    if (n == 0) return; // Nothing to copy

    // Buffer length validation
    if (len(x) < ox+1+(n-1)*abs(ix)) err_buf_len("x");
    if (len(y) < oy+1+(n-1)*abs(iy)) err_buf_len("y");

    // Perform the copy operation based on matrix type
    switch (MAT_ID(x)){
        case DOUBLE:
            dcopy_(&n, MAT_BUFD(x)+ox, &ix, MAT_BUFD(y)+oy, &iy);
            break;

        case COMPLEX:
            zcopy_(&n, MAT_BUFZ(x)+ox, &ix, MAT_BUFZ(y)+oy, &iy);
            break;

        default:
            err_invalid_id;
    }
}


void blas_axpy(matrix *x, matrix *y, number *alpha, int n, int incx, int incy, int offsetx, int offsety)
{
    /* 
        * Constant times a vector plus a vector (y := alpha*x+y).
        *
        * blas_axpy(x, y, alpha=1.0, n=None, incx=1, incy=1, offsetx=0, offsety=0)
        *
        * ARGUMENTS
        * x         'd' or 'z' matrix
        *
        * y         'd' or 'z' matrix.  Must have the same type as x.
        *
        * alpha     number (int, float or complex).  Complex alpha is only
        *           allowed if x is complex.
        *
        * n         integer.  If n<0, the default value of n is used.
        *           The default value is equal to
        *           (len(x)>=offsetx+1) ? 1+(len(x)-offsetx-1)/incx : 0.
        *
        * incx      nonzero integer
        *
        * incy      nonzero integer
        *
        * offsetx   nonnegative integer
        *
        * offsety   nonnegative integer
    */
    number a;
    int ix = incx, iy = incy, ox = offsetx, oy = offsety;

    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (ox < 0) err_nn_int("offsetx");
    if (oy < 0) err_nn_int("offsety");

    if (n < 0) n = (len(x) >= ox+1) ? 1+(len(x)-ox-1)/abs(ix) : 0;
    if (n == 0) return;

    if (len(x) < ox + 1+(n-1)*abs(ix)) err_buf_len("x");
    if (len(y) < oy + 1+(n-1)*abs(iy)) err_buf_len("y");

    if (alpha) a = *alpha;

    switch (MAT_ID(x)){
        case DOUBLE:
            if (!alpha) a.d=1.0;
            daxpy_(&n, &a.d, MAT_BUFD(x)+ox, &ix, MAT_BUFD(y)+oy, &iy);
            break;

        case COMPLEX:
            if (!alpha) a.z=1.0+0.0*I;
            zaxpy_(&n, &a.z, MAT_BUFZ(x)+ox, &ix, MAT_BUFZ(y)+oy, &iy);
            break;

        default:
            err_invalid_id;
    }
}


void blas_scal(void* alpha, matrix* x, int n, int inc, int offset)
{
    /*
        * Scales a vector by a constant (x := alpha*x).
        *
        * blas_scal(alpha, x, n, inc, offset)
        *
        * ARGUMENTS
        * alpha     number (double or double complex).  Complex alpha is only
        *           allowed if x is complex.
        *
        * x         'd' or 'z' matrix
        *
        * n         integer.  If n<0, the default value of n is used.
        *           The default value is equal to
        *           (len(x)>=offset+1) ? 1+(len-offset-1)/inc : 0.  (default = -1)
        *
        * inc       positive integer (default = 1)
        *
        * offset    nonnegative integer (default = 0)
    */                         
    number a;
    int ix = inc, ox = offset;

    // Default parameter handling
    if (n < 0) n = -1;
    if (ix == 0) ix = 1;
    if (ox < 0) ox = 0;

    if (!Matrix_Check(x)) err_mtrx("x");
    if (ix <= 0) err_p_int("inc");
    if (ox < 0) err_nn_int("offset");
    if (n < 0) n = (len(x) >= ox+1) ? 1+(len(x)-ox-1)/ix : 0;
    if (n == 0) return;
    if (len(x) < ox+1+(n-1)*ix) err_buf_len("x");

    switch (MAT_ID(x)){
        case DOUBLE:
            if (number_from_raw(alpha, &a, MAT_ID(x)))
                err_type("alpha");
            dscal_(&n, &a.d, MAT_BUFD(x)+ox, &ix);
	    break;

        case COMPLEX:
            if (!number_from_raw(alpha, &a, DOUBLE))
                zdscal_(&n, &a.d, MAT_BUFZ(x)+ox, &ix);
            else if (!number_from_raw(alpha, &a, COMPLEX))
                zscal_(&n, &a.z, MAT_BUFZ(x)+ox, &ix);
            else
                err_type("alpha");
	    break;

        default:
            err_invalid_id;
    }

}



double blas_nrm2(matrix *x, int n, int inc, int offset)
{
    /* 
        * Returns the Euclidean norm of a vector (returns ||x||_2).
        *
        * blas_nrm2(x, n, inc, offset)
        *
        * ARGUMENTS
        * x         'd' or 'z' matrix
        *
        * n         integer.  If n<0, the default value of n is used.
        *           The default value is equal to
        *           (len(x)>=offsetx+1) ? 1+(len(x)-offsetx-1)/incx : 0. (default = -1)
        *
        * inc       positive integer (default = 1)
        *
        * offset    nonnegative integer (default = 0)
        *
        * Returns 0 if n=0.
    */ 
    int ix = inc, ox = offset;

    // Default parameter handling
    if (n < 0) n = -1;
    if (ix == 0) ix = 1;
    if (ox < 0) ox = 0;

    if (!Matrix_Check(x)) err_mtrx("x");
    if (ix <= 0) err_p_int("incx");
    if (ox < 0) err_nn_int("offsetx");
    if (n < 0) n = (len(x) >= ox+1) ? 1+(len(x)-ox-1)/ix : 0;
    if (n == 0) return 0.0;
    if (len(x) < ox + 1+(n-1)*ix) err_buf_len("x");

    switch (MAT_ID(x)){
        case DOUBLE:
            return dnrm2_(&n, MAT_BUFD(x)+ox, &ix);

        case COMPLEX:
            return dznrm2_(&n, MAT_BUFZ(x)+ox, &ix);

        default:
            err_invalid_id;
    }
    
    return 0.0; // Should never reach here
}



void blas_tbsv(matrix *A, matrix *x, char uplo, char trans, char diag, 
          int n, int k, int ldA, int incx, int offsetA, int offsetx)
{
    /*
        * Solution of a triangular and banded set of equations.
        *
        * blas_tbsv(A, x, uplo, trans, diag, n, k, ldA, incx, offsetA, offsetx)
        *
        * PURPOSE
        * If trans is 'N', computes x := A^{-1}*x.
        * If trans is 'T', computes x := A^{-T}*x.
        * If trans is 'C', computes x := A^{-H}*x.
        * A is banded triangular of order n and with bandwidth k.
        *
        * ARGUMENTS
        * A         'd' or 'z' matrix
        * x         'd' or 'z' matrix.  Must have the same type as A.
        * uplo      'L' or 'U'
        * trans     'N', 'T' or 'C'
        * diag      'N' or 'U'
        * n         nonnegative integer.  If negative, the default value is used.
        * k         nonnegative integer.  If negative, the default value is used.
        * ldA       nonnegative integer.  ldA >= 1+k. If zero the default value is used.
        * incx      nonzero integer
        * offsetA   nonnegative integer
        * offsetx   nonnegative integer
    */
    // Set default values
    if (n < 0) n = -1;
    if (k < 0) k = -1;
    if (ldA == 0) ldA = 0;
    if (incx == 0) incx = 1;
    if (offsetA < 0) offsetA = 0;
    if (offsetx < 0) offsetx = 0;
    
    // Set default character values
    if (uplo == 0) uplo = 'L';
    if (trans == 0) trans = 'N';
    if (diag == 0) diag = 'N';

    // Validate matrix inputs
    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (MAT_ID(A) != MAT_ID(x)) err_conflicting_ids;

    // Validate character parameters
    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (diag != 'N' && diag != 'U') err_char("diag", "'N', 'U'");

    // Validate incx parameter
    if (incx == 0) err_nz_int("incx");

    // Set default values based on matrix dimensions
    if (n < 0) n = A->ncols;
    if (n == 0) return;
    if (k < 0) k = MAX(0, A->nrows-1);

    // Validate and set leading dimension
    if (ldA == 0) ldA = A->nrows;
    if (ldA < k+1) err_ld("ldA");

    // Validate offset parameters
    if (offsetA < 0) err_nn_int("offsetA");
    if (offsetA + (n-1)*ldA + k + 1 > len(A)) err_buf_len("A");
    if (offsetx < 0) err_nn_int("offsetx");
    if (offsetx + (n-1)*abs(incx) + 1 > len(x)) err_buf_len("x");

    // Call appropriate BLAS routine based on matrix type
    switch (MAT_ID(x)){
        case DOUBLE:
            dtbsv_(&uplo, &trans, &diag, &n, &k, MAT_BUFD(A)+offsetA, &ldA,
                MAT_BUFD(x)+offsetx, &incx);
            break;

        case COMPLEX:
            ztbsv_(&uplo, &trans, &diag, &n, &k, MAT_BUFZ(A)+offsetA, &ldA,
                MAT_BUFZ(x)+offsetx, &incx);
            break;

        default:
            err_invalid_id;
    }
}

// Documentation string (preserved as comment)
/*
Returns x^H*y for real or complex x, y.

blas_dot(x, y, n=None, incx=1, incy=1, offsetx=0, offsety=0)

ARGUMENTS
x         'd' or 'z' matrix

y         'd' or 'z' matrix.  Must have the same type as x.

n         integer.  If n<0, the default value of n is used.
          The default value is equal to
          (len(x)>=offsetx+1) ? 1+(len(x)-offsetx-1)/incx : 0.
          If the default value is used, it must be equal to
          len(y)>=offsety+1 ? 1+(len(y)-offsetx-1)/|incy| : 0.

incx      nonzero integer

incy      nonzero integer

offsetx   nonnegative integer

offsety   nonnegative integer

Returns 0 if n=0.
*/

number blas_dot(matrix *x, matrix *y, int n, int incx, int incy, int offsetx, int offsety)
{
    number val;

    int ix = incx;
    int iy = incy;
    int ox = offsetx;
    int oy = offsety;

    // Set default values
    if (n < 0) n = -1;
    if (ix == 0) ix = 1;
    if (iy == 0) iy = 1;
    if (ox < 0) ox = 0;
    if (oy < 0) oy = 0;

    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (ox < 0) err_nn_int("offsetx");
    if (oy < 0) err_nn_int("offsety");

    if (n < 0){
        n = (len(x) >= ox+1) ? 1+(len(x)-ox-1)/abs(ix) : 0;
        if (n != ((len(y) >= oy+1) ? 1+(len(y)-oy-1)/abs(iy) : 0)){
            ERR("arrays have unequal default lengths");
        }
    }

    if (n && len(x) < ox + 1 + (n-1)*abs(ix)) err_buf_len("x");
    if (n && len(y) < oy + 1 + (n-1)*abs(iy)) err_buf_len("y");

    switch (MAT_ID(x)){
        case DOUBLE:
            val.d = (n==0) ? 0.0 : ddot_(&n, MAT_BUFD(x)+ox, &ix,
                MAT_BUFD(y)+oy, &iy);
            return val;

        case COMPLEX:
#ifndef _MSC_VER
	        if (n==0) val.z = 0.0;
#else
	        if (n==0) val.z = _Cbuild(0.0,0.0);
#endif
	        else
#if USE_CBLAS_ZDOT
                cblas_zdotc_sub(n, MAT_BUFZ(x)+ox, ix, MAT_BUFZ(y)+oy,
                    iy, &val.z);
#else
                ix *= 2;
                iy *= 2;
#ifndef _MSC_VER
                val.z = (ddot_(&n, MAT_BUFD(x)+2*ox, &ix,
                    MAT_BUFD(y)+2*oy, &iy) +
                    ddot_(&n, MAT_BUFD(x)+2*ox + 1, &ix,
                    MAT_BUFD(y)+2*oy + 1, &iy)) +
                    I*(ddot_(&n, MAT_BUFD(x)+2*ox, &ix,
                    MAT_BUFD(y)+2*oy + 1, &iy) -
                    ddot_(&n, MAT_BUFD(x)+2*ox + 1, &ix,
                    MAT_BUFD(y)+2*oy, &iy));
#else
                val.z = _Cbuild(ddot_(&n, MAT_BUFD(x)+2*ox, &ix,
				      MAT_BUFD(y)+2*oy, &iy) +
				ddot_(&n, MAT_BUFD(x)+2*ox + 1, &ix,
				      MAT_BUFD(y)+2*oy + 1, &iy),
				ddot_(&n, MAT_BUFD(x)+2*ox, &ix,
				      MAT_BUFD(y)+2*oy + 1, &iy) -
				ddot_(&n, MAT_BUFD(x)+2*ox + 1, &ix,
				      MAT_BUFD(y)+2*oy, &iy));
#endif

#endif
	    return val;

        default:
            err_invalid_id;
    }
}

